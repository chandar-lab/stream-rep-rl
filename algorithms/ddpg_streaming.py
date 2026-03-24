"""
Streaming DDPG in JAX/Flax.
No replay buffer -- each transition is used exactly once and discarded.
Based on ddpg_jax.py from CleanRL.
"""
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
import wandb


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "stream-rl-ablations"
    """the wandb's project name"""
    wandb_entity: str = "None"
    """the entity (team) of wandb's project"""
    exp_class: str = "ddpg-streaming"
    """the class of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the log_dir folder"""
    log_dir: str = "logs"
    """the logging directory"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 1000
    """timestep to start learning (random exploration warmup)"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def flush_csv(logs, log_dir):
    if not logs:
        return
    csv_path = os.path.join(log_dir, "training_data.csv")
    file_exists = os.path.exists(csv_path)
    fieldnames = list(logs[0].keys())
    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="") as csvfile:
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerows(logs)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    # Set up log directory
    args.log_dir = os.path.join(args.log_dir, run_name)
    if "SCRATCH" in os.environ:
        args.log_dir = os.path.join(os.environ["SCRATCH"], "streamRepL", args.log_dir)
        print(f"Using SCRATCH directory for logs: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
        )
    else:
        wandb.init(mode="disabled")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32

    obs, _ = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        terminations: jnp.ndarray,
    ):
        # Compute target Q value
        next_state_actions = actor.apply(
            actor_state.target_params, next_observations
        ).clip(-1, 1)
        qf1_next_target = qf.apply(
            qf1_state.target_params, next_observations, next_state_actions
        ).reshape(-1)
        next_q_value = (
            rewards + (1 - terminations) * args.gamma * qf1_next_target
        ).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: jnp.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(
                qf1_state.params, observations, actor.apply(params, observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, args.tau
            )
        )
        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, args.tau
            )
        )
        return actor_state, qf1_state, actor_loss_value

    all_logs = []
    actor_loss_value = jnp.array(0.0)
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(
                            0, actor.action_scale * args.exploration_noise
                        )[0]
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                ep_return = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                print(
                    f"global_step={global_step}, episodic_return={ep_return}"
                )
                all_logs.append(
                    {"step": global_step, "episodic_return": ep_return}
                )
                wandb.log(
                    {
                        "charts/episodic_return": ep_return,
                        "charts/episodic_length": ep_length,
                    },
                    step=global_step,
                )
                break

        # Handle truncated observations
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Streaming update: use current transition directly (no buffer)
        if global_step >= args.learning_starts:
            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                obs,
                actions,
                real_next_obs,
                rewards.flatten(),
                terminations.flatten(),
            )

            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    obs,
                )

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                wandb.log(
                    {
                        "losses/qf1_loss": float(qf1_loss_value),
                        "losses/qf1_values": float(qf1_a_values),
                        "losses/actor_loss": float(actor_loss_value),
                        "charts/SPS": sps,
                    },
                    step=global_step,
                )
                print("SPS:", sps)

        obs = next_obs

        # Periodically flush CSV logs
        if global_step % 10000 == 0 and all_logs:
            flush_csv(all_logs, args.log_dir)
            all_logs = []

    # Final flush
    if all_logs:
        flush_csv(all_logs, args.log_dir)
    print(f"Logs saved to: {os.path.join(args.log_dir, 'training_data.csv')}")

    if args.save_model:
        model_path = os.path.join(args.log_dir, f"{args.exp_name}.cleanrl_model")
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [actor_state.params, qf1_state.params]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()
    wandb.finish()
