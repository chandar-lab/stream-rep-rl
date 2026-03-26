"""
Stream AC (Actor-Critic) in JAX/Flax for continuous control.
No replay buffer -- each transition is used exactly once and discarded.
Ported from https://github.com/mohmdelsayed/streaming-drl

Uses ObGD (Overshooting-bounded Gradient Descent) with eligibility traces
for both actor and critic, following the streaming RL paradigm.
"""
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import random
import time
from dataclasses import dataclass
from functools import partial

import distrax
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb

from networks.sparse_init import sparse_init
from utils.normalization import NormalizeObservation, ScaleReward
from utils.optimizers import obgd_with_traces


# --- Environment wrapper: AddTimeInfo ---

class AddTimeInfo(gym.Wrapper):
    """Appends normalized episode time [-0.5, 0.5] to the observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.epi_time = -0.5
        if hasattr(env, "spec") and env.spec is not None and env.spec.max_episode_steps is not None:
            self.time_limit = env.spec.max_episode_steps
        else:
            self.time_limit = 1000
        obs_shape = env.observation_space.shape
        low = np.concatenate([env.observation_space.low, np.array([-0.5])])
        high = np.concatenate([env.observation_space.high, np.array([0.5])])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = np.concatenate([obs, np.array([self.epi_time])])
        self.epi_time += 1.0 / self.time_limit
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.epi_time = -0.5
        obs = np.concatenate([obs, np.array([self.epi_time])])
        return obs, info


# --- Args ---

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
    exp_class: str = "stream-ac"
    """the class of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = False
    """whether to save model into the log_dir folder"""
    log_dir: str = "logs"
    """the logging directory"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    lr: float = 1.0
    """ObGD learning rate"""
    gamma: float = 0.99
    """the discount factor gamma"""
    lamda: float = 0.8
    """eligibility trace decay"""
    kappa_policy: float = 3.0
    """ObGD overshooting bound for actor"""
    kappa_value: float = 2.0
    """ObGD overshooting bound for critic"""
    entropy_coeff: float = 0.01
    """entropy regularization coefficient"""
    hidden_size: int = 128
    """hidden layer size for actor and critic"""


# --- Network definitions ---

class Actor(nn.Module):
    """Gaussian policy with parameter-free LayerNorm and sparse init."""
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        mu = nn.Dense(self.action_dim, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        pre_std = nn.Dense(self.action_dim, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        std = nn.softplus(pre_std)
        return mu, std


class Critic(nn.Module):
    """V(s) value function with parameter-free LayerNorm and sparse init."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(1, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        return x.squeeze(-1)


# --- Environment factory ---

def make_env(args, seed, run_name):
    def thunk():
        env = gym.make(args.env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = ScaleReward(env, gamma=args.gamma)
        env = NormalizeObservation(env)
        env = AddTimeInfo(env)
        env.action_space.seed(seed)
        return env
    return thunk


# --- CSV logging ---

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


# --- Main ---

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
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Environment setup (single env, not vectorized)
    env = make_env(args, args.seed, run_name)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs, dtype=np.float32)

    # Initialize networks
    actor = Actor(action_dim=action_dim, hidden_size=args.hidden_size)
    critic = Critic(hidden_size=args.hidden_size)

    dummy_obs = jnp.zeros((obs_dim,))
    actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs)

    # Initialize ObGD optimizers (separate for actor and critic)
    actor_tx = obgd_with_traces(lr=args.lr, gamma=args.gamma, lambd=args.lamda, kappa=args.kappa_policy)
    critic_tx = obgd_with_traces(lr=args.lr, gamma=args.gamma, lambd=args.lamda, kappa=args.kappa_value)

    actor_opt_state = actor_tx.init(actor_params)
    critic_opt_state = critic_tx.init(critic_params)

    # JIT-compiled update functions
    @partial(jax.jit, static_argnames=["reset"])
    def update_step(actor_params, actor_opt_state, critic_params, critic_opt_state,
                    obs, action, reward, next_obs, done_mask, reset):
        # 1. Critic forward: V(s), V(s')
        v_s, critic_grads = jax.value_and_grad(
            lambda p: critic.apply(p, obs)
        )(critic_params)
        v_next = critic.apply(critic_params, next_obs)

        # 2. TD error
        delta = reward + args.gamma * v_next * done_mask - v_s

        # 3. ObGD critic update
        critic_updates, new_critic_opt_state = critic_tx.update(
            critic_grads, critic_opt_state, critic_params, delta, reset=reset
        )
        new_critic_params = optax.apply_updates(critic_params, critic_updates)

        # 4. Actor gradient: grad of (log_prob + entropy * sign(delta))
        sign_delta = jnp.sign(jax.lax.stop_gradient(delta))

        def actor_objective(p):
            mu, std = actor.apply(p, obs)
            dist = distrax.Normal(mu, std)
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
            return log_prob + args.entropy_coeff * entropy * sign_delta

        actor_grads = jax.grad(actor_objective)(actor_params)

        # 5. ObGD actor update
        actor_updates, new_actor_opt_state = actor_tx.update(
            actor_grads, actor_opt_state, actor_params, delta, reset=reset
        )
        new_actor_params = optax.apply_updates(actor_params, actor_updates)

        metrics = {
            "td_error": delta,
            "v_value": v_s,
        }
        return new_actor_params, new_actor_opt_state, new_critic_params, new_critic_opt_state, metrics

    # --- Main training loop ---
    all_logs = []
    episodes = 0
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Action selection: sample from Gaussian policy
        obs_jnp = jnp.array(obs)
        mu, std = actor.apply(actor_params, obs_jnp)
        key, action_key = jax.random.split(key)
        dist = distrax.Normal(mu, std)
        action = dist.sample(seed=action_key)
        action = jax.device_get(action)
        action_clipped = np.clip(action, action_low, action_high)

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action_clipped)
        next_obs = np.array(next_obs, dtype=np.float32)

        # Compute done mask (0 if terminal, 1 otherwise)
        # For truncated episodes, done_mask=1.0 so we still bootstrap V(s')
        done_mask = 1.0 - float(terminated)
        reset = bool(terminated or truncated)

        # Stream update: use transition once and discard
        (actor_params, actor_opt_state, critic_params, critic_opt_state, metrics) = update_step(
            actor_params, actor_opt_state, critic_params, critic_opt_state,
            jnp.array(obs), jnp.array(action, dtype=jnp.float32),
            jnp.float32(reward), jnp.array(next_obs),
            jnp.float32(done_mask), reset=reset,
        )

        # Log episode returns
        if "episode" in info:
            episodes += 1
            ep_return = float(info["episode"]["r"])
            ep_length = int(info["episode"]["l"])
            print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
            all_logs.append({"step": global_step, "episodic_return": ep_return})
            wandb.log(
                {
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_length,
                },
                step=global_step,
            )

        if global_step % 1000 == 0 and global_step > 0:
            sps = int(global_step / (time.time() - start_time))
            wandb.log(
                {
                    "losses/td_error": float(metrics["td_error"]),
                    "losses/v_value": float(metrics["v_value"]),
                    "charts/SPS": sps,
                },
                step=global_step,
            )

        # Reset env on episode boundary (non-vectorized envs don't auto-reset)
        if reset:
            next_obs, _ = env.reset()
            next_obs = np.array(next_obs, dtype=np.float32)

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
        model_path = os.path.join(args.log_dir, f"{args.exp_name}.model")
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes([actor_params, critic_params]))
        print(f"model saved to {model_path}")

    env.close()
    wandb.finish()
