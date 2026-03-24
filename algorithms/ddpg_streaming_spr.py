"""
Streaming DDPG + SPR with Orthogonal Gradient Projection in JAX/Flax.
No replay buffer -- each transition is used exactly once and discarded.
Extends ddpg_streaming.py with Self-Predictive Representations (SPR).
"""
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import random
import time
from dataclasses import dataclass
from functools import partial

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from flax import struct
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
    exp_class: str = "ddpg-streaming-spr"
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
    """target smoothing coefficient for critic/actor"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 1000
    """timestep to start learning (random exploration warmup)"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""

    # SPR specific arguments
    spr_weight: float = 2.0
    """weight for the SPR auxiliary loss"""
    spr_prediction_depth: int = 5
    """prediction horizon K for SPR"""
    spr_tau: float = 0.99
    """EMA coefficient for target encoder"""
    orth_beta: float = 0.99
    """EMA coefficient for gradient momentum in orthogonal projection"""
    spr_projection_dim: int = 128
    """dimension of the projection/prediction heads"""


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


# --- Network definitions ---

class QNetworkSPR(nn.Module):
    """Q-network with a separable encoder for SPR."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        z = self.get_encoder(x)
        x = jnp.concatenate([z, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

    @nn.compact
    def get_encoder(self, x):
        x = nn.Dense(256, name="encoder_dense")(x)
        x = nn.relu(x)
        return x


class TargetEncoder(nn.Module):
    """Target encoder updated via EMA, same architecture as QNetworkSPR encoder."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, name="encoder_dense")(x)
        x = nn.relu(x)
        return x


class TransitionNetwork(nn.Module):
    """Predicts next latent state from (z_t, a_t)."""
    action_dim: int

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        return x


class ProjectionHead(nn.Module):
    """Projects latent to lower-dim space for SPR loss."""
    projection_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.projection_dim)(x)


class PredictionHead(nn.Module):
    """Prediction head on top of projection for SPR."""
    projection_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.projection_dim)(x)


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


# --- Trajectory buffer for SPR ---

@struct.dataclass
class TrajectoryBuffer:
    observations: jnp.ndarray  # (K+1, obs_dim)
    actions: jnp.ndarray  # (K, action_dim)
    ptr: int
    full: bool


def create_trajectory_buffer(K, obs_dim, action_dim):
    return TrajectoryBuffer(
        observations=jnp.zeros((K + 1, obs_dim)),
        actions=jnp.zeros((K, action_dim)),
        ptr=0,
        full=False,
    )


def add_to_trajectory_buffer(buf, obs, action, K):
    """Add a transition to the trajectory buffer (circular)."""
    ptr = buf.ptr
    observations = buf.observations.at[ptr].set(obs)
    actions = buf.actions.at[ptr].set(action)
    new_ptr = (ptr + 1) % (K + 1)
    full = jnp.where(ptr == K, True, buf.full)
    return buf.replace(
        observations=observations,
        actions=actions,
        ptr=new_ptr,
        full=full,
    )


def reset_trajectory_buffer(buf, obs, K, obs_dim, action_dim):
    """Reset trajectory buffer at episode boundary."""
    return TrajectoryBuffer(
        observations=jnp.zeros((K + 1, obs_dim)).at[0].set(obs),
        actions=jnp.zeros((K, action_dim)),
        ptr=1,
        full=False,
    )


# --- Orthogonal gradient projection ---

def orthogonal_gradient_projection(grad_t, momentum_t):
    """Project grad_t orthogonal to momentum_t (EMA of past gradients)."""
    grad_flat, _ = jax.tree_util.tree_flatten(grad_t)
    momentum_flat, _ = jax.tree_util.tree_flatten(momentum_t)
    dot_gm = sum(jnp.sum(g * m) for g, m in zip(grad_flat, momentum_flat))
    dot_mm = sum(jnp.sum(m * m) for m in momentum_flat)
    coeff = dot_gm / (dot_mm + 1e-8)
    return jax.tree.map(lambda g, m: g - coeff * m, grad_t, momentum_t)


# --- SPR loss ---

def cosine_similarity_loss(pred, target):
    """Negative cosine similarity loss."""
    pred_norm = pred / (jnp.linalg.norm(pred) + 1e-8)
    target_norm = target / (jnp.linalg.norm(target) + 1e-8)
    return -jnp.sum(pred_norm * target_norm)


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    K = args.spr_prediction_depth

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
    key, actor_key, qf1_key, trans_key, proj_key, pred_key, tgt_enc_key = (
        jax.random.split(key, 7)
    )

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    obs, _ = envs.reset(seed=args.seed)

    # --- Initialize networks ---
    actor = Actor(
        action_dim=action_dim,
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    qf = QNetworkSPR()
    dummy_action = envs.action_space.sample()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, dummy_action),
        target_params=qf.init(qf1_key, obs, dummy_action),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # SPR auxiliary networks
    transition_net = TransitionNetwork(action_dim=action_dim)
    dummy_z = jnp.zeros((1, 256))
    transition_params = transition_net.init(trans_key, dummy_z, dummy_action.reshape(1, -1))

    projection_head = ProjectionHead(projection_dim=args.spr_projection_dim)
    projection_params = projection_head.init(proj_key, dummy_z)

    prediction_head = PredictionHead(projection_dim=args.spr_projection_dim)
    dummy_proj = jnp.zeros((1, args.spr_projection_dim))
    prediction_params = prediction_head.init(pred_key, dummy_proj)

    target_encoder = TargetEncoder()
    target_encoder_params = target_encoder.init(tgt_enc_key, obs)
    # Initialize target projection with same params
    target_projection_params = projection_params

    # Note: apply functions are called inside jitted update functions,
    # so no need to pre-jit them individually.

    # --- Initialize gradient momentum for orth ---
    # Extract encoder params from qf for momentum tracking
    def get_encoder_params(qf_params):
        """Extract encoder parameters from QNetworkSPR params."""
        return {"params": {"encoder_dense": qf_params["params"]["encoder_dense"]}}

    def set_encoder_grads(full_grads, encoder_grads):
        """Set encoder gradients in the full gradient tree."""
        return {
            "params": {
                **full_grads["params"],
                "encoder_dense": encoder_grads["params"]["encoder_dense"],
            }
        }

    encoder_momentum = jax.tree.map(jnp.zeros_like, get_encoder_params(qf1_state.params))
    transition_momentum = jax.tree.map(jnp.zeros_like, transition_params)
    projection_momentum = jax.tree.map(jnp.zeros_like, projection_params)
    prediction_momentum = jax.tree.map(jnp.zeros_like, prediction_params)

    # --- SPR update function ---

    @partial(jax.jit, static_argnums=())
    def compute_spr_loss_and_grads(
        qf_params,
        transition_params,
        projection_params,
        prediction_params,
        target_encoder_params,
        target_projection_params,
        traj_obs,
        traj_actions,
        traj_ptr,
        traj_full,
    ):
        """Compute SPR loss over trajectory buffer and return gradients."""
        K_local = traj_actions.shape[0]

        def spr_loss_fn(qf_params, transition_params, projection_params, prediction_params):
            # Get current latent from first observation in trajectory
            z_online = qf.apply(qf_params, traj_obs[0:1], method="get_encoder")

            total_loss = jnp.float32(0.0)
            count = jnp.float32(0.0)

            def body_fn(carry, k):
                z_pred, total_loss, count = carry
                # Use dynamic_slice for traced index k
                action_k = jax.lax.dynamic_slice(traj_actions, (k, 0), (1, traj_actions.shape[1]))
                obs_k = jax.lax.dynamic_slice(traj_obs, (k + 1, 0), (1, traj_obs.shape[1]))

                # Predict next latent via transition model
                z_pred = transition_net.apply(
                    transition_params, z_pred, action_k
                )
                # Project and predict
                proj_online = projection_head.apply(projection_params, z_pred)
                pred_online = prediction_head.apply(prediction_params, proj_online)

                # Target: stop-gradient EMA encoder + projection
                z_target = jax.lax.stop_gradient(
                    target_encoder.apply(target_encoder_params, obs_k)
                )
                proj_target = jax.lax.stop_gradient(
                    projection_head.apply(target_projection_params, z_target)
                )

                # Valid mask: only compute loss for steps within the buffer
                valid = jnp.where(traj_full, 1.0, jnp.where(k + 1 < traj_ptr, 1.0, 0.0))
                step_loss = cosine_similarity_loss(
                    pred_online.squeeze(0), proj_target.squeeze(0)
                )
                total_loss = total_loss + valid * step_loss
                count = count + valid
                return (z_pred, total_loss, count), None

            (_, total_loss, count), _ = jax.lax.scan(
                body_fn, (z_online, total_loss, count), jnp.arange(K_local)
            )

            return total_loss / jnp.maximum(count, 1.0)

        loss, (qf_grads, trans_grads, proj_grads, pred_grads) = jax.value_and_grad(
            spr_loss_fn, argnums=(0, 1, 2, 3)
        )(qf_params, transition_params, projection_params, prediction_params)

        return loss, qf_grads, trans_grads, proj_grads, pred_grads

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

        (qf1_loss_value, qf1_a_values), critic_grads = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)

        return qf1_loss_value, qf1_a_values, critic_grads

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
        return actor_state, actor_loss_value

    @jax.jit
    def apply_combined_critic_update(
        qf1_state,
        critic_grads,
        spr_encoder_grads,
        spr_weight,
    ):
        """Combine critic grads with (orthogonalized) SPR encoder grads and apply."""
        # Add SPR encoder gradient to the critic's encoder gradient
        combined_grads = jax.tree.map(
            lambda c, s: c + spr_weight * s,
            critic_grads,
            set_encoder_grads(
                jax.tree.map(jnp.zeros_like, critic_grads),
                spr_encoder_grads,
            ),
        )
        # The non-encoder parts come purely from critic
        combined_encoder = jax.tree.map(
            lambda c, s: c + spr_weight * s,
            critic_grads["params"]["encoder_dense"],
            spr_encoder_grads["params"]["encoder_dense"],
        )
        combined_grads = {
            "params": {
                **critic_grads["params"],
                "encoder_dense": combined_encoder,
            }
        }
        qf1_state = qf1_state.apply_gradients(grads=combined_grads)
        return qf1_state

    @jax.jit
    def update_spr_heads(
        transition_params,
        projection_params,
        prediction_params,
        trans_grads,
        proj_grads,
        pred_grads,
        trans_opt_state,
        proj_opt_state,
        pred_opt_state,
    ):
        """Update SPR auxiliary network parameters."""
        trans_updates, trans_opt_state = trans_optimizer.update(
            trans_grads, trans_opt_state
        )
        transition_params = optax.apply_updates(transition_params, trans_updates)

        proj_updates, proj_opt_state = proj_optimizer.update(
            proj_grads, proj_opt_state
        )
        projection_params = optax.apply_updates(projection_params, proj_updates)

        pred_updates, pred_opt_state = pred_optimizer.update(
            pred_grads, pred_opt_state
        )
        prediction_params = optax.apply_updates(prediction_params, pred_updates)

        return (
            transition_params,
            projection_params,
            prediction_params,
            trans_opt_state,
            proj_opt_state,
            pred_opt_state,
        )

    @jax.jit
    def ema_update_targets(target_encoder_params, target_projection_params, qf_params, projection_params, spr_tau):
        """EMA update for target encoder and target projection."""
        new_encoder = get_encoder_params(qf_params)
        target_encoder_params = optax.incremental_update(
            new_encoder, target_encoder_params, spr_tau
        )
        target_projection_params = optax.incremental_update(
            projection_params, target_projection_params, spr_tau
        )
        return target_encoder_params, target_projection_params

    # SPR head optimizers
    trans_optimizer = optax.adam(learning_rate=args.learning_rate)
    trans_opt_state = trans_optimizer.init(transition_params)
    proj_optimizer = optax.adam(learning_rate=args.learning_rate)
    proj_opt_state = proj_optimizer.init(projection_params)
    pred_optimizer = optax.adam(learning_rate=args.learning_rate)
    pred_opt_state = pred_optimizer.init(prediction_params)

    # Trajectory buffer
    traj_buf = create_trajectory_buffer(K, obs_dim, action_dim)

    all_logs = []
    actor_loss_value = jnp.array(0.0)
    spr_loss_value = jnp.array(0.0)
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

        # Update trajectory buffer
        obs_flat = obs[0] if obs.ndim > 1 else obs
        action_flat = actions[0] if actions.ndim > 1 else actions
        traj_buf = add_to_trajectory_buffer(traj_buf, obs_flat, action_flat, K)

        # Reset trajectory buffer on episode boundary
        episode_ended = bool(terminations[0]) or bool(truncations[0])
        if episode_ended:
            next_obs_flat = real_next_obs[0] if real_next_obs.ndim > 1 else real_next_obs
            traj_buf = reset_trajectory_buffer(traj_buf, next_obs_flat, K, obs_dim, action_dim)

        # Streaming update
        if global_step >= args.learning_starts:
            # 1. Compute critic gradients
            qf1_loss_value, qf1_a_values, critic_grads = update_critic(
                actor_state,
                qf1_state,
                obs,
                actions,
                real_next_obs,
                rewards.flatten(),
                terminations.flatten(),
            )

            # 2. Compute SPR loss and gradients (if trajectory has enough data)
            has_data = bool(traj_buf.full) or int(traj_buf.ptr) > 1
            if has_data:
                spr_loss_value, spr_qf_grads, spr_trans_grads, spr_proj_grads, spr_pred_grads = (
                    compute_spr_loss_and_grads(
                        qf1_state.params,
                        transition_params,
                        projection_params,
                        prediction_params,
                        target_encoder_params,
                        target_projection_params,
                        traj_buf.observations,
                        traj_buf.actions,
                        traj_buf.ptr,
                        traj_buf.full,
                    )
                )

                # 3. Extract encoder grads from SPR and apply orth projection
                spr_encoder_grads = get_encoder_params(spr_qf_grads)
                spr_encoder_grads = orthogonal_gradient_projection(
                    spr_encoder_grads, encoder_momentum
                )

                # Update gradient momentums (EMA)
                encoder_momentum = jax.tree.map(
                    lambda m, g: args.orth_beta * m + (1 - args.orth_beta) * g,
                    encoder_momentum,
                    get_encoder_params(spr_qf_grads),
                )
                transition_momentum = jax.tree.map(
                    lambda m, g: args.orth_beta * m + (1 - args.orth_beta) * g,
                    transition_momentum,
                    spr_trans_grads,
                )
                projection_momentum = jax.tree.map(
                    lambda m, g: args.orth_beta * m + (1 - args.orth_beta) * g,
                    projection_momentum,
                    spr_proj_grads,
                )
                prediction_momentum = jax.tree.map(
                    lambda m, g: args.orth_beta * m + (1 - args.orth_beta) * g,
                    prediction_momentum,
                    spr_pred_grads,
                )

                # Also apply orth to transition/projection/prediction grads
                spr_trans_grads = orthogonal_gradient_projection(
                    spr_trans_grads, transition_momentum
                )
                spr_proj_grads = orthogonal_gradient_projection(
                    spr_proj_grads, projection_momentum
                )
                spr_pred_grads = orthogonal_gradient_projection(
                    spr_pred_grads, prediction_momentum
                )

                # 4. Combine critic encoder grad + SPR encoder grad and apply
                qf1_state = apply_combined_critic_update(
                    qf1_state,
                    critic_grads,
                    spr_encoder_grads,
                    args.spr_weight,
                )

                # 5. Update SPR heads
                (
                    transition_params,
                    projection_params,
                    prediction_params,
                    trans_opt_state,
                    proj_opt_state,
                    pred_opt_state,
                ) = update_spr_heads(
                    transition_params,
                    projection_params,
                    prediction_params,
                    spr_trans_grads,
                    spr_proj_grads,
                    spr_pred_grads,
                    trans_opt_state,
                    proj_opt_state,
                    pred_opt_state,
                )

                # 6. EMA update target encoder and target projection
                target_encoder_params, target_projection_params = ema_update_targets(
                    target_encoder_params,
                    target_projection_params,
                    qf1_state.params,
                    projection_params,
                    args.spr_tau,
                )
            else:
                # Not enough trajectory data yet, just apply critic grads
                qf1_state = qf1_state.apply_gradients(grads=critic_grads)

            # Update actor (delayed)
            if global_step % args.policy_frequency == 0:
                actor_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    obs,
                )
                # EMA update critic target (on actor update steps)
                qf1_state = qf1_state.replace(
                    target_params=optax.incremental_update(
                        qf1_state.params, qf1_state.target_params, args.tau
                    )
                )

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                wandb.log(
                    {
                        "losses/qf1_loss": float(qf1_loss_value),
                        "losses/qf1_values": float(qf1_a_values),
                        "losses/actor_loss": float(actor_loss_value),
                        "losses/spr_loss": float(spr_loss_value),
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
