"""
Stream AC + SPR with Orthogonal Gradient Projection (orth²) in JAX/Flax.
No replay buffer -- each transition is used exactly once and discarded.

Extends stream_ac.py with Self-Predictive Representations (SPR) attached
to the critic encoder. Uses orth (EMA momentum decorrelation) for all SPR
components and orth² (project SPR encoder grad orthogonal to ObGD RL grad)
on the shared critic encoder.
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
from flax import struct
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
    exp_class: str = "stream-ac-spr"
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

    # SPR specific arguments
    spr_weight: float = 2.0
    """weight for the SPR auxiliary loss"""
    spr_prediction_depth: int = 5
    """prediction horizon K for SPR"""
    spr_tau: float = 0.99
    """EMA coefficient for target encoder"""
    orth_beta: float = 0.99
    """EMA coefficient for gradient momentum in orthogonal projection"""
    spr_projection_dim: int = 64
    """dimension of the projection/prediction heads"""
    spr_lr: float = 1e-3
    """Adam learning rate for SPR auxiliary heads"""
    orthogonalize_shared_encoder: bool = True
    """whether to apply orth² (project SPR encoder grad orthogonal to RL encoder grad)"""


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


class CriticEncoder(nn.Module):
    """First layer of critic -> 128-dim latent z. Shared with SPR."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        return x


class CriticHead(nn.Module):
    """Second layer of critic on top of encoder output -> V(s)."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(z)
        z = nn.LayerNorm(use_bias=False, use_scale=False)(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(1, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(z)
        return z.squeeze(-1)


class TargetEncoder(nn.Module):
    """EMA-updated copy of CriticEncoder for stable SPR targets."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=sparse_init(0.9), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.leaky_relu(x)
        return x


class TransitionNetwork(nn.Module):
    """Predicts next latent state from (z_t, a_t)."""
    latent_dim: int = 128

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], -1)
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class ProjectionHead(nn.Module):
    """Projects latent to lower-dim space for SPR loss."""
    projection_dim: int = 64

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.projection_dim)(x)


class PredictionHead(nn.Module):
    """Prediction head on top of projection for SPR."""
    projection_dim: int = 64

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.projection_dim)(x)


# --- Trajectory buffer ---

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
    ptr = buf.ptr
    observations = buf.observations.at[ptr].set(obs)
    actions = buf.actions.at[ptr].set(action)
    new_ptr = (ptr + 1) % (K + 1)
    full = jnp.where(ptr == K, True, buf.full)
    return buf.replace(observations=observations, actions=actions, ptr=new_ptr, full=full)


def reset_trajectory_buffer(buf, obs, K, obs_dim, action_dim):
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


def orthogonal_component_against(update_t, reference_t, epsilon=1e-8):
    """Project update_t orthogonal to reference_t (orth²). Returns (orthogonal_update, cosine_similarity)."""
    update_flat, _ = jax.tree_util.tree_flatten(update_t)
    reference_flat, _ = jax.tree_util.tree_flatten(reference_t)
    dot_ur = sum(jnp.sum(u * r) for u, r in zip(update_flat, reference_flat))
    ref_norm_sq = sum(jnp.sum(r * r) for r in reference_flat)
    update_norm_sq = sum(jnp.sum(u * u) for u in update_flat)
    coeff = dot_ur / (ref_norm_sq + epsilon)
    orth_update = jax.tree.map(lambda u, r: u - coeff * r, update_t, reference_t)
    cosine = dot_ur / (jnp.sqrt((update_norm_sq + epsilon) * (ref_norm_sq + epsilon)) + epsilon)
    return orth_update, jnp.clip(cosine, -1.0, 1.0)


def update_momentum(momentum_t, grad_t, beta):
    """EMA update: c_t = beta * c_{t-1} + (1 - beta) * g_t."""
    return jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, momentum_t, grad_t)


def cosine_similarity_loss(pred, target):
    """Negative cosine similarity loss."""
    pred_norm = pred / (jnp.linalg.norm(pred) + 1e-8)
    target_norm = target / (jnp.linalg.norm(target) + 1e-8)
    return -jnp.sum(pred_norm * target_norm)


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
    key, actor_key, enc_key, head_key, tgt_key, trans_key, proj_key, pred_key = (
        jax.random.split(key, 8)
    )

    # Environment setup (single env, not vectorized)
    env = make_env(args, args.seed, run_name)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs, dtype=np.float32)

    # --- Initialize networks ---
    actor = Actor(action_dim=action_dim, hidden_size=args.hidden_size)
    encoder = CriticEncoder(hidden_size=args.hidden_size)
    head = CriticHead(hidden_size=args.hidden_size)
    target_encoder = TargetEncoder(hidden_size=args.hidden_size)
    transition_net = TransitionNetwork(latent_dim=args.hidden_size)
    projection_head = ProjectionHead(projection_dim=args.spr_projection_dim)
    prediction_head = PredictionHead(projection_dim=args.spr_projection_dim)

    dummy_obs = jnp.zeros((obs_dim,))
    dummy_z = jnp.zeros((args.hidden_size,))
    dummy_action = jnp.zeros((action_dim,))
    dummy_proj = jnp.zeros((args.spr_projection_dim,))

    actor_params = actor.init(actor_key, dummy_obs)
    encoder_params = encoder.init(enc_key, dummy_obs)
    head_params = head.init(head_key, dummy_z)
    target_encoder_params = target_encoder.init(tgt_key, dummy_obs)
    transition_params = transition_net.init(trans_key, dummy_z, dummy_action)
    projection_params = projection_head.init(proj_key, dummy_z)
    prediction_params = prediction_head.init(pred_key, dummy_proj)
    target_projection_params = projection_params  # Initialize target projection same as online

    # --- Initialize ObGD optimizers ---
    actor_tx = obgd_with_traces(lr=args.lr, gamma=args.gamma, lambd=args.lamda, kappa=args.kappa_policy)
    # Critic ObGD operates on combined {encoder, head} params
    critic_tx = obgd_with_traces(lr=args.lr, gamma=args.gamma, lambd=args.lamda, kappa=args.kappa_value)

    actor_opt_state = actor_tx.init(actor_params)
    critic_params = {"encoder": encoder_params, "head": head_params}
    critic_opt_state = critic_tx.init(critic_params)

    # SPR auxiliary head optimizers (Adam)
    trans_optimizer = optax.adam(learning_rate=args.spr_lr)
    trans_opt_state = trans_optimizer.init(transition_params)
    proj_optimizer = optax.adam(learning_rate=args.spr_lr)
    proj_opt_state = proj_optimizer.init(projection_params)
    pred_optimizer = optax.adam(learning_rate=args.spr_lr)
    pred_opt_state = pred_optimizer.init(prediction_params)

    # --- Initialize gradient momentums for orth ---
    encoder_momentum = jax.tree.map(jnp.zeros_like, encoder_params)
    transition_momentum = jax.tree.map(jnp.zeros_like, transition_params)
    projection_momentum = jax.tree.map(jnp.zeros_like, projection_params)
    prediction_momentum = jax.tree.map(jnp.zeros_like, prediction_params)

    # Trajectory buffer
    traj_buf = create_trajectory_buffer(K, obs_dim, action_dim)

    # --- JIT-compiled functions ---

    @partial(jax.jit, static_argnames=["reset"])
    def update_critic_and_actor(actor_params, actor_opt_state, critic_params, critic_opt_state,
                                obs, action, reward, next_obs, done_mask, reset):
        """ObGD update for critic and actor. Returns updates and RL encoder update for orth²."""
        # Critic forward: V(s) = head(encoder(obs))
        def critic_fn(cparams):
            z = encoder.apply(cparams["encoder"], obs)
            return head.apply(cparams["head"], z)

        v_s, critic_grads = jax.value_and_grad(critic_fn)(critic_params)

        # V(s')
        z_next = encoder.apply(critic_params["encoder"], next_obs)
        v_next = head.apply(critic_params["head"], z_next)

        # TD error
        delta = reward + args.gamma * v_next * done_mask - v_s

        # ObGD critic update
        critic_updates, new_critic_opt_state = critic_tx.update(
            critic_grads, critic_opt_state, critic_params, delta, reset=reset
        )

        # Extract RL encoder update before applying (for orth²)
        rl_encoder_update = critic_updates["encoder"]

        # Apply critic updates
        new_critic_params = optax.apply_updates(critic_params, critic_updates)

        # Actor gradient
        sign_delta = jnp.sign(jax.lax.stop_gradient(delta))

        def actor_objective(p):
            mu, std = actor.apply(p, obs)
            dist = distrax.Normal(mu, std)
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
            return log_prob + args.entropy_coeff * entropy * sign_delta

        actor_grads = jax.grad(actor_objective)(actor_params)

        # ObGD actor update
        actor_updates, new_actor_opt_state = actor_tx.update(
            actor_grads, actor_opt_state, actor_params, delta, reset=reset
        )
        new_actor_params = optax.apply_updates(actor_params, actor_updates)

        metrics = {"td_error": delta, "v_value": v_s}
        return (new_actor_params, new_actor_opt_state, new_critic_params, new_critic_opt_state,
                rl_encoder_update, metrics)

    @jax.jit
    def compute_spr_loss_and_grads(encoder_params, transition_params, projection_params,
                                    prediction_params, target_encoder_params,
                                    target_projection_params, traj_obs, traj_actions,
                                    traj_ptr, traj_full):
        """Compute SPR loss and gradients w.r.t. encoder, transition, projection, prediction."""

        def spr_loss_fn(enc_p, trans_p, proj_p, pred_p):
            # Encode first observation in trajectory
            z_curr = encoder.apply(enc_p, traj_obs[0:1])

            total_loss = jnp.float32(0.0)
            count = jnp.float32(0.0)

            def body_fn(carry, k):
                z_pred, total_loss, count = carry
                action_k = jax.lax.dynamic_slice(traj_actions, (k, 0), (1, traj_actions.shape[1]))
                obs_k = jax.lax.dynamic_slice(traj_obs, (k + 1, 0), (1, traj_obs.shape[1]))

                # Predict next latent via transition model
                z_pred = transition_net.apply(trans_p, z_pred, action_k)

                # Project and predict (online path)
                proj_online = projection_head.apply(proj_p, z_pred)
                pred_online = prediction_head.apply(pred_p, proj_online)

                # Target: stop-gradient EMA encoder + projection
                z_target = jax.lax.stop_gradient(target_encoder.apply(target_encoder_params, obs_k))
                proj_target = jax.lax.stop_gradient(
                    projection_head.apply(target_projection_params, z_target)
                )

                # Valid mask
                valid = jnp.where(traj_full, 1.0, jnp.where(k + 1 < traj_ptr, 1.0, 0.0))
                step_loss = cosine_similarity_loss(pred_online.squeeze(0), proj_target.squeeze(0))
                total_loss = total_loss + valid * step_loss
                count = count + valid
                return (z_pred, total_loss, count), None

            (_, total_loss, count), _ = jax.lax.scan(
                body_fn, (z_curr, total_loss, count), jnp.arange(K)
            )
            return total_loss / jnp.maximum(count, 1.0)

        loss, (enc_grads, trans_grads, proj_grads, pred_grads) = jax.value_and_grad(
            spr_loss_fn, argnums=(0, 1, 2, 3)
        )(encoder_params, transition_params, projection_params, prediction_params)

        return loss, enc_grads, trans_grads, proj_grads, pred_grads

    @jax.jit
    def update_spr_heads(transition_params, projection_params, prediction_params,
                          trans_grads, proj_grads, pred_grads,
                          trans_opt_state, proj_opt_state, pred_opt_state):
        """Update SPR auxiliary networks via Adam."""
        trans_updates, trans_opt_state = trans_optimizer.update(trans_grads, trans_opt_state)
        transition_params = optax.apply_updates(transition_params, trans_updates)
        proj_updates, proj_opt_state = proj_optimizer.update(proj_grads, proj_opt_state)
        projection_params = optax.apply_updates(projection_params, proj_updates)
        pred_updates, pred_opt_state = pred_optimizer.update(pred_grads, pred_opt_state)
        prediction_params = optax.apply_updates(prediction_params, pred_updates)
        return (transition_params, projection_params, prediction_params,
                trans_opt_state, proj_opt_state, pred_opt_state)

    @jax.jit
    def ema_update_targets(target_encoder_params, target_projection_params,
                           encoder_params, projection_params, spr_tau):
        """EMA update for target encoder and target projection."""
        target_encoder_params = jax.tree.map(
            lambda t, o: spr_tau * t + (1 - spr_tau) * o,
            target_encoder_params, encoder_params,
        )
        target_projection_params = jax.tree.map(
            lambda t, o: spr_tau * t + (1 - spr_tau) * o,
            target_projection_params, projection_params,
        )
        return target_encoder_params, target_projection_params

    @jax.jit
    def apply_spr_encoder_update(critic_params, spr_encoder_update, spr_weight, lr):
        """Apply SPR encoder update to critic params."""
        encoder_update = jax.tree.map(lambda g: -lr * spr_weight * g, spr_encoder_update)
        new_encoder_params = optax.apply_updates(critic_params["encoder"], encoder_update)
        return {**critic_params, "encoder": new_encoder_params}

    # --- Main training loop ---
    all_logs = []
    episodes = 0
    spr_loss_value = jnp.array(0.0)
    encoder_cos = jnp.array(0.0)
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

        # For truncated episodes, done_mask=1.0 so we still bootstrap V(s')
        done_mask = 1.0 - float(terminated)
        reset = bool(terminated or truncated)

        # Update trajectory buffer
        traj_buf = add_to_trajectory_buffer(traj_buf, jnp.array(obs), jnp.array(action, dtype=jnp.float32), K)

        # --- RL update: ObGD for critic and actor ---
        (actor_params, actor_opt_state, critic_params, critic_opt_state,
         rl_encoder_update, rl_metrics) = update_critic_and_actor(
            actor_params, actor_opt_state, critic_params, critic_opt_state,
            jnp.array(obs), jnp.array(action, dtype=jnp.float32),
            jnp.float32(reward), jnp.array(next_obs),
            jnp.float32(done_mask), reset=reset,
        )

        # --- SPR update (if trajectory buffer has enough data) ---
        has_data = bool(traj_buf.full) or int(traj_buf.ptr) > 1
        if has_data:
            spr_loss_value, spr_enc_grads, spr_trans_grads, spr_proj_grads, spr_pred_grads = (
                compute_spr_loss_and_grads(
                    critic_params["encoder"],
                    transition_params, projection_params, prediction_params,
                    target_encoder_params, target_projection_params,
                    traj_buf.observations, traj_buf.actions,
                    traj_buf.ptr, traj_buf.full,
                )
            )

            # Orth: project each SPR gradient against its own EMA momentum
            spr_enc_grads_orth = orthogonal_gradient_projection(spr_enc_grads, encoder_momentum)
            spr_trans_grads = orthogonal_gradient_projection(spr_trans_grads, transition_momentum)
            spr_proj_grads = orthogonal_gradient_projection(spr_proj_grads, projection_momentum)
            spr_pred_grads = orthogonal_gradient_projection(spr_pred_grads, prediction_momentum)

            # Update EMA momentums
            encoder_momentum = update_momentum(encoder_momentum, spr_enc_grads, args.orth_beta)
            transition_momentum = update_momentum(transition_momentum, spr_trans_grads, args.orth_beta)
            projection_momentum = update_momentum(projection_momentum, spr_proj_grads, args.orth_beta)
            prediction_momentum = update_momentum(prediction_momentum, spr_pred_grads, args.orth_beta)

            # Orth²: project SPR encoder grad orthogonal to RL encoder grad
            if args.orthogonalize_shared_encoder:
                spr_enc_grads_orth, encoder_cos = orthogonal_component_against(
                    spr_enc_grads_orth, rl_encoder_update
                )

            # Apply SPR encoder update to critic params
            critic_params = apply_spr_encoder_update(
                critic_params, spr_enc_grads_orth, args.spr_weight, args.lr
            )

            # Update SPR auxiliary heads via Adam
            (transition_params, projection_params, prediction_params,
             trans_opt_state, proj_opt_state, pred_opt_state) = update_spr_heads(
                transition_params, projection_params, prediction_params,
                spr_trans_grads, spr_proj_grads, spr_pred_grads,
                trans_opt_state, proj_opt_state, pred_opt_state,
            )

            # EMA update targets
            target_encoder_params, target_projection_params = ema_update_targets(
                target_encoder_params, target_projection_params,
                critic_params["encoder"], projection_params,
                args.spr_tau,
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
                    "losses/td_error": float(rl_metrics["td_error"]),
                    "losses/v_value": float(rl_metrics["v_value"]),
                    "losses/spr_loss": float(spr_loss_value),
                    "debug/encoder_cos": float(encoder_cos),
                    "charts/SPS": sps,
                },
                step=global_step,
            )

        # Reset env and trajectory buffer on episode boundary
        if reset:
            next_obs, _ = env.reset()
            next_obs = np.array(next_obs, dtype=np.float32)
            traj_buf = reset_trajectory_buffer(
                traj_buf, jnp.array(next_obs), K, obs_dim, action_dim
            )

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
            f.write(flax.serialization.to_bytes(
                [actor_params, critic_params, transition_params, projection_params, prediction_params]
            ))
        print(f"model saved to {model_path}")

    env.close()
    wandb.finish()
