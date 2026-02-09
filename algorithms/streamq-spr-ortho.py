import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from typing import Callable, NamedTuple, Optional, List
from dataclasses import dataclass
import gymnasium as gym
import itertools
from utils import normalization
import wandb
from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
from collections import namedtuple, deque
import random
import os
import pickle
from flax.training.train_state import TrainState
from flax import struct
from flax.core import frozen_dict
import csv
import flax
import flax.linen as nn

from utils import tree
from networks.MLP import sparse_init
from networks.value_networks import (
    OctaxQNetworkSPR,
    OctaxTransitionNetwork,
    OctaxTargetEncoder,
    OctaxProjection,
    OctaxOnlinePrediction,
)
from utils.optimizers import sgd_with_traces, obgd_with_traces
from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from utils.store_episode_returns_and_lengths import (
    StoreEpisodeReturnsAndLengths,
)
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

import ale_py

gym.register_envs(ale_py)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "None"
    """the wandb's project name"""
    wandb_entity: str = "None"
    """the entity (team) of wandb's project"""
    exp_class: str = "tmp"
    """the class of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    num_checkpoints: int = 100
    """number of checkpoints to save during training"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    log_dir: str = "logs"
    """the logging directory"""
    env_type: str = "octax"
    """the type of environment"""

    # Algorithm specific arguments
    env_id: str = "filter"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    q_lr: float = 1e-4
    """the learning rate of the Q-network optimizer"""
    kappa: float = 2.0
    """kappa for OBGD optimizer"""
    alpha_sc: float = 1.0
    """alpha scaling - not used in current code but good to have"""
    zeta: float = 1.0
    """zeta parameter - not used in current code but good to have"""
    gamma: float = 0.99
    """the discount factor gamma"""
    start_epsilon: float = 1.0
    """the starting epsilon for exploration"""
    end_epsilon: float = 0.01
    """the ending epsilon for exploration"""
    explore_frac: float = 0.20
    """the fraction of total-timesteps for epsilon annealing"""
    h_lr_scale: float = 0.1
    """scale for auxiliary network learning rate"""
    lamda: float = 0.8
    """TD(lambda) parameter"""
    reg_coeff: float = 1.0
    """TDRC regularization parameter for auxiliary network"""
    sparse_init: float = 0.9
    """the sparsity of the neural network weights"""
    layer_norm: bool = True
    """whether to use layer normalization in the networks"""
    activation: str = "leaky_relu"
    """the activation function to use in the networks"""
    gradient_correction: bool = True
    """whether to use gradient correction (TDC vs GTD2)"""
    opt: str = "obgd"
    """optimizer to use"""
    net_arch: str = "minatar"
    """network architecture"""
    mlp_layers: Optional[List[int]] = None
    """MLP hidden layers"""
    periodic_checkpointing: bool = False
    """whether to save periodic checkpoints during training"""
    resume: bool = False
    """whether to resume from the last checkpoint if available"""
    use_spr: bool = True
    """whether to use SPR (Self-Predictive Representations)"""
    spr_latent_dim: int = 128
    """dimension of latent representations"""
    spr_projection_dim: int = 64
    """dimension of projection space"""
    spr_prediction_depth: int = 5
    """K - number of steps to predict"""
    spr_weight: float = 2.0
    """lambda - weight for SPR loss"""
    spr_tau: float = 0.99
    """EMA coefficient for target network (0.0 to disable)"""
    shared_online_proj: bool = False
    """whether to share projection head between Q-network and SPR"""
    use_augmentation: bool = False
    """whether to use data augmentation for SPR"""
    orthogonalize_shared_encoder: bool = True
    """project SPR encoder updates orthogonal to RL updates"""

    max_grad: float = 100.0
    """maximum gradient norm for clipping"""


class OctaxToGymAdapter(gym.Env):
    """Adapter to make Octax/Gymnax environment look like a standard Gym environment."""

    def __init__(self, env_id: str, seed: int = 0):
        try:
            from octax.environments import create_environment
            from octax.wrappers import OctaxGymnaxWrapper
        except ImportError:
            raise ImportError("Please install octax to use octax environments.")

        self.env_raw, self.metadata = create_environment(env_id)
        self.env = OctaxGymnaxWrapper(self.env_raw)
        self.env_params = self.env.default_params
        # Create a unique RNG key based on seed.
        # Note: If environment is destroyed and recreated with same seed, sequence repeats.
        self.rng = jax.random.PRNGKey(seed)
        self.env_state = None

        self.action_space = gym.spaces.Discrete(self.env.num_actions)

        # Get observation shape by running a reset
        rng_dummy = jax.random.PRNGKey(0)
        obs_dummy, _ = self.env.reset(rng_dummy, self.env_params)
        # Octax usually returns (C, H, W), we transpose to (H, W, C)
        self.obs_shape = (obs_dummy.shape[1], obs_dummy.shape[2], obs_dummy.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

        self.rng, rng_reset = jax.random.split(self.rng)
        obs, self.env_state = self.env.reset(rng_reset, self.env_params)
        obs = jnp.transpose(obs, (1, 2, 0))
        return np.array(obs), {}

    def step(self, action):
        self.rng, rng_step = jax.random.split(self.rng)
        obs, self.env_state, reward, done, info = self.env.step(
            rng_step, self.env_state, action, self.env_params
        )
        obs = jnp.transpose(obs, (1, 2, 0))
        return np.array(obs), float(reward), bool(done), False, {}


def make_env(args, idx, run_name):
    if args.env_type == "atari":
        if args.capture_video and idx == 0:
            env = gym.make(args.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(args.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env = RecordEpisodeStatisticsJAX(env)
        # env = StoreEpisodeReturnsAndLengthsJAX(env)

        # Normalization must come after stats for correct results
        # if env_config.normalize_reward:
        env = ScaleRewardJAX(env, gamma=args.gamma)
        # if env_config.normalize_obs:
        env = NormalizeObservationJAX(env)

        env.action_space.seed(args.seed)
        return env

    elif args.env_type == "octax":
        print(f"Creating Octax environment: {args.env_id}")
        env = OctaxToGymAdapter(args.env_id, seed=args.seed)

        env = RecordEpisodeStatisticsJAX(env)
        env = ScaleRewardJAX(env, gamma=args.gamma)
        env = NormalizeObservationJAX(env)

        return env

    elif args.env_type == "minatar":
        print(f"Creating standard Gym Minatar environment: {args.env_id}")
        env = gym.make(args.env_id)

        # Episode statistics wrapper
        env = RecordEpisodeStatistics(env)
        env = StoreEpisodeReturnsAndLengths(env)

        # Normalization must come after stats for correct results
        env = normalization.ScaleReward(env, gamma=args.gamma)
        env = normalization.NormalizeObservation(env)

        return env


# TODO make this faster / compatible with jax.jit
class RecordEpisodeStatisticsJAX(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: jnp.ndarray = None
        self.episode_returns: Optional[jnp.ndarray] = None
        self.episode_lengths: Optional[jnp.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = jnp.full(
            self.num_envs, time.perf_counter(), dtype=jnp.float32
        )
        self.episode_returns = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = jnp.logical_or(terminations, truncations)
        num_dones = jnp.sum(dones)
        if num_dones:
            # if "episode" in infos or "_episode" in infos:
            #     raise ValueError(
            #         "Attempted to add episode stats when they already exist"
            #     )
            # else:
            #     infos["episode"] = {
            #         "r": jnp.where(dones, self.episode_returns, 0.0),
            #         "l": jnp.where(dones, self.episode_lengths, 0),
            #         "t": jnp.where(
            #             dones,
            #             jnp.round(time.perf_counter() - self.episode_start_times, 6),
            #             0.0,
            #         ),
            #     }
            #     if self.is_vector_env:
            #         infos["_episode"] = jnp.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones

            self.episode_lengths = self.episode_lengths.at[dones].set(0)
            self.episode_returns = self.episode_returns.at[dones].set(0)
            self.episode_start_times = self.episode_start_times.at[dones].set(
                time.perf_counter()
            )
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


# TODO make this faster / compatible with jax.jit
class SampleMeanStdJAX:
    def __init__(self, shape=()):
        self.mean = jnp.zeros(shape, "float32")
        self.var = jnp.ones(shape, "float32")
        self.p = jnp.ones(shape, "float32")
        self.count = 0

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = jnp.zeros_like(x)
        self.mean, self.var, self.p, self.count = (
            self.update_mean_var_count_from_moments(
                self.mean, self.p, self.count, x * 1.0
            )
        )

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count


# TODO make this faster / compatible with jax.jit
class NormalizeObservationJAX(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_stats = SampleMeanStdJAX(shape=self.single_observation_space.shape)
        else:
            self.obs_stats = SampleMeanStdJAX(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(jnp.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(jnp.array([obs]))[0], info

    def normalize(self, obs):
        self.obs_stats.update(obs)
        return (obs - self.obs_stats.mean) / jnp.sqrt(self.obs_stats.var + self.epsilon)


# TODO make this faster / compatible with jax.jit
class ScaleRewardJAX(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    __doc__ = gym.core.Wrapper.__doc__

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        self.reward_stats = SampleMeanStdJAX(shape=())
        self.reward_trace = jnp.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = jnp.array([rews])
        term = terminateds or truncateds
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        return rews / jnp.sqrt(self.reward_stats.var + self.epsilon)


@struct.dataclass
class Config:
    d: dict = struct.field(pytree_node=False)

    def __getattribute__(self, name):
        d = object.__getattribute__(self, "d")
        try:
            return d[name]
        except KeyError:
            return object.__getattribute__(self, name)

    @classmethod
    def from_dict(cls: "Config", d: dict):
        return cls(d)

    @classmethod
    def from_args(cls: "Config", args: Args):
        return cls(vars(args))

    def to_dict(self, expand: bool = True):
        state_dict = flax_struct_to_dict(self)
        return transform_dict(state_dict, expand)


Agent = namedtuple("Agent", ["init_state", "step", "update"])


###################################################################################
@struct.dataclass
class TrajectoryBuffer:
    """Stores short trajectory for SPR prediction."""

    observations: jnp.ndarray  # shape: (K+1, *obs_shape)
    actions: jnp.ndarray  # shape: (K,)
    ptr: int  # current position
    K: int = struct.field(pytree_node=False)  # max length

    @classmethod
    def create(cls, K: int, obs_shape: tuple):
        return cls(
            observations=jnp.zeros((K + 1, *obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((K,), dtype=jnp.int32),
            ptr=0,
            K=K,
        )

    def add(self, obs, action):
        """Add new observation and action, shifting if full."""
        new_obs = jnp.roll(self.observations, -1, axis=0)
        new_obs = new_obs.at[-1].set(obs)
        new_actions = jnp.roll(self.actions, -1, axis=0)
        new_actions = new_actions.at[-1].set(action)
        return self.replace(
            observations=new_obs,
            actions=new_actions,
            ptr=jnp.minimum(self.ptr + 1, self.K + 1),
        )

    def reset(self):
        return self.replace(ptr=0)

    def is_full(self):
        return self.ptr >= self.K + 1


def ema_update(target_params, online_params, tau):
    """Exponential moving average update for target network."""
    return jax.tree.map(
        lambda t, o: tau * t + (1 - tau) * o, target_params, online_params
    )


def orthogonal_gradient_projection(grad_t, momentum_t):
    """
    Compute orthogonal gradient by projecting out component parallel to momentum.

    ut = gt - proj_ct-1(gt)
    where proj_ct-1(gt) = (gt · ct-1) / (ct-1 · ct-1) * ct-1

    Args:
        grad_t: Current gradient (pytree)
        momentum_t: EMA momentum of past gradients (pytree)

    Returns:
        Orthogonal gradient (pytree)
    """
    # Flatten gradients for dot product computation
    grad_flat, grad_treedef = jax.tree_util.tree_flatten(grad_t)
    momentum_flat, _ = jax.tree_util.tree_flatten(momentum_t)

    # Compute dot products: gt · ct-1 and ct-1 · ct-1
    dot_grad_momentum = sum([jnp.sum(g * m) for g, m in zip(grad_flat, momentum_flat)])
    dot_momentum_momentum = sum([jnp.sum(m * m) for m in momentum_flat])

    # Avoid division by zero for first step or zero momentum
    epsilon = 1e-8
    projection_coeff = dot_grad_momentum / (dot_momentum_momentum + epsilon)

    # Compute orthogonal gradient: ut = gt - projection_coeff * ct-1
    orthogonal_grad = jax.tree.map(
        lambda g, m: g - projection_coeff * m, grad_t, momentum_t
    )

    return orthogonal_grad


def orthogonal_component_against(update_t, reference_t, epsilon=1e-8):
    """Project ``update_t`` to be orthogonal to ``reference_t`` and return cosine similarity."""
    update_flat, _ = jax.tree_util.tree_flatten(update_t)
    reference_flat, _ = jax.tree_util.tree_flatten(reference_t)

    dot_update_ref = sum([jnp.sum(u * r) for u, r in zip(update_flat, reference_flat)])
    ref_norm_sq = sum([jnp.sum(r * r) for r in reference_flat])
    update_norm_sq = sum([jnp.sum(u * u) for u in update_flat])

    projection_coeff = dot_update_ref / (ref_norm_sq + epsilon)
    orthogonal_update = jax.tree.map(
        lambda u, r: u - projection_coeff * r, update_t, reference_t
    )

    cosine = dot_update_ref / (
        jnp.sqrt((update_norm_sq + epsilon) * (ref_norm_sq + epsilon)) + epsilon
    )
    cosine = jnp.clip(cosine, -1.0, 1.0)
    return orthogonal_update, cosine


def update_momentum(momentum_t, grad_t, beta):
    """
    Update EMA momentum with current gradient.

    ct = β * ct-1 + (1 - β) * gt

    Args:
        momentum_t: Current momentum (pytree)
        grad_t: Current gradient (pytree)
        beta: Momentum factor (scalar)

    Returns:
        Updated momentum (pytree)
    """
    return jax.tree.map(lambda m, g: beta * m + (1.0 - beta) * g, momentum_t, grad_t)


def cosine_similarity_loss(pred, target):
    """Negative cosine similarity with improved numerical stability."""
    epsilon = 1e-8
    pred_norm = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + epsilon)
    target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + epsilon)
    cos_sim = jnp.sum(pred_norm * target_norm, axis=-1)
    # Clip to avoid numerical issues
    cos_sim = jnp.clip(cos_sim, -1.0 + epsilon, 1.0 - epsilon)
    return -cos_sim


class AgentState(NamedTuple):
    agent_config: Config
    train_state: TrainState
    # SPR components
    target_encoder_state: Optional[TrainState]
    proj_online_state: Optional[TrainState]
    proj_target_state: Optional[TrainState]
    predictor_state: Optional[TrainState]
    transition_state: Optional[TrainState]
    # Orthogonal SPR momentum terms
    spr_momentum_online: Optional[any]  # momentum for shared encoder
    spr_momentum_proj: Optional[any]  # momentum for projection head
    spr_momentum_pred: Optional[any]  # momentum for predictor
    spr_momentum_trans: Optional[any]  # momentum for transition model


@partial(jax.jit, static_argnames=["action_dim"])
def agent_step(
    agent_state: AgentState,
    obs: jnp.ndarray,
    action_dim: int,
    epsilon: float,
    rng: jax.random.PRNGKey,
):
    params = agent_state.train_state.params
    q = agent_state.train_state.apply_fn(params, obs)
    argmax = jnp.argmax(q)

    rng, rng_e = jax.random.split(rng)

    def random_action_fn(rng_in):
        rng_out, rng_a = jax.random.split(rng_in)
        action = jax.random.randint(rng_a, shape=(), minval=0, maxval=action_dim)
        return action, rng_out

    def greedy_action_fn(rng_in):
        return argmax, rng_in

    action, rng_out = jax.lax.cond(
        jax.random.uniform(rng_e, minval=0.0, maxval=1.0) < epsilon,
        random_action_fn,
        greedy_action_fn,
        rng,
    )
    is_nongreedy = action != argmax
    return action, is_nongreedy, rng_out


def init_agent_state_streamq(
    agent_config: Config, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey
):
    if agent_config.env_type == "atari":
        net_kwargs = {
            "action_dim": action_dim,
            "layer_norm": agent_config.layer_norm,
            "activation": agent_config.activation,
            "kernel_init": sparse_init(sparsity=agent_config.sparse_init),
            "features_list": (32, 64, 64),
            "kernel_sizes": ((8, 8), (4, 4), (3, 3)),
            "strides_list": ((4, 4), (2, 2), (1, 1)),
            "hidden_layer_sizes": (256,),
            "dyn_features_list": (64, 64),
            "dyn_kernel_sizes": ((3, 3), (3, 3)),
            "dyn_strides_list": ((1, 1), (1, 1)),
        }
    elif agent_config.env_type == "octax":
        net_kwargs = {
            "action_dim": action_dim,
            "layer_norm": agent_config.layer_norm,
            "activation": agent_config.activation,
            "kernel_init": sparse_init(sparsity=agent_config.sparse_init),
            "features_list": (32, 64, 64),
            "kernel_sizes": ((8, 8), (4, 4), (3, 3)),
            "strides_list": ((4, 4), (2, 2), (1, 1)),
            "hidden_layer_sizes": (256,),
            "dyn_features_list": (64, 64),
            "dyn_kernel_sizes": ((3, 3), (3, 3)),
            "dyn_strides_list": ((1, 1), (1, 1)),
        }
    elif agent_config.env_type == "minatar":
        net_kwargs = {
            "action_dim": action_dim,
            "layer_norm": agent_config.layer_norm,
            "activation": agent_config.activation,
            "kernel_init": sparse_init(sparsity=agent_config.sparse_init),
            "features_list": (16,),
            "kernel_sizes": ((3, 3)),
            "strides_list": ((1, 1)),
            "hidden_layer_sizes": (128,),
            "dyn_features_list": (16, 16),
            "dyn_kernel_sizes": ((3, 3), (3, 3)),
            "dyn_strides_list": ((1, 1), (1, 1)),
        }
    else:
        raise ValueError(f"unknown environment type: {agent_config.env_type}")
    net_arch = agent_config.net_arch
    if net_arch == "mlp":
        net_kwargs["hiddens"] = agent_config.mlp_layers

    network = OctaxQNetworkSPR(**net_kwargs)
    init_x = jnp.zeros(obs_shape)

    # Initialize network parameters
    rng, _rng = jax.random.split(rng)
    params = network.init(_rng, init_x)

    def params_sum(params):
        return sum(
            jax.tree_util.tree_leaves(jax.tree.map(lambda x: np.prod(x.shape), params))
        )

    print(f"Total number of params: {params_sum(params)}")

    # Create optimizer
    tx_cls = {
        "obgd": obgd_with_traces,
        "sgd": sgd_with_traces,
    }[agent_config.opt]

    # Configure optimizer arguments
    opt_kwrgs = {
        "lr": agent_config.q_lr,
        "gamma": agent_config.gamma,
        "lambd": agent_config.lamda,
    }

    if agent_config.opt == "obgd":
        opt_kwrgs["kappa"] = agent_config.kappa

    tx = tx_cls(**opt_kwrgs)

    # Create train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    # Initialize SPR components
    target_encoder_state = None
    proj_online_state = None
    proj_target_state = None
    predictor_state = None
    transition_state = None

    # Initialize momentum terms
    spr_momentum_online = None
    spr_momentum_proj = None
    spr_momentum_pred = None
    spr_momentum_trans = None

    if agent_config.use_spr:
        # Helper for creating optimizers
        def make_spr_tx():
            return optax.sgd(agent_config.q_lr)

        # Target Encoder
        target_encoder = OctaxTargetEncoder(**net_kwargs)
        rng, _rng = jax.random.split(rng)
        target_encoder_params = target_encoder.init(_rng, jnp.zeros(obs_shape))
        target_encoder_state = TrainState.create(
            apply_fn=target_encoder.apply,
            params=target_encoder_params,
            tx=make_spr_tx(),
        )

        # Initialize momentum for shared encoder to decorrelate streaming updates
        spr_momentum_online = jax.tree.map(jnp.zeros_like, params)

        # Infer latent shape by running a dummy pass
        dummy_out = target_encoder.apply(
            target_encoder_params, jnp.zeros((1, *obs_shape))
        )
        # Note: dummy_out might include batch dim

        # Projection heads

        if not agent_config.shared_online_proj:
            rng, _rng = jax.random.split(rng)
            proj_online = OctaxProjection(**net_kwargs)
            proj_online_params = proj_online.init(_rng, dummy_out)
            proj_online_state = TrainState.create(
                apply_fn=proj_online.apply,
                params=proj_online_params,
                tx=make_spr_tx(),
            )
            spr_momentum_proj = jax.tree.map(jnp.zeros_like, proj_online_params)

        rng, _rng = jax.random.split(rng)
        proj_target = OctaxProjection(**net_kwargs)
        proj_target_params = proj_target.init(_rng, dummy_out)
        proj_target_state = TrainState.create(
            apply_fn=proj_target.apply,
            params=proj_target_params,
            tx=make_spr_tx(),
        )

        # Run dummy projection to get shape for Predictor/Transition
        dummy_proj = proj_target.apply(proj_target_params, dummy_out)

        # Predictor
        # Predictor operates on projected output
        rng, _rng = jax.random.split(rng)
        predictor = OctaxOnlinePrediction(**net_kwargs)
        predictor_params = predictor.init(_rng, dummy_proj)
        predictor_state = TrainState.create(
            apply_fn=predictor.apply,
            params=predictor_params,
            tx=make_spr_tx(),
        )
        spr_momentum_pred = jax.tree.map(jnp.zeros_like, predictor_params)

        # Transition model
        # Transition model operates on Latent state (dummy_out), NOT projected state
        rng, _rng = jax.random.split(rng)
        transition_model = OctaxTransitionNetwork(**net_kwargs)
        transition_params = transition_model.init(_rng, dummy_out, jnp.int32(0))
        transition_state = TrainState.create(
            apply_fn=transition_model.apply,
            params=transition_params,
            tx=make_spr_tx(),
        )
        spr_momentum_trans = jax.tree.map(jnp.zeros_like, transition_params)

        print("SPR Components Initialized")

    return (
        AgentState(
            agent_config,
            train_state,
            target_encoder_state,
            proj_online_state,
            proj_target_state,
            predictor_state,
            transition_state,
            spr_momentum_online,
            spr_momentum_proj,
            spr_momentum_pred,
            spr_momentum_trans,
        ),
        rng,
    )


@partial(jax.jit, static_argnames=["terminated", "truncated", "is_nongreedy"])
def update_step_streamq(
    agent_state,
    transition,
    terminated,
    truncated,
    is_nongreedy,
    traj_buffer=None,
    rng=None,
):
    obs, action, next_obs, reward = transition

    config = agent_state.agent_config
    train_state = agent_state.train_state
    params = train_state.params
    opt_state = train_state.opt_state

    def get_q(params):
        q = train_state.apply_fn(params, obs)
        return q[action]

    q_taken, grads = jax.value_and_grad(get_q)(params)

    td_error = reward - q_taken
    if not terminated:
        next_q_vect = train_state.apply_fn(params, next_obs)
        td_error += config.gamma * jnp.max(next_q_vect, axis=-1)

    tx = train_state.tx
    updates, opt_state = tx.update(
        grads,
        opt_state,
        params,
        td_error,
        reset=(terminated or truncated or is_nongreedy),
    )

    rl_encoder_update = None
    updates_params = updates.get("params") if hasattr(updates, "get") else None
    if updates_params is not None and "online_encoder" in updates_params:
        rl_encoder_update = updates_params["online_encoder"]

    # Apply RL updates
    params = optax.apply_updates(params, updates)

    # SPR updates
    spr_loss = 0.0
    encoder_update_cos = jnp.array(0.0, dtype=jnp.float32)

    target_encoder_state = agent_state.target_encoder_state
    proj_online_state = agent_state.proj_online_state
    proj_target_state = agent_state.proj_target_state
    predictor_state = agent_state.predictor_state
    transition_state = agent_state.transition_state

    spr_momentum_online = agent_state.spr_momentum_online
    spr_momentum_proj = agent_state.spr_momentum_proj
    spr_momentum_pred = agent_state.spr_momentum_pred
    spr_momentum_trans = agent_state.spr_momentum_trans

    if config.use_spr and traj_buffer is not None and rng is not None:
        spr_obs = traj_buffer.observations
        spr_actions = traj_buffer.actions

        # Target Projections (No grad)
        target_latents = target_encoder_state.apply_fn(
            target_encoder_state.params, spr_obs
        )
        target_projections = proj_target_state.apply_fn(
            proj_target_state.params, target_latents
        )
        target_projections = target_projections[1:]  # Targets t=1..K

        def spr_loss_fn(online_params, online_proj_params, pred_params, trans_params):
            # Online projections
            # Map online_params to target_encoder structure to reuse apply_fn
            enc_params_mapped = {
                "params": {"target_encoder": online_params["params"]["online_encoder"]}
            }
            # Start with z_0 (Latent), keep batch dim
            z_curr = target_encoder_state.apply_fn(
                enc_params_mapped, spr_obs[0][None, ...]
            )

            loss = 0.0

            for k in range(config.spr_prediction_depth):
                # Transition z_t -> z_{t+1}
                z_curr = transition_state.apply_fn(
                    trans_params, z_curr, spr_actions[k][None, ...]
                )

                # Project z_{t+1} -> y_{t+1}
                if config.shared_online_proj:
                    proj_params_mapped = {
                        "params": {"proj": online_params["params"]["q_projection"]}
                    }
                    y_curr = proj_target_state.apply_fn(proj_params_mapped, z_curr)
                else:
                    y_curr = proj_online_state.apply_fn(online_proj_params, z_curr)

                # Predict y_{t+1}
                pred = predictor_state.apply_fn(pred_params, y_curr)

                # Target is (K, F), k-th element is (F,). Expand to (1, F)
                target = target_projections[k][None, ...]
                loss_k = cosine_similarity_loss(pred, target)
                loss += jnp.mean(loss_k)

            return loss / config.spr_prediction_depth

        # Compute Gradients (only when trajectory buffer is ready)
        spr_grads_fn = jax.grad(spr_loss_fn, argnums=(0, 1, 2, 3))

        zero_online = jax.tree.map(jnp.zeros_like, params)
        zero_proj = (
            jax.tree.map(jnp.zeros_like, proj_online_state.params)
            if proj_online_state
            else None
        )
        zero_pred = jax.tree.map(jnp.zeros_like, predictor_state.params)
        zero_trans = jax.tree.map(jnp.zeros_like, transition_state.params)

        buffer_ready = jnp.asarray(traj_buffer.is_full(), dtype=jnp.bool_)
        not_terminated = jnp.logical_not(jnp.asarray(terminated, dtype=jnp.bool_))
        not_truncated = jnp.logical_not(jnp.asarray(truncated, dtype=jnp.bool_))
        not_nongreedy = jnp.logical_not(jnp.asarray(is_nongreedy, dtype=jnp.bool_))
        spr_condition = buffer_ready & not_terminated & not_truncated & not_nongreedy

        def compute_spr(_):
            grads_online, grads_proj, grads_pred, grads_trans = spr_grads_fn(
                params,
                proj_online_state.params if proj_online_state else None,
                predictor_state.params,
                transition_state.params,
            )
            loss_val = spr_loss_fn(
                params,
                proj_online_state.params if proj_online_state else None,
                predictor_state.params,
                transition_state.params,
            )
            return loss_val, grads_online, grads_proj, grads_pred, grads_trans

        def skip_spr(_):
            return (
                jnp.array(0.0, dtype=jnp.float32),
                zero_online,
                zero_proj,
                zero_pred,
                zero_trans,
            )

        spr_loss, grads_online, grads_proj, grads_pred, grads_trans = jax.lax.cond(
            spr_condition,
            compute_spr,
            skip_spr,
            operand=None,
        )

        if spr_momentum_online is not None:
            grads_online = orthogonal_gradient_projection(
                grads_online, spr_momentum_online
            )
            spr_momentum_online = update_momentum(
                spr_momentum_online, grads_online, 0.99
            )

        # Apply orthogonal gradient projection for auxiliary heads
        # 1. Projection head
        if not config.shared_online_proj and proj_online_state:
            grads_proj = orthogonal_gradient_projection(grads_proj, spr_momentum_proj)
            spr_momentum_proj = update_momentum(spr_momentum_proj, grads_proj, 0.99)

        # 2. Predictor
        grads_pred = orthogonal_gradient_projection(grads_pred, spr_momentum_pred)
        spr_momentum_pred = update_momentum(spr_momentum_pred, grads_pred, 0.99)

        # 3. Transition
        grads_trans = orthogonal_gradient_projection(grads_trans, spr_momentum_trans)
        spr_momentum_trans = update_momentum(spr_momentum_trans, grads_trans, 0.99)

        # Apply Updates
        # 1. Online Encoder (shared) - add to params with orthogonalization
        spr_update_vec = jax.tree.map(
            lambda g: -config.q_lr * config.spr_weight * g, grads_online
        )

        if (
            config.orthogonalize_shared_encoder
            and rl_encoder_update is not None
            and "params" in spr_update_vec
            and "online_encoder" in spr_update_vec["params"]
        ):
            spr_encoder_update = spr_update_vec["params"]["online_encoder"]
            (
                orth_update,
                encoder_update_cos,
            ) = orthogonal_component_against(spr_encoder_update, rl_encoder_update)
            if isinstance(spr_update_vec, frozen_dict.FrozenDict):
                spr_update_mut = frozen_dict.unfreeze(spr_update_vec)
            else:
                spr_update_mut = {**spr_update_vec}

            params_block = spr_update_mut.get("params", {})
            if isinstance(params_block, frozen_dict.FrozenDict):
                params_block = frozen_dict.unfreeze(params_block)
            else:
                params_block = {**params_block}

            params_block["online_encoder"] = orth_update
            spr_update_mut["params"] = params_block
            spr_update_vec = spr_update_mut

        params = optax.apply_updates(params, spr_update_vec)

        # 2. Others
        if proj_online_state:
            proj_online_state = proj_online_state.apply_gradients(
                grads=tree.scale(config.spr_weight, grads_proj)
            )

        predictor_state = predictor_state.apply_gradients(
            grads=tree.scale(config.spr_weight, grads_pred)
        )

        transition_state = transition_state.apply_gradients(
            grads=tree.scale(config.spr_weight, grads_trans)
        )

        # EMA Updates
        target_encoder_params = ema_update(
            target_encoder_state.params["params"]["target_encoder"],
            params["params"]["online_encoder"],
            config.spr_tau,
        )
        target_encoder_state = target_encoder_state.replace(
            params={"params": {"target_encoder": target_encoder_params}}
        )

        if config.shared_online_proj:
            src_proj = params["params"]["q_projection"]
        else:
            src_proj = proj_online_state.params["params"]["proj"]

        proj_target_params = ema_update(
            proj_target_state.params["params"]["proj"],
            src_proj,
            config.spr_tau,
        )
        proj_target_state = proj_target_state.replace(
            params={"params": {"proj": proj_target_params}}
        )

    train_state = TrainState(
        step=train_state.step + 1,
        apply_fn=train_state.apply_fn,
        params=params,
        tx=train_state.tx,
        opt_state=opt_state,
    )

    metrics = {
        "td_error": td_error,
        "q_val": q_taken,
        "spr_loss": spr_loss,
        "encoder_update_cos": encoder_update_cos,
    }
    return (
        AgentState(
            config,
            train_state,
            target_encoder_state,
            proj_online_state,
            proj_target_state,
            predictor_state,
            transition_state,
            spr_momentum_online,
            spr_momentum_proj,
            spr_momentum_pred,
            spr_momentum_trans,
        ),
        metrics,
    )


StreamQAgent = Agent(init_agent_state_streamq, agent_step, update_step_streamq)


###################################################################################


def get_linear_epsilon_schedule(args: Args) -> Callable[[int], float]:
    start_epsilon = args.start_epsilon
    end_epsilon = args.end_epsilon
    assert 0.0 <= end_epsilon <= start_epsilon <= 1.0
    anneal_time = args.explore_frac * args.total_timesteps
    assert anneal_time > 0.0

    def epsilon_schedule(t: int) -> float:
        frac_annealed = min(t / anneal_time, 1.0)
        return (1.0 - frac_annealed) * start_epsilon + frac_annealed * end_epsilon

    return epsilon_schedule


def experiment(args: Args, agent: Agent, run_name: str):
    agent_config = Config.from_args(args)
    rng = jax.random.PRNGKey(args.seed)

    # Create and initialize the environment (single environment, not vectorized)
    env = make_env(args, 0, run_name)
    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs)
    obs = obs.transpose(1, 2, 0)
    episodes = 0
    episode_return = 0.0
    episode_length = 0

    # Initialize the agent
    action_dim = int(env.action_space.n)
    agent_state, rng = agent.init_state(agent_config, action_dim, obs.shape, rng)
    epsilon_schedule = get_linear_epsilon_schedule(args)

    # SPR Buffer
    traj_buffer = TrajectoryBuffer.create(args.spr_prediction_depth, obs.shape)

    # Tracking for logging
    start_time = time.time()
    last_log_step = 0

    # List to accumulate all log dictionaries
    all_logs = []
    checkpointing_frequency = args.total_timesteps // args.num_checkpoints

    def flush_logs(logs, log_dir):
        if not logs:
            return
        csv_path = os.path.join(log_dir, "training_data.csv")
        file_exists = os.path.exists(csv_path)
        mode = "a" if file_exists else "w"
        with open(csv_path, mode, newline="") as csvfile:
            fieldnames = logs[0].keys()
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer_csv.writeheader()
            writer_csv.writerows(logs)

    start_step = 0
    if args.resume:
        checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)

            # Helper to restore state
            def restore_train_state(current, state_bytes):
                if current is None or state_bytes is None:
                    return current
                params = flax.serialization.from_bytes(
                    current.params, state_bytes["params"]
                )
                opt_state = flax.serialization.from_bytes(
                    current.opt_state, state_bytes["opt_state"]
                )
                return current.replace(
                    params=params, opt_state=opt_state, step=checkpoint_data["step"]
                )

            def restore_momentum(current, state_bytes):
                if current is None or state_bytes is None:
                    return current
                return flax.serialization.from_bytes(current, state_bytes)

            new_train_state = restore_train_state(
                agent_state.train_state, checkpoint_data["train_state"]
            )

            new_target_encoder_state = restore_train_state(
                agent_state.target_encoder_state,
                checkpoint_data.get("target_encoder_state"),
            )
            new_proj_online_state = restore_train_state(
                agent_state.proj_online_state, checkpoint_data.get("proj_online_state")
            )
            new_proj_target_state = restore_train_state(
                agent_state.proj_target_state, checkpoint_data.get("proj_target_state")
            )
            new_predictor_state = restore_train_state(
                agent_state.predictor_state, checkpoint_data.get("predictor_state")
            )
            new_transition_state = restore_train_state(
                agent_state.transition_state, checkpoint_data.get("transition_state")
            )

            new_spr_momentum_online = restore_momentum(
                agent_state.spr_momentum_online,
                checkpoint_data.get("spr_momentum_online"),
            )
            new_spr_momentum_proj = restore_momentum(
                agent_state.spr_momentum_proj, checkpoint_data.get("spr_momentum_proj")
            )
            new_spr_momentum_pred = restore_momentum(
                agent_state.spr_momentum_pred, checkpoint_data.get("spr_momentum_pred")
            )
            new_spr_momentum_trans = restore_momentum(
                agent_state.spr_momentum_trans,
                checkpoint_data.get("spr_momentum_trans"),
            )

            agent_state = AgentState(
                agent_config=agent_state.agent_config,
                train_state=new_train_state,
                target_encoder_state=new_target_encoder_state,
                proj_online_state=new_proj_online_state,
                proj_target_state=new_proj_target_state,
                predictor_state=new_predictor_state,
                transition_state=new_transition_state,
                spr_momentum_online=new_spr_momentum_online,
                spr_momentum_proj=new_spr_momentum_proj,
                spr_momentum_pred=new_spr_momentum_pred,
                spr_momentum_trans=new_spr_momentum_trans,
            )

            if "traj_buffer" in checkpoint_data:
                traj_buffer = flax.serialization.from_bytes(
                    traj_buffer, checkpoint_data["traj_buffer"]
                )

            start_step = checkpoint_data["step"] + 1
            rng = checkpoint_data["rng"]
            episodes = checkpoint_data["episodes"]
            episode_return = checkpoint_data["episode_return"]
            episode_length = checkpoint_data["episode_length"]
            print(f"Resumed at step {start_step}")

    for t in range(start_step, args.total_timesteps):
        epsilon = epsilon_schedule(t)
        action, is_nongreedy, rng = agent.step(
            agent_state, obs, action_dim, epsilon, rng
        )
        action = action.item()
        is_nongreedy = is_nongreedy.item()

        # Add to SPR buffer
        traj_buffer = traj_buffer.add(obs, action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.array(next_obs)
        next_obs = next_obs.transpose(1, 2, 0)

        done = terminated or truncated

        episode_return += reward
        episode_length += 1

        # Handle final observation for truncated episodes
        real_next_obs = next_obs.copy()
        # if truncated:
        #     real_next_obs = np.array(
        #         info["final_observation"]
        #     )  # Convert to numpy array

        transition = (obs, action, real_next_obs, reward)

        rng, _rng = jax.random.split(rng)
        agent_state, metrics = agent.update(
            agent_state,
            transition,
            terminated,
            truncated,
            is_nongreedy,
            traj_buffer=traj_buffer,
            rng=_rng,
        )

        if done:
            episodes += 1

            episode_return = 0.0
            episode_length = 0

            traj_buffer = traj_buffer.reset()

            next_obs, info = env.reset()
            next_obs = np.array(next_obs)
            next_obs = next_obs.transpose(1, 2, 0)

        # Periodic logging every 1000 steps
        if t % 1000 == 0 and t > 0:
            steps_elapsed = t - last_log_step
            sps = int(steps_elapsed / (time.time() - start_time))

            if len(env.get_wrapper_attr("return_queue")) > 0:
                avg_return = np.mean(env.get_wrapper_attr("return_queue"))
                avg_length = np.mean(env.get_wrapper_attr("length_queue"))
            else:
                avg_return = 0.0
                avg_length = 0.0

            # print(
            #     f"Step: {t}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}, SPS: {sps}, Epsilon: {epsilon:.3f}"
            # )

            # Create log dictionary
            log_dict = {
                "global_step": t,
                "avg_return": avg_return,
                "avg_length": avg_length,
                "td_loss": float(metrics["td_error"]),
                "q_values": float(metrics["q_val"]),
                "spr_loss": float(metrics.get("spr_loss", 0.0)),
                "encoder_update_cos": float(metrics.get("encoder_update_cos", 0.0)),
                "SPS": sps,
                "epsilon": epsilon,
                "episodes": episodes,
            }

            wandb.log(
                {
                    # "global_step": t,
                    "charts/episodic_return": avg_return,
                    "charts/episodic_length": avg_length,
                    "charts/SPS": sps,
                    "charts/epsilon": epsilon,
                    "losses/td_loss": log_dict["td_loss"],
                    "losses/q_values": log_dict["q_values"],
                    "losses/spr_loss": log_dict["spr_loss"],
                    "debug/encoder_cos": log_dict["encoder_update_cos"],
                    "episodes": episodes,
                },
                step=t,
            )

            # Accumulate log_dict for CSV export
            all_logs.append(log_dict.copy())

            last_log_step = t
            start_time = time.time()

        # Periodic model checkpointing AND RESUME CHECKPOINT
        if (t % checkpointing_frequency == 0 or t % 10000 == 0) and t > 0:
            flush_logs(all_logs, args.log_dir)
            all_logs = []

            def save_train_state(ts):
                if ts is None:
                    return None
                return {
                    "params": flax.serialization.to_bytes(ts.params),
                    "opt_state": flax.serialization.to_bytes(ts.opt_state),
                }

            def save_momentum(mom):
                if mom is None:
                    return None
                return flax.serialization.to_bytes(mom)

            checkpoint_data = {
                "step": t,
                "train_state": save_train_state(agent_state.train_state),
                "target_encoder_state": save_train_state(
                    agent_state.target_encoder_state
                ),
                "proj_online_state": save_train_state(agent_state.proj_online_state),
                "proj_target_state": save_train_state(agent_state.proj_target_state),
                "predictor_state": save_train_state(agent_state.predictor_state),
                "transition_state": save_train_state(agent_state.transition_state),
                "spr_momentum_online": save_momentum(agent_state.spr_momentum_online),
                "spr_momentum_proj": save_momentum(agent_state.spr_momentum_proj),
                "spr_momentum_pred": save_momentum(agent_state.spr_momentum_pred),
                "spr_momentum_trans": save_momentum(agent_state.spr_momentum_trans),
                "traj_buffer": flax.serialization.to_bytes(traj_buffer),
                "rng": rng,
                "episodes": episodes,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "wandb_run_id": wandb.run.id if wandb.run else None,
            }

            checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
            tmp_path = checkpoint_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            os.rename(tmp_path, checkpoint_path)
            # print(f"Resume checkpoint saved to {checkpoint_path}")

        # Periodic model checkpointing
        if (
            args.periodic_checkpointing
            and t % checkpointing_frequency == 0
            and t > 0
            and args.save_model
        ):
            model_path = f"{args.log_dir}/checkpoint_{t}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([agent_state.train_state.params]))
            print(f"Checkpoint saved to {model_path}")

        obs = next_obs

    # Final model saving
    if args.save_model:
        model_path = f"{args.log_dir}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes([agent_state.train_state.params]))
        print(f"Final model saved to {model_path}")

    # Save logs to CSV
    if all_logs:
        flush_logs(all_logs, args.log_dir)
        print(f"Logs saved to: {os.path.join(args.log_dir, 'training_data.csv')}")

    return env


def define_metrics():
    wandb.define_metric("global_step")
    wandb.define_metric("episodes")
    wandb.define_metric("charts/*", step_metric="global_step")


def main(
    experiment: Callable,
    agent: Agent,
    define_metrics: Callable[[None], None],
):
    import tyro

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    args.log_dir = os.path.join(args.log_dir, run_name)
    # Check if SCRATCH environment variable exists
    if "SCRATCH" in os.environ:
        args.log_dir = os.path.join(os.environ["SCRATCH"], "streamRepL", args.log_dir)
        print(f"Using SCRATCH directory for logs: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    wandb_id = wandb.util.generate_id()
    if args.resume:
        checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint to resume: {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            wandb_id = checkpoint_data.get("wandb_run_id")

    # Save config to JSON
    import json

    config_path = os.path.join(args.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to: {config_path}")

    print(f"Run name: {run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
            id=wandb_id,
            resume="allow",
        )
    else:
        wandb.init(mode="disabled")

    define_metrics()

    start_time = time.time()
    env = jax.block_until_ready(experiment(args, agent, run_name))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
    total_steps = args.total_timesteps
    wandb.run.summary["SPS"] = int(total_steps / elapsed_time)
    wandb.finish()


if __name__ == "__main__":
    main(
        experiment,
        StreamQAgent,
        define_metrics,
    )
