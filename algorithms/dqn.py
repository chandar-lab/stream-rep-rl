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
import csv
import flax

from utils import tree
from networks.MLP import sparse_init
from networks.value_networks import (
    DenseQNetwork,
    AtariQNetwork,
    OctaxQNetwork,
    MinAtarQNetwork,
)
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
    gamma: float = 0.99
    """the discount factor gamma"""
    start_epsilon: float = 1.0
    """the starting epsilon for exploration"""
    end_epsilon: float = 0.01
    """the ending epsilon for exploration"""
    explore_frac: float = 0.10
    """the fraction of total-timesteps for epsilon annealing"""
    sparse_init: float = 0.9
    """the sparsity of the neural network weights"""
    layer_norm: bool = True
    """whether to use layer normalization in the networks"""
    activation: str = "leaky_relu"
    """the activation function to use in the networks"""
    opt: str = "sgd"
    """optimizer to use"""
    net_arch: str = "atari3"
    """network architecture"""
    mlp_layers: Optional[List[int]] = None
    """MLP hidden layers"""
    periodic_checkpointing: bool = False
    """whether to save periodic checkpoints during training"""
    resume: bool = False
    """whether to resume from the last checkpoint if available"""
    use_spr: bool = False
    """whether to use SPR (Self-Predictive Representations)"""
    buffer_size: int = 1
    """the replay memory buffer size"""
    batch_size: int = 1
    """the batch size of sample from the reply memory"""
    learning_starts: int = 100
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    target_network_frequency: int = 1000
    """the frequency of updates for the target nerwork"""
    replay_buffer_type: str = "standard"
    """which replay buffer to use: standard or minred"""
    minred_alpha: float = 0.95
    """moving average coefficient for MinRed latent tracking"""


class ReplayBuffer:
    requires_latent: bool = False

    def __init__(self, buffer_size, obs_shape, action_dim, obs_dtype=np.float32):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, *obs_shape), dtype=obs_dtype)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.pos = 0
        self.full = False
        self.size = 0

    def _store_transition(self, idx, obs, action, reward, next_obs, done):
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done

    def add(self, obs, action, reward, next_obs, done):
        self._store_transition(self.pos, obs, action, reward, next_obs, done)
        self.size = min(self.size + 1, self.buffer_size)
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.size == self.buffer_size

    def _sample_indices(self, batch_size):
        effective_size = self.buffer_size if self.full else self.size
        if effective_size == 0:
            raise ValueError("Replay buffer is empty.")
        return np.random.randint(0, effective_size, size=batch_size)

    def _gather(self, indices):
        return (
            self.obs[indices],
            self.actions[indices],
            self.next_obs[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def sample(self, batch_size):
        indices = self._sample_indices(batch_size)
        return self._gather(indices)


class MinRedReplayBuffer(ReplayBuffer):
    requires_latent: bool = True

    def __init__(
        self,
        buffer_size,
        obs_shape,
        action_dim,
        obs_dtype=np.float32,
        alpha: float = 0.95,
    ):
        super().__init__(buffer_size, obs_shape, action_dim, obs_dtype)
        self.alpha = alpha
        self.latents = None
        self.latent_dim = None

    def _flatten_latent(self, latent):
        latent_np = np.asarray(latent, dtype=np.float32)
        return latent_np.reshape(-1)

    def _ensure_latent_storage(self, latent_vec):
        if self.latents is None:
            self.latent_dim = latent_vec.shape[0]
            self.latents = np.zeros(
                (self.buffer_size, self.latent_dim), dtype=np.float32
            )

    def _find_redundant_index(self):
        if self.latents is None:
            return self.pos
        latent_matrix = self.latents if self.full else self.latents[: self.size]
        norms = np.linalg.norm(latent_matrix, axis=1, keepdims=True) + 1e-8
        normalized = latent_matrix / norms
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        nearest_similarity = np.max(similarity, axis=1)
        redundant_local_idx = int(np.argmax(nearest_similarity))
        return redundant_local_idx

    def add(self, obs, action, reward, next_obs, done, *, latent):
        latent_vec = self._flatten_latent(latent)
        self._ensure_latent_storage(latent_vec)
        if not self.full:
            insert_idx = self.pos
            self._store_transition(insert_idx, obs, action, reward, next_obs, done)
            self.latents[insert_idx] = latent_vec
            self.size = min(self.size + 1, self.buffer_size)
            self.pos = (self.pos + 1) % self.buffer_size
            self.full = self.size == self.buffer_size
        else:
            insert_idx = self._find_redundant_index()
            self._store_transition(insert_idx, obs, action, reward, next_obs, done)
            self.latents[insert_idx] = latent_vec

    def sample(self, batch_size, feature_fn=None):
        indices = self._sample_indices(batch_size)
        batch = self._gather(indices)
        if feature_fn is not None and self.latents is not None:
            latents = feature_fn(batch[0])
            latents = np.asarray(latents, dtype=np.float32)
            if latents.ndim == 1:
                latents = latents[None]
            latents = latents.reshape(latents.shape[0], -1)
            update = self.alpha * self.latents[indices] + (1.0 - self.alpha) * latents
            self.latents[indices] = update
        return batch


LATENT_METHODS = {
    "atari3": AtariQNetwork.get_online_latent,
}


def compute_latent_vectors(
    agent_state: "AgentState", observations: np.ndarray
) -> np.ndarray:
    net_arch = agent_state.agent_config.net_arch
    if net_arch not in LATENT_METHODS:
        raise ValueError(
            "Selected network architecture does not expose a latent representation."
        )
    method = LATENT_METHODS[net_arch]
    obs_np = np.asarray(observations)
    obs_tensor = jnp.array(obs_np)
    latents = agent_state.train_state.apply_fn(
        agent_state.train_state.params,
        obs_tensor,
        method=method,
    )
    latents = np.asarray(latents)
    batch = latents.shape[0] if latents.ndim > 1 else 1
    return latents.reshape(batch, -1)


def compute_single_latent(agent_state: "AgentState", obs: np.ndarray) -> np.ndarray:
    obs_batch = np.expand_dims(np.asarray(obs), axis=0)
    return compute_latent_vectors(agent_state, obs_batch)[0]


def build_replay_buffer(
    args: Args, obs_shape: tuple, action_dim: int, obs_dtype: np.dtype
) -> ReplayBuffer:
    buffer_type = getattr(args, "replay_buffer_type", "standard").lower()
    if buffer_type == "minred":
        if args.net_arch not in LATENT_METHODS:
            raise ValueError(
                "Minimum Redundancy Buffer requires a network with latent support (e.g., net_arch='atari3')."
            )
        return MinRedReplayBuffer(
            args.buffer_size,
            obs_shape,
            action_dim,
            obs_dtype=obs_dtype,
            alpha=args.minred_alpha,
        )
    if buffer_type == "standard":
        return ReplayBuffer(
            args.buffer_size, obs_shape, action_dim, obs_dtype=obs_dtype
        )
    raise ValueError(f"Unknown replay_buffer_type: {args.replay_buffer_type}")


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

        def thunk():
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

        return thunk

    elif args.env_type == "octax":

        def thunk():
            print(f"Creating Octax environment: {args.env_id}")
            env = OctaxToGymAdapter(args.env_id, seed=args.seed)

            env = RecordEpisodeStatisticsJAX(env)
            env = ScaleRewardJAX(env, gamma=args.gamma)
            env = NormalizeObservationJAX(env)

            return env

        return thunk

    elif args.env_type == "minatar":

        def thunk():
            print(f"Creating standard Gym Minatar environment: {args.env_id}")
            env = gym.make(args.env_id)

            # Episode statistics wrapper
            env = RecordEpisodeStatistics(env)
            env = StoreEpisodeReturnsAndLengths(env)

            # Normalization must come after stats for correct results
            env = normalization.ScaleReward(env, gamma=args.gamma)
            env = normalization.NormalizeObservation(env)

            return env

        return thunk

    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")


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
class AgentState(NamedTuple):
    agent_config: Config
    train_state: TrainState
    target_params: flax.core.FrozenDict


@partial(jax.jit, static_argnames=["action_dim"])
def agent_step(
    agent_state: AgentState,
    obs: jnp.ndarray,
    action_dim: int,
    epsilon: float,
    rng: jax.random.PRNGKey,
):
    params = agent_state.train_state.params
    # Handle both single observation and batch of observations (if vmapped)
    # But usually agent_step is called for action selection in environment loop
    # If obs has batch dim, expand_dims might be needed or not depending on network
    # The existing code expects single obs
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


def init_agent_state_dqn_agent(
    agent_config: Config, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey
):
    net_kwargs = {
        "action_dim": action_dim,
        "layer_norm": agent_config.layer_norm,
        "activation": agent_config.activation,
        "kernel_init": sparse_init(sparsity=agent_config.sparse_init),
    }
    net_arch = agent_config.net_arch
    if net_arch == "mlp":
        net_kwargs["hiddens"] = agent_config.mlp_layers

    # Select network based on env_type
    env_type = getattr(agent_config, "env_type", "atari")
    if env_type == "atari":
        network = AtariQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    elif env_type == "minatar":
        network = MinAtarQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    elif env_type == "octax":
        network = OctaxQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    elif net_arch == "mlp":
        network = DenseQNetwork(**net_kwargs)
        init_x = jnp.zeros(obs_shape)
    else:
        raise ValueError(f"unknown env_type: {env_type}")

    rng, _rng = jax.random.split(rng)
    params = network.init(_rng, init_x)

    tx = getattr(optax, agent_config.opt)(agent_config.q_lr)

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    def params_sum(params):
        return sum(
            jax.tree_util.tree_leaves(jax.tree.map(lambda x: np.prod(x.shape), params))
        )

    print(f"Total number of params: {params_sum(train_state.params)}")

    return (
        AgentState(agent_config, train_state, params),
        rng,
    )


## Updates for DQN agent
@partial(jax.jit, static_argnames=[])
def update_step_dqn_agent(agent_state, transition):
    obs, actions, next_obs, rewards, dones = transition

    config = agent_state.agent_config
    q_train_state = agent_state.train_state
    target_params = agent_state.target_params
    q_params = q_train_state.params

    # Calculate target values
    next_q_values = q_train_state.apply_fn(target_params, next_obs)
    max_next_q = jnp.max(next_q_values, axis=-1)
    targets = rewards + config.gamma * max_next_q * (1 - dones)

    def loss_fn(params):
        q_values = q_train_state.apply_fn(params, obs)
        q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
        loss = jnp.mean((targets - q_action) ** 2)
        return loss, q_action

    (loss, q_vals), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_params)

    new_train_state = q_train_state.apply_gradients(grads=grads)

    metrics = {
        "td_error": loss * 2,  # approx
        "q_val": jnp.mean(q_vals),
        "loss": loss,
        "q_update_l2": optax.global_norm(grads),
        "h_val": 0.0,
        "h_update_l2": 0.0,
    }

    return (
        AgentState(config, new_train_state, target_params),
        metrics,
    )


QRCAgent = Agent(init_agent_state_dqn_agent, agent_step, update_step_dqn_agent)


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
    env = make_env(args, 0, run_name)()
    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs)  # Convert LazyFrames to numpy array
    # Handle observation transposing for atari (C, H, W) -> (H, W, C)
    if args.env_type == "atari" and obs.shape == (4, 84, 84):
        obs = obs.transpose(1, 2, 0)
    episodes = 0
    episode_return = 0.0
    episode_length = 0

    # Initialize the agent
    action_dim = int(env.action_space.n)
    agent_state, rng = agent.init_state(agent_config, action_dim, obs.shape, rng)
    epsilon_schedule = get_linear_epsilon_schedule(args)

    # Tracking for logging
    start_time = time.time()
    last_log_step = 0
    metrics = None

    # Initialize Replay Buffer
    rb = build_replay_buffer(args, obs.shape, action_dim, obs_dtype=obs.dtype)

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

            def restore_train_state(current, state_bytes):
                if current is None or state_bytes is None:
                    return current
                try:
                    params = flax.serialization.from_bytes(
                        current.params, state_bytes["params"]
                    )
                    opt_state = flax.serialization.from_bytes(
                        current.opt_state, state_bytes["opt_state"]
                    )
                    return current.replace(
                        params=params, opt_state=opt_state, step=checkpoint_data["step"]
                    )
                except Exception as e:
                    print(f"Failed to restore state component: {e}")
                    return current

            new_train_state = restore_train_state(
                agent_state.train_state, checkpoint_data["train_state"]
            )
            new_target_params = flax.serialization.from_bytes(
                agent_state.target_params, checkpoint_data["target_params"]
            )

            agent_state = AgentState(
                agent_config=agent_state.agent_config,
                train_state=new_train_state,
                target_params=new_target_params,
            )

            if "rb" in checkpoint_data:
                rb = pickle.loads(checkpoint_data["rb"])

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

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.array(next_obs)  # Convert LazyFrames to numpy array
        # Handle observation transposing for atari (C, H, W) -> (H, W, C)
        if args.env_type == "atari" and next_obs.shape == (4, 84, 84):
            next_obs = next_obs.transpose(1, 2, 0)
        done = terminated or truncated

        episode_return += reward
        episode_length += 1

        # Handle final observation for truncated episodes
        real_next_obs = next_obs.copy()

        # Add to Replay Buffer
        if getattr(rb, "requires_latent", False):
            latent = compute_single_latent(agent_state, obs)
            rb.add(obs, action, reward, real_next_obs, done, latent=latent)
        else:
            rb.add(obs, action, reward, real_next_obs, done)

        # Training
        if t > args.learning_starts and t % args.train_frequency == 0:
            if getattr(rb, "requires_latent", False):
                feature_fn = lambda batch_obs: compute_latent_vectors(
                    agent_state, batch_obs
                )
                data = rb.sample(args.batch_size, feature_fn=feature_fn)
            else:
                data = rb.sample(args.batch_size)
            # data is (obs, actions, next_obs, rewards, dones)
            agent_state, metrics = agent.update(agent_state, data)

        # Target network update
        if t > args.learning_starts and t % args.target_network_frequency == 0:
            agent_state = agent_state._replace(
                target_params=agent_state.train_state.params
            )

        if done:
            episodes += 1

            episode_return = 0.0
            episode_length = 0

            next_obs, info = env.reset()
            next_obs = np.array(next_obs)  # Convert LazyFrames to numpy array
            # Handle observation transposing for atari (C, H, W) -> (H, W, C)
            if args.env_type == "atari" and next_obs.shape == (4, 84, 84):
                next_obs = next_obs.transpose(1, 2, 0)

        # Check if training is complete (moved outside the if done block)
        if t >= args.total_timesteps:
            break

        # Periodic logging every 1000 steps
        if t % 1000 == 0 and t > 0 and metrics is not None:
            steps_elapsed = t - last_log_step
            sps = int(steps_elapsed / (time.time() - start_time))

            avg_return = np.mean(env.get_wrapper_attr("return_queue"))
            avg_length = np.mean(env.get_wrapper_attr("length_queue"))

            print(
                f"Step: {t}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}, SPS: {sps}, Epsilon: {epsilon:.3f}"
            )

            # Create log dictionary
            log_dict = {
                "global_step": t,
                "avg_return": avg_return,
                "avg_length": avg_length,
                "td_loss": float(metrics["td_error"]),
                "q_values": float(metrics["q_val"]),
                "q_update_l2": float(metrics["q_update_l2"]),
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
                    "updates/q_update_l2": log_dict["q_update_l2"],
                    "episodes": episodes,
                },
                step=t,
            )

            # Accumulate log_dict for CSV export
            all_logs.append(log_dict.copy())

            last_log_step = t
            start_time = time.time()

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

            checkpoint_data = {
                "step": t,
                "train_state": save_train_state(agent_state.train_state),
                "target_params": flax.serialization.to_bytes(agent_state.target_params),
                "rb": pickle.dumps(rb),
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

        # Periodic model checkpointing
        if (
            args.periodic_checkpointing
            and t % checkpointing_frequency == 0
            and t > 0
            and args.save_model
        ):
            model_path = f"{args.log_dir}/checkpoint_{t}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(agent_state.train_state.params))
            print(f"Checkpoint saved to {model_path}")

        obs = next_obs

    # Final model saving
    if args.save_model:
        model_path = f"{args.log_dir}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(agent_state.train_state.params))
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
        QRCAgent,
        define_metrics,
    )
