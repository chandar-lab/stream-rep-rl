import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import abstractmethod
from typing import Callable, Iterable, Tuple

import flax.linen as nn
from flax.linen.initializers import constant
import jax.numpy as jnp
import jax

from networks.MLP import MLP, layer_norm
from networks.ActorCritic import act_funcs


## Networks for backward-view algos
class QNetwork(nn.Module):
    """Action-value function: Q(s,a)"""

    action_dim: int
    layer_norm: bool
    activation: str
    kernel_init: Callable
    bias_init: Callable = constant(0.0)
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    hidden_layer_sizes: Tuple[int] = (256,)
    dyn_features_list: Tuple[int] = (64, 64)
    dyn_kernel_sizes: Tuple[Tuple[int, int] | int] = ((3, 3), (3, 3))
    dyn_strides_list: Tuple[Tuple[int, int] | int] = ((1, 1), (1, 1))

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class DenseQNetwork(QNetwork):
    hiddens: Iterable[int] = ()

    @nn.compact
    def __call__(self, x):
        no_batch_dim = x.ndim == 1
        if no_batch_dim:
            x = x[None]
        activation = act_funcs[self.activation]
        x = MLP(
            hiddens=self.hiddens,
            act=activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            pre_act_norm=self.layer_norm,
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=constant(0.0)
        )(x)
        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x


class MinAtarQNetwork(QNetwork):
    @nn.compact
    def __call__(self, x):
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        x = nn.Conv(
            16,
            kernel_size=[3, 3],
            strides=1,
            padding="VALID",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            128,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation(x)

        x = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
        )(x)
        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x


class AtariQNetwork(QNetwork):
    @nn.compact
    def __call__(self, x):
        return self._forward(x, return_latent=False)

    @nn.compact
    def get_online_latent(self, x, use_augmentation=False, rng=None):
        # use_augmentation/rng ignored as not part of original
        return self._forward(x, return_latent=True)

    def _forward(self, x, return_latent=False):
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        def activation_fn(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            name="Conv_0",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation_fn(x)

        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            name="Conv_1",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation_fn(x)

        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            name="Conv_2",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation_fn(x)

        if return_latent:
            if no_batch_dim:
                x = jnp.squeeze(x, axis=0)
            return x

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            512, kernel_init=self.kernel_init, bias_init=constant(0.0), name="Dense_0"
        )(x)
        x = activation_fn(x)

        x = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
            name="Dense_1",
        )(x)

        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x


class OctaxQNetwork(QNetwork):
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        layers = []

        for features, kernel_size, strides in zip(
            self.features_list, self.kernel_sizes, self.strides_list
        ):
            layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            layers.append(activation)
        layers.append(lambda x: x.reshape(x.shape[0], -1))

        self.features = nn.Sequential(layers)

        # Q-value projection and head
        self.Qproj = nn.Dense(
            self.hidden_layer_sizes[0],
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.Qhead = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
        )

    @nn.compact
    def __call__(self, x):
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        x = self.features(x)

        # Q-value head
        x = self.Qproj(x)
        x = nn.relu(x)
        x = self.Qhead(x)

        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)

        return x


##########################################


class OctaxQNetworkSPR(QNetwork):
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        layers = []

        for features, kernel_size, strides in zip(
            self.features_list, self.kernel_sizes, self.strides_list
        ):
            layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            layers.append(activation)

        self.online_encoder = nn.Sequential(layers)

        # Q-value projection and head
        self.q_projection = nn.Sequential(
            [
                lambda x: x.reshape(x.shape[0], -1),  # Flatten
                nn.Dense(
                    self.hidden_layer_sizes[0],
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
            ]
        )

        self.q_head = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
        )

    @nn.compact
    def __call__(self, x):
        no_batch_dim = x.ndim == 3
        z = self.get_online_latent(x)

        # Q-value head
        x = self.q_projection(z)
        x = nn.relu(x)
        x = self.q_head(x)

        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)

        return x

    @nn.compact
    def get_online_latent(self, x, use_augmentation=False, rng=None):
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        # Apply augmentation if requested
        if use_augmentation and rng is not None:
            x = apply_augmentations(rng, x, shift_pad=4, intensity_scale=0.05)

        # Keep spatial structure - don't flatten yet
        z = self.online_encoder(x)

        return z

    @nn.compact
    def get_online_projection(self, x):
        x = self.q_projection(x)
        return x


class OctaxTransitionNetwork(nn.Module):
    action_dim: int
    layer_norm: bool
    activation: str
    kernel_init: Callable
    bias_init: Callable = constant(0.0)
    dyn_features_list: Tuple[int] = (64, 64)
    dyn_kernel_sizes: Tuple[Tuple[int, int] | int] = ((3, 3), (3, 3))
    dyn_strides_list: Tuple[Tuple[int, int] | int] = ((1, 1), (1, 1))
    features_list: Tuple[int] = (32, 64, 64)
    hidden_layer_sizes: Tuple[int] = (256,)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))

    """Transition model for SPR - predicts next latent state given current state and action."""

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        dyn_encoder_layers = []
        for features, kernel_size, strides in zip(
            self.dyn_features_list, self.dyn_kernel_sizes, self.dyn_strides_list
        ):
            dyn_encoder_layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="SAME",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            # add layer normalization
            dyn_encoder_layers.append(activation)

        self.dynamic_model = nn.Sequential(dyn_encoder_layers)

    @nn.compact
    def __call__(self, z, action):
        if z.ndim == 2:
            raise ValueError(
                f"Expected 4D latent (N,H,W,C), got 2D shape {z.shape}. Encoder should not flatten for SPR."
            )

        # One-hot encode action
        action_onehot = jnp.zeros(self.action_dim)
        action_onehot = action_onehot.at[action].set(1.0)

        # Broadcast action_onehot to match z's batch and spatial dimensions
        # z shape: (N, H, W, C), action_onehot shape: (A,)
        # We want action_onehot shape: (N, H, W, A)
        action_shape = z.shape[:-1] + (self.action_dim,)
        action_onehot_broadcast = jnp.broadcast_to(action_onehot, action_shape)

        # Concatenate latent state and action along the channel dimension
        z = jnp.concatenate([z, action_onehot_broadcast], axis=-1)
        z_next = self.dynamic_model(z)
        return z_next


class OctaxTargetEncoder(QNetwork):
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        layers = []

        for features, kernel_size, strides in zip(
            self.features_list, self.kernel_sizes, self.strides_list
        ):
            layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            layers.append(activation)

        self.target_encoder = nn.Sequential(layers)

    @nn.compact
    def __call__(self, x, use_augmentation=False, rng=None):
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        # Apply augmentation if requested
        if use_augmentation and rng is not None:
            x = apply_augmentations(rng, x, shift_pad=4, intensity_scale=0.05)

        # Keep spatial structure - don't flatten yet
        z = self.target_encoder(x)

        return z


class OctaxProjection(QNetwork):
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:

        # Q-value projection and head
        self.proj = nn.Sequential(
            [
                lambda x: x.reshape(x.shape[0], -1),  # Flatten
                nn.Dense(
                    self.hidden_layer_sizes[0],
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
            ]
        )

    @nn.compact
    def __call__(self, x):
        x = self.proj(x)
        return x


class OctaxOnlinePrediction(QNetwork):
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:

        # Q-value projection and head
        self.predictor = nn.Dense(
            self.hidden_layer_sizes[0],
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, x):
        x = self.predictor(x)
        return x


def random_shift(key, x, pad=4):
    """Apply random shift augmentation.

    Args:
        key: JAX random key
        x: Input tensor of shape (N, H, W, C)
        pad: Maximum shift in pixels (default: 4)

    Returns:
        Shifted tensor of same shape
    """
    n, h, w, c = x.shape

    # Pad the input
    padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")

    # Random crop position
    key1, key2 = jax.random.split(key)
    crop_h = jax.random.randint(key1, shape=(), minval=0, maxval=2 * pad + 1)
    crop_w = jax.random.randint(key2, shape=(), minval=0, maxval=2 * pad + 1)

    # Crop back to original size
    shifted = jax.lax.dynamic_slice(padded, (0, crop_h, crop_w, 0), (n, h, w, c))

    return shifted


def random_intensity(key, x, scale=0.05):
    """Apply random intensity scaling augmentation.

    Args:
        key: JAX random key
        x: Input tensor of shape (N, H, W, C)
        scale: Standard deviation of intensity scaling (default: 0.05)

    Returns:
        Intensity-scaled tensor of same shape
    """
    # Sample intensity multiplier from normal distribution
    intensity_scale = 1.0 + jax.random.normal(key) * scale

    # Apply intensity scaling and clip to valid range
    augmented = x * intensity_scale
    augmented = jnp.clip(augmented, 0.0, 1.0)

    return augmented


def apply_augmentations(key, x, shift_pad=4, intensity_scale=0.05):
    """Apply both random shift and intensity augmentations.

    Args:
        key: JAX random key
        x: Input tensor of shape (N, H, W, C)
        shift_pad: Maximum shift in pixels (default: 4)
        intensity_scale: Standard deviation of intensity scaling (default: 0.05)

    Returns:
        Augmented tensor of same shape
    """
    key1, key2 = jax.random.split(key)

    # Apply random shift
    x = random_shift(key1, x, pad=shift_pad)

    # Apply random intensity
    x = random_intensity(key2, x, scale=intensity_scale)

    return x


###################################################################################
# CURL Networks
###################################################################################


class OctaxQNetworkCURL(QNetwork):
    """Q-Network with separate encoder for CURL contrastive learning."""

    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    hidden_layer_sizes: Tuple[int] = (256,)
    curl_latent_dim: int = 128

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        # Online encoder (query encoder f_q)
        encoder_layers = []
        for features, kernel_size, strides in zip(
            self.features_list, self.kernel_sizes, self.strides_list
        ):
            encoder_layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            encoder_layers.append(activation)

        self.query_encoder = nn.Sequential(encoder_layers)

        # Q-value projection and head
        self.q_projection = nn.Sequential(
            [
                lambda x: x.reshape(x.shape[0], -1),  # Flatten
                nn.Dense(
                    self.hidden_layer_sizes[0],
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
            ]
        )

        self.q_head = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
        )

        # CURL latent projection
        self.curl_projection = nn.Dense(
            self.curl_latent_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, x):
        no_batch_dim = x.ndim == 3
        z = self.get_query_latent(x)

        # Initialize CURL projection (even if not used in Q-value computation)
        # This ensures the parameters are created during init
        _ = self.get_curl_embedding(z)

        # Q-value head
        x = self.q_projection(z)
        x = nn.relu(x)
        x = self.q_head(x)

        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)

        return x

    @nn.compact
    def get_query_latent(self, x, use_augmentation=False, rng=None):
        """Get latent representation from query encoder."""
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        # Apply augmentation if requested
        if use_augmentation and rng is not None:
            x = apply_augmentations(rng, x, shift_pad=4, intensity_scale=0.05)

        # Spatial features
        z = self.query_encoder(x)
        return z

    @nn.compact
    def get_curl_embedding(self, z):
        """Project encoder output to CURL embedding space."""
        # Flatten spatial features
        z_flat = z.reshape(z.shape[0], -1)
        # Project to latent dimension
        return self.curl_projection(z_flat)


class OctaxKeyEncoder(nn.Module):
    """Key encoder (f_k) for CURL - momentum-averaged version of query encoder."""

    action_dim: int
    layer_norm: bool
    activation: str
    kernel_init: Callable
    bias_init: Callable = constant(0.0)
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((4, 8), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((2, 4), (2, 2), (1, 1))
    curl_latent_dim: int = 128

    def setup(self) -> None:
        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        # Key encoder (same architecture as query encoder)
        encoder_layers = []
        for features, kernel_size, strides in zip(
            self.features_list, self.kernel_sizes, self.strides_list
        ):
            encoder_layers.append(
                nn.Conv(
                    features=features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="VALID",
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            )
            encoder_layers.append(activation)

        self.key_encoder = nn.Sequential(encoder_layers)

        # CURL latent projection (same as query)
        self.curl_projection = nn.Dense(
            self.curl_latent_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, x, use_augmentation=False, rng=None):
        """Encode observation with key encoder."""
        no_batch_dim = x.ndim == 3
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        # Apply augmentation if requested
        if use_augmentation and rng is not None:
            x = apply_augmentations(rng, x, shift_pad=4, intensity_scale=0.05)

        # Spatial features
        z = self.key_encoder(x)

        # Flatten and project to CURL embedding
        z_flat = z.reshape(z.shape[0], -1)
        return self.curl_projection(z_flat)
