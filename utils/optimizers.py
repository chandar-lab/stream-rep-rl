from functools import partial

import jax
import jax.numpy as jnp
import optax

from gtd_algos.src import tree


def obgd_with_traces(lr, gamma, lambd, kappa):
    """Overshooting-bounded Gradient Descent (ObGD)
    Compare to Pytorch implementation in
    https://github.com/mohmdelsayed/streaming-drl/blob/main/optim.py"""
    assert lr > 0.0
    assert 0.0 <= gamma <= 1.0
    assert 0.0 <= lambd <= 1.0
    assert kappa > 0.0

    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    @partial(jax.jit, static_argnames=["reset"])
    def update(grads, opt_state, params, td_error, reset):
        # Spike and decay traces
        z = opt_state
        z = jax.tree.map(lambda x: gamma * lambd * x, z)
        z = jax.tree.map(jnp.add, z, grads)

        # See Algorithm 3 of https://arxiv.org/pdf/2410.14606v2
        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        dot_product = (
            lr * kappa * delta_bar * tree.l1_norm(z)
        )  # Denoted by 'M' in the paper
        step_size = lr / jnp.maximum(dot_product, 1.0)

        # Compute update to parameters
        updates = jax.tree.map(lambda x: step_size * td_error * x, z)

        # Reset traces at end of episode or other condition
        if reset:
            z = init(params)

        opt_state = z
        return updates, opt_state

    return optax.GradientTransformationExtraArgs(init, update)


def sgd_with_traces(lr, gamma, lambd):
    assert lr > 0.0
    assert 0.0 <= gamma <= 1.0
    assert 0.0 <= lambd <= 1.0

    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    @partial(jax.jit, static_argnames=["reset"])
    def update(grads, opt_state, params, td_error, reset):
        # Spike and decay traces
        z = opt_state
        z = jax.tree.map(lambda x: gamma * lambd * x, z)
        z = jax.tree.map(jnp.add, z, grads)

        # Compute update to parameters
        updates = jax.tree.map(lambda x: lr * td_error * x, z)

        # Reset traces at end of episode or other condition
        if reset:
            z = init(params)

        opt_state = z
        return updates, opt_state

    return optax.GradientTransformationExtraArgs(init, update)
