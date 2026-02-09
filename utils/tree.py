from functools import reduce

from jax import tree, vmap
import jax.numpy as jnp


def add(*args):
    return reduce(lambda a, b: tree.map(jnp.add, a, b), args)


def subtract(*args):
    return reduce(lambda a, b: tree.map(jnp.subtract, a, b), args)


def scale(scalar, arg):
    return tree.map(lambda x: scalar * x, arg)


def vmap_scale(vector, arg):
    f = vmap(scale, in_axes=[0, 0])
    return f(vector, arg)


def neg(arg):
    return scale(-1.0, arg)


def zeros(arg):
    return scale(0.0, arg)


def all_but_last(arg):
    return jax.tree.map(lambda x: x[:-1], arg)


def last(arg):
    return jax.tree.map(lambda x: x[-1], arg)


import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce


def l2_norm(x):
    return jnp.sqrt(
        tree_reduce(jnp.add, tree_map(lambda x: jnp.sum(x**2), x)),
    )


def l1_norm(x):
    return tree_reduce(
        jnp.add,
        tree_map(lambda x: jnp.sum(jnp.abs(x)), x),
    )


def l1_normalize(x):
    norm = l1_norm(x)
    return tree_map(lambda x: x / norm, x)


def l2_normalize(x):
    norm = l2_norm(x)
    return tree_map(lambda x: x / norm, x)
