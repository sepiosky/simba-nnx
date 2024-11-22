"""
Implementation of commonly used critics that can be shared across agents.
"""
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from scale_rl.networks.utils import orthogonal_init


class LinearCritic(nnx.Module):
    def __init__(self, din: int, dtype: int = jnp.float32, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, 1, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


class CategoricalCritic(nnx.Module):
    """
    C51: https://arxiv.org/pdf/1707.06887
    """
    def __init__(self, din: int, num_bins: int = 51, dtype: int = jnp.float32, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, num_bins, dtype=dtype, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:        # return log probability of bins
        return jax.nnx.log_softmax(self.linear(inputs), axis=1)


class QuantileRegressionCritic(nnx.Module):
    """
    QR-Qnet: https://arxiv.org/pdf/1806.06923
    """

    def __init__(self, din: int, num_quantiles: int = 32, dtype: int = jnp.float32, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, num_quantiles, dtype=dtype, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.linear(inputs)


def compute_categorical_bin_values(
    num_bins: int, min_v: float, max_v: float
) -> jnp.ndarray:
    return (
        jnp.arange(num_bins, dtype=jnp.float32) * (max_v - min_v) / (num_bins - 1)
        + min_v
    ).reshape((1, -1))


def compute_categorical_loss(
    log_probs: jnp.ndarray,
    gamma: float,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    target_log_probs: jnp.ndarray,
    entropy: jnp.ndarray,
    min_v: float,
    max_v: float,
) -> jnp.ndarray:
    _, num_bins = log_probs.shape

    # compute target value buckets
    bin_values = compute_categorical_bin_values(num_bins, min_v, max_v)
    target_bin_values = jnp.clip(
        reward + gamma * (bin_values - entropy) * (1 - done), min_v, max_v
    )

    # update indices
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    u = jnp.ceil(b)
    u_mask = jax.nn.one_hot(u.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    # target label
    m_l = (target_log_probs * (1 - (b - l))).reshape((-1, num_bins, 1))
    m_u = (target_log_probs * (b - l)).reshape((-1, num_bins, 1))
    m = jax.lax.stop_gradient(jnp.sum(m_l * l_mask + m_u * u_mask, axis=1))

    # regression loss
    loss = -jnp.mean(jnp.sum(jnp.exp(m) * log_probs, axis=1))

    return loss


def compute_quantile_loss(quantile: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    batch_size, num_quantiles = quantile.shape
    _, num_target_quantiles = target.shape

    # compute taus
    steps = jnp.arange(num_quantiles, dtype=jnp.float32)
    taus = ((steps + 1) / num_quantiles).reshape((1, 1, -1))
    taus_dot = (steps / num_quantiles).reshape((1, 1, -1))
    taus_hat = (taus + taus_dot) / 2.0

    expanded_quantile = quantile.reshape((batch_size, 1, num_quantiles))
    expanded_target = target.reshape((batch_size, num_target_quantiles, 1))

    # huber loss
    diff = expanded_target - expanded_quantile
    cond = jax.lax.stop_gradient(jnp.abs(diff) < 1.0)
    huber_loss = cond * 0.5 * diff**2 + (1 - cond) * (jnp.abs(diff) - 0.5)

    delta = jax.lax.stop_gradient((expanded_target - expanded_quantile) < 0)
    element_wise_loss = jnp.abs(taus_hat - delta) * huber_loss

    return element_wise_loss.sum(axis=2).mean(axis=1)
