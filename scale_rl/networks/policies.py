"""
Implementation of commonly used policies that can be shared across agents.
"""
from typing import Any

from flax import nnx
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.networks.utils import orthogonal_init

class NormalTanhPolicy(nnx.Module):

    def __init__(
        self,
        action_dim: int,
        state_dependent_std: bool = True,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        dtype: int = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.means = nnx.Linear(action_dim, dtype=dtype, rngs=rngs)
        self.log_stds = nnx.Linear(action_dim, dtype=dtype, rngs=rngs) if state_dependent_std else nnx.Parameter(jnp.zeros(action_dim), dtype=dtype)
        self.log_stds_min = log_std_min
        self.log_stds_max = log_std_max

    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfp.distributions.Distribution:

        means = self.means(inputs)

        log_stds = self.log_stds(inputs)
        # suggested by Ilya for stability
        log_stds = self.log_stds_min + (self.log_stds_max - self.log_stds_min) * 0.5 * (1 + jnp.tanh(log_stds))

        # N(mu, exp(log_sigma))
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=tfp.bijectors.Tanh())

        return dist


class TanhPolicy(nnx.Module):
    def __init__(self, action_dim: int, dtype: int = jnp.float32, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(action_dim, dtype=dtype, rngs=rngs)

    @nnx.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> tfp.distributions.Distribution:
        return nnx.tanh(self.linear(inputs))
