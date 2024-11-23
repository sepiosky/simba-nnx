from typing import Any

from flax import nnx
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import MLPBlock, ResidualBlock
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.utils import orthogonal_init

class SACEncoder(nnx.Module):
    def __init__(self, block_type: str, num_blocks: int, hidden_dim: int, dtype: Any = jnp.float32, *, rngs: nnx.Rngs):
        if block_type == "mlp":
            self.layers = [
                MLPBlock(hidden_dim, dtype=dtype, rngs=rngs)
            ]
        else:
            self.layers = [
                nnx.Linear(hidden_dim, dtype=dtype, rngs=rngs),
                *[ResidualBlock(hidden_dim, dtype=dtype, rngs=rngs) for _ in range(num_blocks)],
                nnx.LayerNorm(dtype=dtype, rngs=rngs)
            ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class SACActor(nnx.Module):
    block_type: str
    num_blocks: int
    hidden_dim: int
    action_dim: int
    dtype: Any

    def __init__(self, block_type: str, num_blocks: int, hidden_dim: int, action_dim: int, dtype: Any = jnp.float32, *, rngs: nnx.Rngs):
        self.encoder = SACEncoder(
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dtype=dtype,
            rngs=rngs
        )
        self.dtype = dtype
        self.predictor = NormalTanhPolicy(action_dim, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfp.distributions.Distribution:
        observations = convert_element_type(observations, self.dtype)
        z = self.encoder(observations)
        dist = self.predictor(z, temperature)
        return dist


class SACCritic(nnx.Module):
    def __init__(self, block_type: str, num_blocks: int, hidden_dim: int, dtype: Any = jnp.float32, *, rngs: nnx.Rngs):
        self.encoder = SACEncoder(
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dtype=dtype,
            rngs=rngs
        )
        self.dtype = dtype
        self.predictor = LinearCritic(din=hidden_dim, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z = self.encoder(inputs)
        q = self.predictor(z)
        return q


class SACClippedDoubleCritic(nnx.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    def __init__(self, block_type: str, num_blocks: int, hidden_dim: int, dtype: Any = jnp.float32, num_qs: int = 2, *, rngs: nnx.Rngs):
        #state_axes = nnx.StateAxes({nnx.Param: 0, nnx.Rng: 0})
        VmapCritic =nnx.split_rngs(nnx.vmap(
            SACCritic,
            in_axes=None,
            out_axes=0,
            axis_size=num_qs
        ), splits=num_qs) #Question: why VMAP two critics ?
        self.critics = VmapCritic(
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dtype=dtype,
            rngs=rngs
        )
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.critics(observations, actions)


class SACTemperature(nnx.Module):
    def __init__(self, initial_value: float = 1.0):
        self.log_temp = nnx.Param(name="log_temp",value=jnp.full(shape=(), fill_value=jnp.log(initial_value)))
    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_temp)
