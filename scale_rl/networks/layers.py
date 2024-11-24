from typing import Any

from flax import nnx
import jax.numpy as jnp

class MLPBlock(nnx.Module):
    def __init__(self, din: int, dout: int, dtype: Any, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dout, dtype=dtype, rngs=rngs)
        self.linear2 = nnx.Linear(dout, dout, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nnx.relu(self.linear2(nnx.relu(self.linear1(x))))

class ResidualBlock(nnx.Module):
    def __init__(self, din: int, dout:int, dtype: Any, *, rngs: nnx.Rngs):
        self.layernorm = nnx.LayerNorm(num_features=din, rngs=rngs, dtype=dtype)
        self.linear1 = nnx.Linear(din, dout * 4, dtype=dtype, rngs=rngs)
        self.linear2 = nnx.Linear(dout * 4, dout, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.linear2(nnx.relu(self.linear1(self.layernorm(x))))
