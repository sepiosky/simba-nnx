from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from flax.training import dynamic_scale as dynamic_scale_lib

from scale_rl.networks.utils import tree_norm

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Trainer:
    model: nnx.Module = flax.struct.field(pytree_node=False)
    optimizer: nnx.Optimizer = flax.struct.field(pytree_node=False)
    update_step: int = 0
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    """
    dataclass decorator makes custom class to be passed safely to Jax.
    https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

    Trainer class wraps network & optimizer to easily optimize the network under the hood.

    args:
        model: model (nnx.module).
        tx: optimizer (e.g., optax.Adam).
        opt_state: current state of the optimizer (e.g., beta_1 in Adam).
        update_step: number of update step so far.
    """

    @classmethod
    def create(
        cls,
        model: nnx.Module,
        tx: Optional[optax.GradientTransformation] = None,
        dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None,
    ) -> "Trainer":

        network = cls(
            model=model,
            optimizer=None if tx is None else nnx.Optimizer(model, tx),
            dynamic_scale=dynamic_scale,
        )

        return network

    #TODO check if nnx.jits are needed

    @nnx.jit
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @nnx.jit
    def apply(self, *args, **kwargs): # with nnx there is no need for apply but for compatability its keeped for now
        return self.model(*args, **kwargs)

    @nnx.jit
    def apply_gradient(self, loss_fn) -> Tuple[Any, "Trainer"]:
        if self.dynamic_scale:
            raise NotImplementedError("Dynamic scale is not implemented yet.")
        else:
            grad_fn = nnx.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.model)
            dynamic_scale = None

        grad_norm = tree_norm(grads)
        info["grad_norm"] = grad_norm

        self.optimizer.update(grads) #TODO dynamic scale apply
        self.update_step += 1
        self.dynamic_scale = dynamic_scale

        return info
