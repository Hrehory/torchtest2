"""Plugin for PyTorch Lightning ``LightningModule`` s."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

from torchtest2.plugin_base import LossFn, Plugin


class LightningPlugin(Plugin):
    """Supports common ``LightningModule`` patterns without reproducing the Trainer loop:
    training passes call ``module.training_step(batch, 0)`` and use the loss it returns (a tensor
    or a dict with ``"loss"``); eval passes call ``module(inputs)``; the optimizer, if you pass
    none, is the (first) one from ``module.configure_optimizers()``.

    Not emulated: Trainer strategies and distributed execution, precision plugins / autocast,
    manual optimization, multiple optimizers and optimizer indices, schedulers, gradient
    accumulation, callbacks and hooks, and other Trainer-managed state.

    Usage::

        check_model(module, x, y, plugin="lightning")

    Arguments:
        make_batch: builds the batch given to ``training_step`` from inputs and targets;
            defaults to ``(inputs, targets)``.
    """

    name = "lightning"

    def __init__(self, make_batch: Callable[[Any, Any], Any] | None = None) -> None:
        self.make_batch = make_batch

    def forward(self, model: nn.Module, inputs: Any, targets: Any) -> Any:
        batch = (inputs, targets) if self.make_batch is None else self.make_batch(inputs, targets)
        with warnings.catch_warnings():
            # self.log() outside a Trainer is a harmless no-op here
            warnings.filterwarnings("ignore", message=r"You are trying to `self.log\(\)`")
            return model.training_step(batch, 0)  # type: ignore[operator]

    def loss(self, outputs: Any, targets: Any, loss_fn: LossFn | None) -> torch.Tensor | None:
        if loss_fn is not None:
            return loss_fn(outputs, targets)
        if isinstance(outputs, Mapping):
            return outputs["loss"]
        return outputs

    def configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer | None:
        from torchtest2.plugins import _first_optimizer

        return _first_optimizer(model.configure_optimizers())  # type: ignore[operator]
