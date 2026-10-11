"""Plugins: reusable adapters for model families whose training or inference is not simply
``model(inputs)`` + ``loss_fn(outputs, targets)``.

A plugin bundles what would otherwise be passed as separate ``forward_fn``, ``eval_forward_fn``,
``loss_fn``, ``batch_dim`` and ``optimizer`` arguments, so it can be reused across tests and
projects, shipped as a package and registered under the ``torchtest2.plugins`` entry-point group.
Every method has a sensible default; override only what your models need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from torchtest2._tree import call_model, extract_loss

LossFn = Callable[..., torch.Tensor]


class Plugin:
    """Base class of torchtest2 plugins.

    A plugin tells torchtest2 how a family of models is run; the checks stay the same. For every
    training-mode pass (loss, gradients, optimizer steps, overfitting) torchtest2 does::

        train_out = plugin.forward(model, inputs, targets)    # whatever the model returns
        loss = plugin.loss(train_out, targets, loss_fn)      # -> one scalar, then backward()

    so the first argument of :meth:`loss` is exactly what :meth:`forward` returned - logits, a
    dict of losses, a Hugging Face output object - and an override may name it after what it
    holds (e.g. ``losses``). For every eval-mode pass it calls
    ``plugin.eval_forward(model, inputs)``.

    Attributes:
        name: the name under which the plugin can be selected (``plugin="name"``).
        batch_dim: the batch dimension of inputs and outputs, or ``None`` when the inputs have
            no batch dimension (e.g. a list of images of different sizes); the batch checks are
            then skipped.
        output_batch_dim: the batch dimension of the outputs when it differs from
            ``batch_dim`` (``None``: the same).
    """

    name: str = ""
    batch_dim: int | None = 0
    output_batch_dim: int | None = None

    def split_batch(self, batch: Any) -> dict[str, Any]:
        """Split a ``ModelTester``/``TestSuite`` ``batch`` into ``{"inputs": ..., "targets": ...}``
        (optionally with ``"batch_dim"``). Not used by :func:`torchtest2.check_model`, which
        receives inputs and targets separately."""
        return {"inputs": batch[0], "targets": batch[1]}

    def forward(self, model: nn.Module, inputs: Any, targets: Any) -> Any:
        """Forward pass in training mode. Its return value, whatever it is, is passed unchanged
        to :meth:`loss` as the first argument."""
        return call_model(model, inputs)

    def eval_forward(self, model: nn.Module, inputs: Any) -> Any:
        """Forward pass in evaluation mode (the outputs that output/batch checks inspect)."""
        return call_model(model, inputs)

    def loss(self, outputs: Any, targets: Any, loss_fn: LossFn | None) -> torch.Tensor | None:
        """The training loss as one scalar. ``outputs`` is the return value of :meth:`forward`
        (not necessarily model outputs: a model may return its losses in training mode);
        ``loss_fn`` is the one given by the user, if any. The default uses ``loss_fn`` or,
        without one, ``outputs.loss``."""
        if loss_fn is not None:
            return loss_fn(outputs) if targets is None else loss_fn(outputs, targets)
        return extract_loss(outputs)

    def configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer | None:
        """The optimizer to use when the user passes none (``None`` = torchtest2's default)."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def overrides(plugin: Plugin, method: str) -> bool:
    """Whether ``plugin`` overrides ``method`` of :class:`Plugin`."""
    return getattr(type(plugin), method) is not getattr(Plugin, method)


class PluginBase(Plugin, ABC):
    """The 2021 plugin interface, kept so that existing plugins keep working.

    Its methods are mapped onto :class:`Plugin`: ``train_inference`` -> :meth:`Plugin.forward`,
    ``test_inference`` -> :meth:`Plugin.eval_forward`, ``compute_loss(loss_criterion, outputs,
    targets)`` -> :meth:`Plugin.loss`. New plugins should subclass :class:`Plugin` directly.
    """

    @abstractmethod
    def compute_loss(self, loss_criterion: Any, model_outputs: Any, targets: Any) -> torch.Tensor:
        """Compute the loss, e.g. ``return loss_criterion(model_outputs, targets)``."""

    @abstractmethod
    def train_inference(self, model: nn.Module, inputs: Any) -> Any:
        """Forward pass in training mode, e.g. ``return model(inputs[0], inputs[1])``."""

    @abstractmethod
    def test_inference(self, model: nn.Module, inputs: Any) -> Any:
        """Forward pass in evaluation mode."""

    @abstractmethod
    def split_batch(self, batch: Any) -> dict:
        """Return ``{"inputs": ..., "targets": ..., "batch_size": ..., "batch_dim": ...}``."""

    def mask_outputs(self, model_outputs: Any, mask_idx: int, batch_dimension: int) -> Any:
        """Not used any more: batch independence is now tested by perturbing an input sample,
        which works for any output structure. Kept so that existing subclasses still work."""
        raise NotImplementedError

    def forward(self, model: nn.Module, inputs: Any, targets: Any) -> Any:
        return self.train_inference(model, inputs)

    def eval_forward(self, model: nn.Module, inputs: Any) -> Any:
        return self.test_inference(model, inputs)

    def loss(self, outputs: Any, targets: Any, loss_fn: LossFn | None) -> torch.Tensor | None:
        return self.compute_loss(loss_fn, outputs, targets)


class _DuckTypedPlugin(PluginBase):
    """Wraps an object that has the 2021 plugin methods but does not subclass PluginBase."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.name = getattr(obj, "name", "") or type(obj).__name__

    def compute_loss(self, loss_criterion: Any, model_outputs: Any, targets: Any) -> torch.Tensor:
        return self.obj.compute_loss(loss_criterion, model_outputs, targets)

    def train_inference(self, model: nn.Module, inputs: Any) -> Any:
        return self.obj.train_inference(model, inputs)

    def test_inference(self, model: nn.Module, inputs: Any) -> Any:
        return self.obj.test_inference(model, inputs)

    def split_batch(self, batch: Any) -> dict:
        return self.obj.split_batch(batch)


_LEGACY_METHODS = ("compute_loss", "train_inference", "test_inference", "split_batch")


def as_plugin(obj: Any) -> Plugin:
    """Return ``obj`` as a :class:`Plugin`: plugins as they are, 2021-style objects wrapped."""
    if isinstance(obj, Plugin):
        return obj
    if all(callable(getattr(obj, m, None)) for m in _LEGACY_METHODS):
        return _DuckTypedPlugin(obj)
    raise TypeError(
        f"{obj!r} is not a torchtest2 plugin: subclass torchtest2.Plugin (or implement "
        f"{', '.join(_LEGACY_METHODS)})"
    )


class DefaultPlugin(PluginBase):
    """The 2021 default plugin: ``batch = [inputs, targets]`` and ``model(inputs)``."""

    def compute_loss(self, loss_criterion: Any, model_outputs: Any, targets: Any) -> torch.Tensor:
        return loss_criterion(model_outputs, targets)

    def train_inference(self, model: nn.Module, inputs: Any) -> Any:
        return model(inputs)

    def test_inference(self, model: nn.Module, inputs: Any) -> Any:
        return model(inputs)

    def split_batch(self, batch: Any) -> dict:
        return {
            "inputs": batch[0],
            "targets": batch[1],
            "batch_size": batch[0].shape[0],
            "batch_dim": 0,
        }
