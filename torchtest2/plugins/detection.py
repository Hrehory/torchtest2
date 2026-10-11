"""Plugin for torchvision detection models (Faster/Mask/Keypoint R-CNN, RetinaNet, FCOS, SSD)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchtest2.plugin_base import LossFn, Plugin


class TorchvisionDetectionPlugin(Plugin):
    """torchvision detection models take ``images`` (a list of ``(C, H, W)`` tensors) and, in
    training mode, ``targets`` (a list of dicts with ``boxes``, ``labels``, ...), and return a
    dict of losses; in eval mode they return a list of predictions.

    Usage::

        check_model(model, images, targets, plugin="torchvision_detection")

    The loss is the sum of the loss dict (or ``loss_fn(loss_dict, targets)`` if given). The
    images are a list, not a batched tensor, so the batch checks are skipped.
    """

    name = "torchvision_detection"
    batch_dim = None

    def split_batch(self, batch: Any) -> dict[str, Any]:
        return {"inputs": _images(batch[0]), "targets": batch[1]}

    def forward(self, model: nn.Module, inputs: Any, targets: Any) -> Any:
        return model(_images(inputs), targets)

    def eval_forward(self, model: nn.Module, inputs: Any) -> Any:
        return model(_images(inputs))

    def loss(self, outputs: Any, targets: Any, loss_fn: LossFn | None) -> torch.Tensor | None:
        if loss_fn is not None:
            return loss_fn(outputs, targets)
        return sum(outputs.values())


def _images(inputs: Any) -> list[torch.Tensor]:
    """Accept a list of images, a ``(images,)`` tuple or a batched ``(N, C, H, W)`` tensor."""
    if isinstance(inputs, torch.Tensor):
        return list(inputs)
    if isinstance(inputs, tuple) and len(inputs) == 1 and isinstance(inputs[0], list):
        return inputs[0]
    return list(inputs)
