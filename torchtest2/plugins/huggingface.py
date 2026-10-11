"""Plugin for Hugging Face ``transformers`` models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchtest2._tree import extract_loss
from torchtest2.plugin_base import LossFn, Plugin


class HuggingFacePlugin(Plugin):
    """``transformers`` models take the tokenizer output as keyword arguments and compute their
    own loss when given ``labels``.

    Usage::

        enc = tokenizer(texts, return_tensors="pt", padding=True)
        check_model(model, enc, labels, plugin="huggingface")

    Training passes call ``model(**inputs, labels=targets)`` and use ``outputs.loss``; eval passes
    call ``model(**inputs)``. Keeping the labels out of the inputs means that the input checks
    probe only the real inputs (token ids, attention mask, ...).
    """

    name = "huggingface"

    def forward(self, model: nn.Module, inputs: Any, targets: Any) -> Any:
        if targets is None:
            return _call(model, inputs)
        return _call(model, inputs, labels=targets)

    def eval_forward(self, model: nn.Module, inputs: Any) -> Any:
        return _call(model, inputs)

    def loss(self, outputs: Any, targets: Any, loss_fn: LossFn | None) -> torch.Tensor | None:
        if loss_fn is not None:
            return loss_fn(outputs) if targets is None else loss_fn(outputs, targets)
        loss = extract_loss(outputs)
        if loss is None:
            raise ValueError(
                "the model returned no loss - pass the labels as targets, or give a loss_fn"
            )
        return loss


def _call(model: nn.Module, inputs: Any, **extra: Any) -> Any:
    """Call with ``return_dict=True`` so that models configured with ``return_dict=False``
    (tuple outputs) still return an output object with ``.loss``."""
    try:
        return model(**inputs, **extra, return_dict=True)
    except TypeError as e:
        if "return_dict" not in str(e):
            raise
        return model(**inputs, **extra)
