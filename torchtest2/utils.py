"""Public utilities."""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn


def seed_everything(seed: int = 0, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy (if installed) and PyTorch (CPU and all GPUs).

    torchtest2 itself never calls this - checks run on a forked RNG and leave the global state
    untouched. Use it in your own tests when you need reproducible model initialisation.

    Arguments:
        seed: the seed.
        deterministic: additionally ask PyTorch for deterministic algorithms
            (``torch.use_deterministic_algorithms(True, warn_only=True)``, deterministic cuDNN).
    """
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of (trainable) scalar parameters of ``model``; shared parameters count once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def plugin_discovery(group: str | None = None) -> dict[str, Any]:
    """Plugin classes by name; see :func:`torchtest2.plugins.plugin_discovery`."""
    from torchtest2.plugins import plugin_discovery as discover

    return discover(group)
