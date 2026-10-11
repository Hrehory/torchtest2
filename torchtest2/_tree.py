"""Helpers for walking the (possibly nested) inputs and outputs of a model.

Models take a tensor, a tuple/list of tensors (positional arguments) or a dict of tensors
(keyword arguments), and return a tensor, a tuple, a dict or a dict-like output object
(e.g. Hugging Face ``ModelOutput``). These helpers treat all of them uniformly. Containers they
walk: mappings, namedtuples, lists, tuples and dataclass instances; any other object is an opaque
leaf without tensors (its tensor attributes are neither copied nor compared).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def tree_map(fn: Callable[[str, torch.Tensor], Any], obj: Any, path: str = "") -> Any:
    """Apply ``fn(path, tensor)`` to every tensor in ``obj``, keeping the container structure."""
    if isinstance(obj, torch.Tensor):
        return fn(path, obj)
    if isinstance(obj, Mapping):
        items = {k: tree_map(fn, v, f"{path}[{k!r}]") for k, v in obj.items()}
        mapping_type: Any = type(obj)
        try:
            return mapping_type(items)
        except Exception:  # exotic mapping types - a plain dict is good enough for model(**x)
            return items
    if _is_namedtuple(obj):
        return type(obj)(*(tree_map(fn, v, f"{path}.{f}") for f, v in zip(obj._fields, obj)))
    if isinstance(obj, (list, tuple)):
        return type(obj)(tree_map(fn, v, f"{path}[{i}]") for i, v in enumerate(obj))
    if _is_dataclass_instance(obj):
        # a shallow copy with the fields replaced: unlike dataclasses.replace this works for
        # frozen dataclasses, fields with init=False, __slots__ and custom constructors
        new = copy.copy(obj)
        for f in dataclasses.fields(obj):
            if hasattr(obj, f.name):
                value = tree_map(fn, getattr(obj, f.name), f"{path}.{f.name}")
                object.__setattr__(new, f.name, value)
        return new
    return obj


def _is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def tree_leaves(obj: Any, path: str = "") -> list[tuple[str, torch.Tensor]]:
    """Return ``(path, tensor)`` for every tensor in ``obj`` (depth first)."""
    if isinstance(obj, torch.Tensor):
        return [(path, obj)]
    if isinstance(obj, Mapping):
        return [leaf for k, v in obj.items() for leaf in tree_leaves(v, f"{path}[{k!r}]")]
    if _is_namedtuple(obj):
        return [leaf for f, v in zip(obj._fields, obj) for leaf in tree_leaves(v, f"{path}.{f}")]
    if isinstance(obj, (list, tuple)):
        return [leaf for i, v in enumerate(obj) for leaf in tree_leaves(v, f"{path}[{i}]")]
    if _is_dataclass_instance(obj):
        return [
            leaf
            for f in dataclasses.fields(obj)
            if hasattr(obj, f.name)
            for leaf in tree_leaves(getattr(obj, f.name), f"{path}.{f.name}")
        ]
    return []


def structure_difference(
    a: Any,
    b: Any,
    name: str = "output",
    *,
    shape_matters: Callable[[torch.Tensor], bool] | None = None,
) -> str | None:
    """Why the tensors of ``a`` and ``b`` cannot be compared leaf by leaf - different paths
    (a missing or extra output, a changed dict key or tuple length), shapes or dtypes - or
    ``None`` when they have the same structure. Leaves are identified by their path, so the
    insertion order of dict keys does not matter (list and tuple indices are part of the path).
    ``shape_matters(tensor_of_a)`` can exempt leaves whose shape may legitimately change."""
    leaves_a, leaves_b = dict(tree_leaves(a, name)), dict(tree_leaves(b, name))
    if leaves_a.keys() != leaves_b.keys():
        missing = [p for p in leaves_a if p not in leaves_b]
        extra = [p for p in leaves_b if p not in leaves_a]
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else ""))
        if extra:
            parts.append("new " + ", ".join(extra[:3]) + (" ..." if len(extra) > 3 else ""))
        return f"{len(leaves_a)} vs {len(leaves_b)} output tensors ({'; '.join(parts)})"
    for path, x in leaves_a.items():
        y = leaves_b[path]
        if x.shape != y.shape and (shape_matters is None or shape_matters(x)):
            return f"{path} changed shape from {tuple(x.shape)} to {tuple(y.shape)}"
        if x.dtype != y.dtype:
            return f"{path} changed dtype from {x.dtype} to {y.dtype}"
    return None


def paired_leaves(a: Any, b: Any, name: str = "output") -> list[tuple[str, Any, Any]]:
    """``(path, tensor_of_a, tensor_of_b)`` for every path present in both ``a`` and ``b``, in
    the order of ``a`` - pairs by path, not by position, so reordered dict keys still match."""
    leaves_b = dict(tree_leaves(b, name))
    return [(p, x, leaves_b[p]) for p, x in tree_leaves(a, name) if p in leaves_b]


def call_model(model: nn.Module, inputs: Any) -> Any:
    """Call ``model`` with ``inputs``: a dict is passed as keyword arguments, a tuple/list as
    positional arguments and anything else (a tensor, a namedtuple) as the single argument."""
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)) and not _is_namedtuple(inputs):
        return model(*inputs)
    return model(inputs)


def to_device(obj: Any, device: torch.device) -> Any:
    """Detached copy of every tensor in ``obj`` on ``device``. Copying guarantees that a model
    modifying its inputs in-place cannot change the caller's tensors."""
    return tree_map(lambda _, t: t.detach().to(device, copy=True), obj)


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter (or buffer) of ``model``; CPU for stateless models."""
    for tensor in model.parameters():
        return tensor.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def batch_size_of(obj: Any, batch_dim: int | None = 0) -> int | None:
    """Size of dimension ``batch_dim`` of the first tensor in ``obj`` that has it (``None`` when
    ``batch_dim`` is ``None``, i.e. the inputs have no batch dimension)."""
    if batch_dim is None:
        return None
    for _, tensor in tree_leaves(obj):
        if tensor.ndim > batch_dim:
            return tensor.shape[batch_dim]
    return None


def describe(obj: Any, name: str = "output") -> str:
    """Short human readable description of the tensors in ``obj``."""
    leaves = tree_leaves(obj, name)
    if len(leaves) == 1 and leaves[0][0] == name:
        return _describe_tensor(leaves[0][1])
    parts = [f"{path}: {_describe_tensor(t)}" for path, t in leaves[:4]]
    if len(leaves) > 4:
        parts.append(f"... ({len(leaves) - 4} more)")
    return ", ".join(parts)


def _describe_tensor(tensor: torch.Tensor) -> str:
    dtype = str(tensor.dtype).removeprefix("torch.")
    return f"Tensor[{', '.join(map(str, tensor.shape))}] {dtype}"


def extract_loss(outputs: Any) -> torch.Tensor | None:
    """Return ``outputs.loss`` / ``outputs["loss"]`` if the model computes its own loss."""
    if isinstance(outputs, Mapping):
        loss = outputs.get("loss")
    else:
        loss = getattr(outputs, "loss", None)
    return loss if isinstance(loss, torch.Tensor) else None
