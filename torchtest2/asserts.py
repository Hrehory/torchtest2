"""Small, composable tensor assertions. Each raises a :class:`~torchtest2.exceptions.TorchTestError`
subclass (an ``AssertionError``), so they can be used directly inside pytest tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from torchtest2.exceptions import (
    ConsistencyException,
    GpuUnusedException,
    InfTensorException,
    NaNTensorException,
    RangeException,
)


def assert_gpu_available() -> None:
    """Check that a CUDA GPU is available and accessible.

    Raises:
        GpuUnusedException: if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise GpuUnusedException()


def _tensor_label(tensor_name: str | None) -> str:
    return "Tensor" if tensor_name is None else f"Tensor {tensor_name!r}"


def assert_never_nan(tensor: torch.Tensor, tensor_name: str | None = None) -> None:
    """Raise if there are any NaN values in ``tensor``.

    Arguments:
        tensor: tensor to check.
        tensor_name: optional name used in the error message.

    Raises:
        NaNTensorException: if one or more NaN values occur in the tensor.
    """
    count = int(torch.isnan(tensor).sum())
    if count:
        raise NaNTensorException(f"{_tensor_label(tensor_name)} contains {count} NaN value(s)!")


def assert_never_inf(tensor: torch.Tensor, tensor_name: str | None = None) -> None:
    """Raise if there are any +/-inf values in ``tensor`` (NaN is *not* reported here).

    Arguments:
        tensor: tensor to check.
        tensor_name: optional name used in the error message.

    Raises:
        InfTensorException: if one or more infinite values occur in the tensor.
    """
    count = int(torch.isinf(tensor).sum())
    if count:
        raise InfTensorException(
            f"{_tensor_label(tensor_name)} contains {count} infinite value(s)!"
        )


def assert_all_finite(tensor: torch.Tensor, tensor_name: str | None = None) -> None:
    """Raise if ``tensor`` contains NaN or +/-inf values.

    Raises:
        NaNTensorException: if there is a NaN value (checked first).
        InfTensorException: if there is an infinite value.
    """
    assert_never_nan(tensor, tensor_name)
    assert_never_inf(tensor, tensor_name)


def assert_in_range(
    tensor: torch.Tensor,
    low: float | None = None,
    high: float | None = None,
    *,
    inclusive: bool = True,
) -> None:
    """Raise if any value of ``tensor`` lies outside of ``[low, high]``.

    Arguments:
        tensor: tensor to check (NaN values count as out of range).
        low: lower bound, ``None`` for no bound.
        high: upper bound, ``None`` for no bound.
        inclusive: whether the bounds themselves are allowed (``False`` means ``(low, high)``).

    Raises:
        RangeException: if some value is out of range.

    Example:
        >>> assert_in_range(torch.sigmoid(logits), 0, 1)
    """
    if low is None and high is None:
        raise ValueError("at least one of low/high must be given")
    values = tensor.detach()
    ok = torch.ones_like(values, dtype=torch.bool)
    if low is not None:
        ok &= values >= low if inclusive else values > low
    if high is not None:
        ok &= values <= high if inclusive else values < high
    if not bool(ok.all()):
        raise RangeException(low, high, float(values.min()), float(values.max()))


def assert_tensor_range(
    tensor: torch.Tensor,
    value: float,
    relation: Literal["greater", "less"],
    comparison_condition: Literal["all", "any"] = "all",
) -> None:
    """Raise unless all/any values of ``tensor`` are strictly greater/less than ``value``.

    Kept for compatibility with the 2021 development API; :func:`assert_in_range` is simpler for
    the common "all values within bounds" case.

    Arguments:
        tensor: tensor to check.
        value: the threshold.
        relation: ``"greater"`` or ``"less"``.
        comparison_condition: ``"all"`` (every value) or ``"any"`` (at least one value).

    Raises:
        RangeException: if the condition is not met.
    """
    if relation not in ("greater", "less"):
        raise ValueError("relation must be 'greater' or 'less'")
    if comparison_condition not in ("all", "any"):
        raise ValueError("comparison_condition must be 'all' or 'any'")
    values = tensor.detach()
    ok = values > value if relation == "greater" else values < value
    ok = ok.all() if comparison_condition == "all" else ok.any()
    if not bool(ok):
        raise RangeException(
            value if relation == "greater" else None,
            value if relation == "less" else None,
            float(values.min()),
            float(values.max()),
            message=f"{comparison_condition} elements of tensor must be {relation} than {value}!",
        )


def assert_shape(tensor: torch.Tensor, expected: Sequence[int | None]) -> None:
    """Raise if ``tensor.shape`` differs from ``expected``; ``None`` (or ``-1``) matches any size.

    Raises:
        ConsistencyException: on mismatch.

    Example:
        >>> assert_shape(model(x), (None, 10))  # any batch size, 10 classes
    """
    if not shape_matches(tensor.shape, expected):
        raise ConsistencyException(
            f"Expected shape {format_shape(expected)}, got {format_shape(tensor.shape)}."
        )


def shape_matches(actual: Sequence[int], expected: Sequence[int | None]) -> bool:
    return len(actual) == len(expected) and all(
        e is None or e == -1 or a == e for a, e in zip(actual, expected)
    )


def format_shape(shape: Sequence[int | None]) -> str:
    """``(None, 10)`` -> ``"(*, 10)"``."""
    parts = ["*" if s is None or s == -1 else str(s) for s in shape]
    return f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"
