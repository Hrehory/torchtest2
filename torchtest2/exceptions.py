"""Exceptions raised by torchtest2.

Every exception derives from :class:`TorchTestError`, which itself derives from
:class:`AssertionError`, so pytest reports them as test *failures* (not errors) and
``pytest.raises(AssertionError)`` catches all of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchtest2.report import Report


def _format_names(names: Sequence[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", ... ({len(names) - limit} more)"
    return shown


class TorchTestError(AssertionError):
    """Base class of every torchtest2 exception."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelCheckError(TorchTestError):
    """Raised by :meth:`Report.raise_on_failure` / :func:`torchtest2.assert_model_ok`.

    Attributes:
        report: the full :class:`~torchtest2.report.Report`.
    """

    def __init__(self, message: str, report: Report) -> None:
        self.report = report
        super().__init__(message)


class GpuUnusedException(TorchTestError):
    """Raised when CUDA is not available (or the model is not on a GPU)."""

    def __init__(self, message: str = "GPU is not available!") -> None:
        super().__init__(message)


class ParametersChangeException(TorchTestError):
    """Raised when parameters do (not) change during a training step.

    Arguments:
        param_names: names of the offending parameters.
        should_change: whether the parameters were expected to change.
    """

    def __init__(self, param_names: str | Sequence[str], should_change: bool) -> None:
        if isinstance(param_names, str):
            param_names = [param_names]
        self.param_names = list(param_names)
        self.should_change = should_change
        verb = "did not change" if should_change else "changed"
        expected = "expected to change" if should_change else "expected to stay fixed"
        super().__init__(
            f"{len(self.param_names)} parameter(s) {expected} {verb} after a training step: "
            f"{_format_names(self.param_names)}"
        )


class RangeException(TorchTestError):
    """Raised when tensor values fall outside of the expected range.

    Arguments:
        low, high: expected (inclusive) bounds; ``None`` means unbounded.
        actual_min, actual_max: the observed extremes of the tensor.
    """

    def __init__(
        self,
        low: float | None,
        high: float | None,
        actual_min: float,
        actual_max: float,
        message: str | None = None,
    ) -> None:
        self.low, self.high = low, high
        self.actual_min, self.actual_max = actual_min, actual_max
        lo = "-inf" if low is None else f"{low:g}"
        hi = "inf" if high is None else f"{high:g}"
        super().__init__(
            message
            or f"Tensor values are outside of the expected range [{lo}, {hi}]: "
            f"observed min = {actual_min:g}, max = {actual_max:g}."
        )


class NaNTensorException(TorchTestError):
    """Raised when a tensor contains NaN values."""

    def __init__(self, message: str = "There was a NaN value in tensor!") -> None:
        super().__init__(message)


class InfTensorException(TorchTestError):
    """Raised when a tensor contains +/-inf values."""

    def __init__(self, message: str = "There was an Infinity value in tensor!") -> None:
        super().__init__(message)


class InputDependencyException(TorchTestError):
    """Raised when (part of) an input has no influence on the model output."""


class BatchIndependencyException(TorchTestError):
    """Raised when the output for one sample depends on other samples in the batch.

    See https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html for background.
    """


class ConsistencyException(TorchTestError):
    """Raised when a shape does not match the expected one."""


class OverfitException(TorchTestError):
    """Raised by :meth:`ModelTester.test_overfit` when the expected loss is not reached."""

    def __init__(self, loss: float, expected_loss: float, relation: str = "greater") -> None:
        self.loss, self.expected_loss, self.relation = loss, expected_loss, relation
        comparison = "above" if relation == "greater" else "below"
        super().__init__(
            f"Overfitting a single batch ended with loss {loss:.4g}, {comparison} the expected "
            f"{expected_loss:.4g}."
        )


class NaNLossException(TorchTestError):
    """Raised when the loss is NaN (kept for compatibility with the 2021 development API)."""

    def __init__(self, message: str = "The loss is NaN!") -> None:
        super().__init__(message)


class OverfitNaNException(TorchTestError):
    """Raised by :meth:`ModelTester.test_overfit` when the loss becomes NaN/inf."""

    def __init__(self, message: str = "The loss became NaN/inf while overfitting a batch.") -> None:
        super().__init__(message)


class GradCheckException(TorchTestError):
    """Raised by :meth:`ModelTester.test_gradcheck` when analytical and numerical gradients
    disagree (typically a custom ``autograd.Function`` with a wrong ``backward``)."""


class AnomalyDetectionError(TorchTestError):
    """Raised by :meth:`ModelTester.test_anomaly` when ``torch.autograd.detect_anomaly`` finds
    NaN values in the backward pass; the original ``RuntimeError`` is the ``__cause__``."""


class PostTrainException(TorchTestError):
    """Base class of the behavioural (post-train) test failures."""

    def __init__(self, message: str, failing_indices: Sequence[int] = (), **details: Any) -> None:
        self.failing_indices = list(failing_indices)
        self.details = details
        super().__init__(message)


class InvarianceException(PostTrainException):
    """Raised by :func:`torchtest2.assert_invariance`."""


class DirectionalExpectationException(PostTrainException):
    """Raised by :func:`torchtest2.assert_directional`."""


class MinimumFunctionalityException(PostTrainException):
    """Raised by :func:`torchtest2.assert_minimum_functionality`."""
