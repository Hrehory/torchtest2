"""Result objects returned by :func:`torchtest2.check_model`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from torchtest2.exceptions import ModelCheckError


class Status(str, Enum):
    """Outcome of a single check."""

    PASS = "PASS"
    """The check ran and found nothing suspicious."""
    WARN = "WARN"
    """Something looks suspicious, but it can be legitimate - read the message."""
    FAIL = "FAIL"
    """The check found a problem that almost certainly is a bug."""
    SKIP = "SKIP"
    """The check did not apply (e.g. no loss function given) or was disabled."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CheckResult:
    """Result of one check.

    Attributes:
        name: identifier of the check, e.g. ``"gradient_flow"`` (see ``torchtest2.CHECKS``).
        status: :class:`Status` of the check.
        message: one-line, human readable explanation.
        details: machine readable extras, e.g. the names of offending parameters.
        seconds: wall-clock time the check took (excluding checks it depends on).
    """

    name: str
    status: Status
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def __str__(self) -> str:
        return f"{self.status.value:<4}  {self.name:<19} {self.message}"


@dataclass
class Report:
    """Collection of :class:`CheckResult` for one model.

    ``print(report)`` gives a readable table; :meth:`raise_on_failure` turns failures into an
    exception (handy inside ``pytest``); ``report["gradient_flow"]`` returns a single result.
    """

    model_name: str
    results: list[CheckResult]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """``True`` when no check failed (warnings are allowed)."""
        return not self.failures

    @property
    def failures(self) -> list[CheckResult]:
        return self._with_status(Status.FAIL)

    @property
    def warnings(self) -> list[CheckResult]:
        return self._with_status(Status.WARN)

    @property
    def passed(self) -> list[CheckResult]:
        return self._with_status(Status.PASS)

    @property
    def skipped(self) -> list[CheckResult]:
        return self._with_status(Status.SKIP)

    def _with_status(self, status: Status) -> list[CheckResult]:
        return [result for result in self.results if result.status is status]

    def __getitem__(self, name: str) -> CheckResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(f"no check named {name!r} in this report")

    def __contains__(self, name: object) -> bool:
        return any(result.name == name for result in self.results)

    def __iter__(self) -> Iterator[CheckResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def raise_on_failure(self, *, strict: bool = False) -> Report:
        """Raise :class:`~torchtest2.exceptions.ModelCheckError` if any check failed.

        Arguments:
            strict: also raise on warnings.

        Returns:
            the report itself, so that ``check_model(...).raise_on_failure()`` can be chained.
        """
        bad = self.failures + (self.warnings if strict else [])
        if bad:
            names = ", ".join(result.name for result in bad)
            raise ModelCheckError(f"{len(bad)} check(s) did not pass: {names}\n\n{self}", self)
        return self

    def summary(self) -> str:
        counts = [
            (len(self.passed), "passed"),
            (len(self.warnings), "warning" if len(self.warnings) == 1 else "warnings"),
            (len(self.failures), "failed"),
            (len(self.skipped), "skipped"),
        ]
        return ", ".join(f"{n} {label}" for n, label in counts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (e.g. for logging to an experiment tracker)."""
        return {
            "model": self.model_name,
            "ok": self.ok,
            "info": dict(self.info),
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": dict(r.details),
                    "seconds": r.seconds,
                }
                for r in self.results
            ],
        }

    def __str__(self) -> str:
        header = f"torchtest2 report for {self.model_name}"
        if self.info:
            header += "  (" + ", ".join(_format_info(k, v) for k, v in self.info.items()) + ")"
        lines = [header, *(f"  {result}" for result in self.results)]
        lines.append(f"  {self.summary()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Report {self.model_name}: {self.summary()}>"


def _format_info(key: str, value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{key}: {value:,}"
    return f"{key}: {value}"
