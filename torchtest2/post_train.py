"""Behavioural (post-train) tests in the spirit of CheckList (Ribeiro et al., 2020):

* :func:`assert_minimum_functionality` (MFT) - simple cases the model must get right,
* :func:`assert_invariance` (INV) - label-preserving perturbations must not change predictions,
* :func:`assert_directional` (DIR) - perturbations must move a score in a known direction.

All functions run the model in eval mode without gradients and restore its train/eval flags.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
import torch.nn as nn

from torchtest2._tree import (
    call_model,
    model_device,
    paired_leaves,
    structure_difference,
    to_device,
)
from torchtest2.exceptions import (
    DirectionalExpectationException,
    InvarianceException,
    MinimumFunctionalityException,
)

OutputFn = Callable[[Any], torch.Tensor]


@contextlib.contextmanager
def _inference(model: nn.Module) -> Iterator[None]:
    training = [m.training for m in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, flag in zip(model.modules(), training):
            module.training = flag


def _predict(model: nn.Module, inputs: Any, output_fn: OutputFn | None) -> Any:
    outputs = call_model(model, to_device(inputs, model_device(model)))
    return outputs if output_fn is None else output_fn(outputs)


def _argmax(outputs: Any) -> torch.Tensor:
    return outputs.argmax(dim=-1)


def assert_minimum_functionality(
    model: nn.Module,
    inputs: Any,
    expected: torch.Tensor,
    *,
    output_fn: OutputFn = _argmax,
    min_accuracy: float = 1.0,
) -> float:
    """MFT: the model must produce ``expected`` on a curated set of easy cases.

    Arguments:
        model: trained model.
        inputs: a batch of inputs (tensor, tuple or dict - see :func:`torchtest2.check_model`).
        expected: expected predictions, one per sample.
        output_fn: turns the model output into predictions; defaults to ``argmax(-1)``.
        min_accuracy: required fraction of correct predictions.

    Returns:
        the accuracy on ``inputs``.

    Raises:
        MinimumFunctionalityException: if the accuracy is below ``min_accuracy``.
    """
    with _inference(model):
        predictions = _predict(model, inputs, output_fn)
    expected = expected.to(predictions.device)
    correct = (predictions == expected).reshape(len(expected), -1).all(dim=1)
    accuracy = float(correct.float().mean())
    if accuracy < min_accuracy:
        failing = (~correct).nonzero().flatten().tolist()
        raise MinimumFunctionalityException(
            f"accuracy {accuracy:.1%} is below the required {min_accuracy:.1%}; "
            f"{len(failing)} failing sample(s), e.g. {failing[:10]}",
            failing,
            accuracy=accuracy,
        )
    return accuracy


def assert_invariance(
    model: nn.Module,
    inputs: Any,
    perturb: Callable[[Any], Any],
    *,
    output_fn: OutputFn | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """INV: ``perturb(inputs)`` must not change the (selected) outputs.

    Arguments:
        model: trained model.
        inputs: a batch of inputs.
        perturb: label-preserving transformation of the inputs, e.g. a horizontal flip, adding a
            neutral word, changing a name in a sentence.
        output_fn: selects what has to stay the same, e.g. ``lambda out: out.argmax(-1)`` to only
            require the same predicted class. Defaults to the raw outputs.
        rtol, atol: tolerances for floating point outputs (``torch.isclose``).

    Raises:
        InvarianceException: listing the samples whose outputs changed.
    """
    with _inference(model):
        base = _predict(model, inputs, output_fn)
        perturbed = _predict(model, perturb(inputs), output_fn)
    changed: torch.Tensor | None = None
    difference = structure_difference(base, perturbed)
    if difference:
        raise InvarianceException(f"the outputs changed structure: {difference}")
    for _, a, b in paired_leaves(base, perturbed):
        if a.is_floating_point():
            same = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = a == b
        per_sample = ~same.reshape(same.shape[0], -1).all(dim=1) if same.ndim else ~same[None]
        changed = per_sample if changed is None else changed | per_sample
    if changed is not None and bool(changed.any()):
        failing = changed.nonzero().flatten().tolist()
        raise InvarianceException(
            f"the output changed under the perturbation for {len(failing)}/{len(changed)} "
            f"sample(s), e.g. {failing[:10]}",
            failing,
        )


def assert_directional(
    model: nn.Module,
    inputs: Any,
    perturb: Callable[[Any], Any],
    *,
    output_fn: OutputFn,
    expect: Literal["increase", "decrease"],
    min_delta: float = 0.0,
) -> torch.Tensor:
    """DIR: ``perturb(inputs)`` must move a per-sample score in the expected direction.

    Arguments:
        model: trained model.
        inputs: a batch of inputs.
        perturb: transformation with a known effect, e.g. appending "I loved it" to a review.
        output_fn: maps the model output to one score per sample, e.g.
            ``lambda out: out.softmax(-1)[:, POSITIVE]``.
        expect: ``"increase"`` or ``"decrease"``.
        min_delta: the score has to move by more than this.

    Returns:
        the per-sample change of the score.

    Raises:
        DirectionalExpectationException: listing the samples that moved the wrong way.
    """
    if expect not in ("increase", "decrease"):
        raise ValueError("expect must be 'increase' or 'decrease'")
    with _inference(model):
        base = _predict(model, inputs, output_fn)
        perturbed = _predict(model, perturb(inputs), output_fn)
    delta = (perturbed - base).reshape(-1)
    ok = delta > min_delta if expect == "increase" else delta < -min_delta
    if not bool(ok.all()):
        failing = (~ok).nonzero().flatten().tolist()
        raise DirectionalExpectationException(
            f"the score was expected to {expect} (by more than {min_delta:g}) but did not for "
            f"{len(failing)}/{len(ok)} sample(s), e.g. {failing[:10]}",
            failing,
            delta=delta.tolist(),
        )
    return delta
