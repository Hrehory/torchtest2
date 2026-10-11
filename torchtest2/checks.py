"""One-call health check for PyTorch models: :func:`check_model`, and the checks it runs.

Every check runs on a snapshot of the model and optimizer, which is restored afterwards, on a
forked RNG, and on copies of the inputs; :mod:`torchtest2._session` implements this execution
model and states what is (and is not) restored.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import inspect
import math
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtest2._session import (  # noqa: F401 - re-exported for torchtest2.model_tester
    DEFAULT_BATCH_DIM,
    EvalForwardFn,
    ForwardFn,
    PluginSpec,
    _Context,
    _cuda_devices,
    _DefaultBatchDim,
    _fork_python_rngs,
    _Hooks,
    _matches_prefix,
    _materialize,
    _Options,
    _reseed,
    _resolve_hooks,
    _session,
    _Snapshot,
)
from torchtest2._tree import (
    call_model,
    describe,
    extract_loss,
    model_device,
    paired_leaves,
    structure_difference,
    tree_leaves,
    tree_map,
)
from torchtest2.asserts import format_shape, shape_matches
from torchtest2.exceptions import _format_names
from torchtest2.plugin_base import LossFn
from torchtest2.report import CheckResult, Report, Status

CHECKS: tuple[str, ...] = (
    "forward",
    "output_shape",
    "output_finite",
    "output_range",
    "output_varies",
    "eval_determinism",
    "loss",
    "initial_loss",
    "optimizer_coverage",
    "gradient_flow",
    "gradient_magnitude",
    "params_change",
    "frozen_params",
    "input_dependency",
    "batch_independence",
    "overfit_batch",
    "gradcheck",
    "gpu",
)
"""Names of all checks, in the order :func:`check_model` runs them."""

# tolerances used when comparing outputs of two forward passes
_RTOL = 1e-4
_ATOL = 1e-5


class _Skip(Exception):
    """Raised by a check that does not apply; the message becomes the SKIP reason."""


# --------------------------------------------------------------------------------------------
# shared building blocks


def _compute_loss(ctx: _Context, outputs: Any) -> Any:
    if ctx.loss_fn is not None:
        if ctx.targets is None:
            return ctx.loss_fn(outputs)
        return ctx.loss_fn(outputs, ctx.targets)
    return extract_loss(outputs)


def _surrogate(outputs: Any, seed: int) -> torch.Tensor | None:
    """Random projection of all differentiable outputs to a scalar.

    Unlike ``outputs.sum()`` it cannot be constant (``softmax(x).sum() == 1``), so every output
    element that depends on a parameter/input yields a non-zero gradient almost surely.
    """
    generator = torch.Generator().manual_seed(seed)
    total = None
    for _, tensor in tree_leaves(outputs):
        if tensor.is_complex() and tensor.requires_grad:
            tensor = torch.view_as_real(tensor)  # real and imaginary parts
        if tensor.is_floating_point() and tensor.requires_grad:
            weights = torch.randn(tensor.shape, generator=generator)
            term = (tensor * weights.to(tensor.device, tensor.dtype)).sum()
            total = term if total is None else total + term
    return total


def _objective(ctx: _Context, outputs: Any) -> torch.Tensor | None:
    """The real loss if available, otherwise the surrogate."""
    if ctx.loss_source is not None and ctx.loss_error is None:
        return _compute_loss(ctx, outputs)
    return _surrogate(outputs, ctx.options.seed)


def _train_forward(ctx: _Context) -> Any:
    """Forward pass in train mode (``forward_fn`` if given)."""
    ctx.model.train()
    if ctx.forward_fn is not None:
        return ctx.forward_fn(ctx.model, ctx.inputs, ctx.targets)
    return call_model(ctx.model, ctx.inputs)


def _train_objective(ctx: _Context) -> torch.Tensor | None:
    """Training-mode forward pass and objective; a failure is remembered so that the other
    training checks are skipped instead of reporting the same error again."""
    _require_train(ctx)
    try:
        outputs = _train_forward(ctx)
    except Exception as e:
        ctx.train_error = f"{type(e).__name__}: {e}"
        raise
    return _objective(ctx, outputs)


def _require_train(ctx: _Context) -> None:
    if ctx.train_error is not None:
        raise _Skip(f"the training-mode forward pass failed ({ctx.train_error[:120]})")


def needs_closure(optimizer: torch.optim.Optimizer) -> bool:
    """Whether ``optimizer.step`` requires a closure (e.g. LBFGS)."""
    try:
        closure = inspect.signature(optimizer.step).parameters.get("closure")
    except (TypeError, ValueError):
        return False
    return closure is not None and closure.default is inspect.Parameter.empty


def _dense(grad: torch.Tensor) -> torch.Tensor:
    """The stored values of a (possibly sparse) gradient."""
    return grad.coalesce().values() if grad.is_sparse else grad


def _gradients(ctx: _Context) -> dict[str, torch.Tensor | None]:
    """Gradients of the objective after one forward/backward pass in train mode (cached)."""
    if ctx.grads is None:
        ctx.reset()
        ctx.model.zero_grad(set_to_none=True)
        loss = _train_objective(ctx)
        if loss is not None and loss.requires_grad:
            loss.backward()
        ctx.grads = {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in ctx.named_params()
        }
    return ctx.grads


def _gradient_under_other_seeds(ctx: _Context, names: Sequence[str], tries: int = 3) -> set[str]:
    """Which of ``names`` receive a gradient in a training forward pass with another random
    seed: stochastic depth and LayerDrop skip whole layers in some passes but not in others."""
    found: set[str] = set()
    params = dict(ctx.named_params())
    for k in range(1, tries + 1):
        ctx.reset()
        _reseed(ctx.options.seed + k, ctx.cuda_devices)
        ctx.model.zero_grad(set_to_none=True)
        loss = _train_objective(ctx)
        if loss is not None and loss.requires_grad:
            loss.backward()
        found |= {n for n in names if params[n].grad is not None}
        if len(found) == len(names):
            break
    ctx.model.zero_grad(set_to_none=True)
    ctx.reset()
    return found


def _zero_after_step(ctx: _Context) -> set[str]:
    """Parameters whose gradient is still exactly zero after one optimizer step (cached).

    A zero-initialised layer (a classification head, the last normalisation of a residual branch)
    blocks the gradient of everything before it, but only until the first update; a parameter
    that is really dead (dead ReLUs, multiplied by a constant zero) stays at zero.
    """
    if ctx.still_zero is None:
        grads = _gradients(ctx)
        zero = {n for n, g in grads.items() if g is not None and not bool(_dense(g).any())}
        if not zero or ctx.optimizer is None:
            ctx.still_zero = zero
        else:
            _one_step(ctx)
            ctx.model.zero_grad(set_to_none=True)
            loss = _train_objective(ctx)
            if loss is not None and loss.requires_grad:
                loss.backward()
            params = dict(ctx.named_params())
            ctx.still_zero = {
                n for n in zero if (g := params[n].grad) is None or not bool(_dense(g).any())
            }
            ctx.model.zero_grad(set_to_none=True)
    return ctx.still_zero


def _changed_after_step(ctx: _Context) -> dict[str, bool]:
    """Which parameters changed after one optimizer step (cached)."""
    if ctx.changed is None:
        if ctx.optimizer is None:
            raise _Skip("the model has no trainable parameters")
        _one_step(ctx)
        before = ctx.snapshot.params
        ctx.changed = {
            name: not torch.equal(p.detach().cpu(), before[name]) for name, p in ctx.named_params()
        }
    return ctx.changed


def _require_forward(ctx: _Context) -> None:
    if ctx.forward_error is not None:
        raise _Skip("the forward pass failed")


def _require_loss(ctx: _Context) -> None:
    _require_forward(ctx)
    if ctx.loss_source is None:
        raise _Skip("no loss_fn given and the model output has no .loss")
    if ctx.loss_error is not None:
        raise _Skip("the loss could not be computed")


def _require_batch(ctx: _Context) -> int:
    if ctx.batch_dim is None:
        raise _Skip("the inputs have no batch dimension (batch_dim=None)")
    if ctx.batch_size is None or ctx.batch_size < 2:
        raise _Skip("needs a batch with at least 2 samples")
    return ctx.batch_size


def _names(names: Sequence[str]) -> str:
    return _format_names(list(names))


def _float_leaves(obj: Any) -> list[tuple[str, torch.Tensor]]:
    return [(p, t) for p, t in tree_leaves(obj, "output") if t.is_floating_point()]


def _half_precision(ctx: _Context) -> bool:
    return any(p.dtype in (torch.float16, torch.bfloat16) for p in ctx.model.parameters())


_HALF_HINT = (
    " - the parameters are in half precision: pure float16/bfloat16 training is often unstable "
    "(e.g. Adam's eps=1e-8 underflows to 0 in float16); train in float32 with torch.autocast"
)


_OUT_HINT = "; pass output_batch_dim= if it differs from that of the inputs"


def _batched(
    leaves: list[tuple[str, torch.Tensor]], batch_size: int | None, batch_dim: int | None = 0
) -> list[tuple[str, torch.Tensor]]:
    """The leaves that have the batch dimension, moved to the front (as views)."""
    if batch_dim is None or batch_size is None:
        return []
    return [
        (p, t.movedim(batch_dim, 0))
        for p, t in leaves
        if t.ndim > batch_dim and t.shape[batch_dim] == batch_size
    ]


def _eval_forward(
    ctx: _Context,
    inputs: Any | None = None,
    *,
    reseed: bool = True,
    grad: bool = False,
    model: nn.Module | None = None,
) -> Any:
    """Forward pass in eval mode (``eval_forward_fn`` if given). Reseeding first makes passes
    that are compared with each other see the same random numbers, so randomness in eval mode
    (reported by ``eval_determinism``) does not cause false alarms in other checks."""
    if reseed:
        _reseed(ctx.options.seed, ctx.cuda_devices)
    model = ctx.model if model is None else model
    inputs = ctx.inputs if inputs is None else inputs
    model.eval()
    with torch.set_grad_enabled(grad):
        if ctx.eval_forward_fn is not None:
            return ctx.eval_forward_fn(model, inputs)
        return call_model(model, inputs)


def _close(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise closeness (exact for non-float tensors)."""
    if a.is_floating_point() or a.is_complex():
        return torch.isclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    return a == b


# --------------------------------------------------------------------------------------------
# the checks


def _check_forward(ctx: _Context) -> CheckResult:
    try:
        outputs = _eval_forward(ctx)
    except Exception as e:  # any error in user code is a finding
        ctx.forward_error = f"{type(e).__name__}: {e}"
        return CheckResult("forward", Status.FAIL, f"model(inputs) raised {ctx.forward_error}")
    if not tree_leaves(outputs):
        ctx.forward_error = "no tensors"
        return CheckResult(
            "forward", Status.FAIL, f"the model output contains no tensors ({type(outputs)})"
        )
    ctx.eval_outputs = outputs
    if ctx.loss_fn is not None:
        ctx.loss_source = ctx.options.loss_label
    elif extract_loss(outputs) is not None:
        ctx.loss_source = "output.loss"
    return CheckResult("forward", Status.PASS, f"output: {describe(outputs)}")


def _check_output_shape(ctx: _Context) -> CheckResult:
    expected = ctx.options.output_shape
    if expected is None:
        raise _Skip("no output_shape given")
    _require_forward(ctx)
    path, tensor = tree_leaves(ctx.eval_outputs, "output")[0]
    shown = format_shape(expected)
    if not shape_matches(tensor.shape, expected):
        return CheckResult(
            "output_shape",
            Status.FAIL,
            f"{path} has shape {format_shape(tensor.shape)}, expected {shown}",
            {"actual": tuple(tensor.shape)},
        )
    return CheckResult("output_shape", Status.PASS, f"{path} has the expected shape {shown}")


def _check_output_finite(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    problems = []
    numeric = [
        (p, t)
        for p, t in tree_leaves(ctx.eval_outputs, "output")
        if t.is_floating_point() or t.is_complex()
    ]
    for path, tensor in numeric:
        n_nan, n_inf = int(torch.isnan(tensor).sum()), int(torch.isinf(tensor).sum())
        if n_nan or n_inf:
            problems.append(f"{path}: {n_nan} NaN, {n_inf} inf of {tensor.numel()}")
    if problems:
        return CheckResult(
            "output_finite",
            Status.FAIL,
            "non-finite outputs - " + "; ".join(problems),
            {"problems": problems},
        )
    return CheckResult("output_finite", Status.PASS, "no NaN/inf in the outputs")


def _check_output_range(ctx: _Context) -> CheckResult:
    if ctx.options.output_range is None:
        raise _Skip("no output_range given")
    _require_forward(ctx)
    low, high = ctx.options.output_range
    bad = []
    for path, tensor in _float_leaves(ctx.eval_outputs):
        tensor = tensor.detach()
        lo, hi = float(tensor.min()), float(tensor.max())
        if (low is not None and lo < low) or (high is not None and hi > high):
            bad.append(f"{path} spans [{lo:.4g}, {hi:.4g}]")
    shown = f"[{'-inf' if low is None else low}, {'inf' if high is None else high}]"
    if bad:
        return CheckResult("output_range", Status.FAIL, f"outputs leave {shown}: " + "; ".join(bad))
    return CheckResult("output_range", Status.PASS, f"all outputs within {shown}")


def _check_output_varies(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    batch_size = _require_batch(ctx)
    batched_inputs = _batched(tree_leaves(ctx.inputs), batch_size, ctx.batch_dim)
    if all(torch.equal(t, t[:1].expand_as(t)) for _, t in batched_inputs):
        raise _Skip("all samples in the batch are identical")
    out_bd = ctx.output_batch_dim
    outputs = _batched(_float_leaves(ctx.eval_outputs), batch_size, out_bd)
    if not outputs:
        raise _Skip(f"no floating point output has the batch dimension (dim {out_bd}){_OUT_HINT}")

    # (almost) exact equality: outputs of an untrained network can legitimately be very close
    def constant(leaves):
        return all(torch.allclose(t, t[:1].expand_as(t), rtol=1e-6, atol=1e-8) for _, t in leaves)

    if constant(outputs) and ctx.optimizer is not None:
        # a zero-initialised output layer makes every output equal at initialisation
        _one_step(ctx)
        later = _batched(_float_leaves(_eval_forward(ctx)), batch_size, out_bd)
        if not constant(later):
            return CheckResult(
                "output_varies",
                Status.PASS,
                "outputs differ between samples after one optimizer step (all equal at "
                "initialisation: zero-initialised output layer?)",
            )
    if constant(outputs):
        return CheckResult(
            "output_varies",
            Status.WARN,
            f"all {batch_size} samples produce the same output although the inputs differ "
            "(input ignored, collapsed activations or a broadcasting bug?)",
        )
    return CheckResult("output_varies", Status.PASS, "outputs differ between samples")


def _check_eval_determinism(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    # two consecutive passes: the RNG advances in between, as it would at inference time
    first_outputs = _eval_forward(ctx)
    second_outputs = _eval_forward(ctx, reseed=False)
    changed = structure_difference(first_outputs, second_outputs)
    if changed:
        return CheckResult(
            "eval_determinism",
            Status.WARN,
            f"two forward passes in eval mode return differently structured outputs: {changed}",
        )
    pairs = paired_leaves(first_outputs, second_outputs)
    differ = [p for p, a, b in pairs if not bool(_close(a, b).all())]
    if differ:
        return CheckResult(
            "eval_determinism",
            Status.WARN,
            f"two forward passes in eval mode differ ({_names(differ)}); randomness in eval mode "
            "- e.g. F.dropout(x) without training=self.training, or torch.rand in forward()",
        )
    return CheckResult("eval_determinism", Status.PASS, "eval-mode forward passes are reproducible")


def _check_loss(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    if ctx.loss_source is None:
        raise _Skip(
            "no loss_fn given and the model output has no .loss; gradient checks use a "
            "random projection of the outputs instead"
        )

    def fail(message: str) -> CheckResult:
        ctx.loss_error = message
        return CheckResult("loss", Status.FAIL, message)

    try:
        outputs = _train_forward(ctx)
    except Exception as e:
        ctx.train_error = f"{type(e).__name__}: {e}"
        return fail(f"the training-mode forward pass raised {ctx.train_error}")
    try:
        loss = _compute_loss(ctx, outputs)
    except Exception as e:
        return fail(f"computing the loss raised {type(e).__name__}: {e}")
    if not isinstance(loss, torch.Tensor):
        return fail(f"the loss must be a torch.Tensor, got {type(loss).__name__}")
    if loss.numel() != 1:
        return fail(
            f"the loss must be a scalar, got shape {tuple(loss.shape)} "
            "(reduction='none' without .mean()?)"
        )
    value = loss.item()
    if not math.isfinite(value):
        return fail(f"the loss is {value}")
    if not loss.requires_grad:
        return fail(
            f"the loss ({value:.4g}) is not connected to the autograd graph - .detach(), .item(), "
            "a NumPy round-trip or torch.no_grad() somewhere in forward()/loss_fn?"
        )
    if loss.grad_fn is None and not any(loss is p for p in ctx.model.parameters()):
        return fail(
            f"the loss ({value:.4g}) is a leaf tensor: it was re-created, e.g. with "
            "torch.tensor(loss.item(), requires_grad=True), so no gradient reaches the model"
        )
    return CheckResult(
        "loss", Status.PASS, f"loss = {value:.4g} ({ctx.loss_source})", {"value": value}
    )


def _check_initial_loss(ctx: _Context) -> CheckResult:
    _require_loss(ctx)
    loss_fn = ctx.loss_fn
    if not (loss_fn is F.cross_entropy or isinstance(loss_fn, nn.CrossEntropyLoss)):
        raise _Skip("only implemented for cross-entropy")
    # the loss training starts from: training mode (batch statistics in BatchNorm - an
    # untrained network in eval mode uses running statistics that do not match its activations)
    _require_train(ctx)
    with torch.no_grad():
        logits = _train_forward(ctx)
        if not isinstance(logits, torch.Tensor) or logits.ndim < 2 or logits.shape[1] < 2:
            raise _Skip("expects logits of shape (batch, classes, ...)")
        value = float(_compute_loss(ctx, logits))
    n_classes = logits.shape[1]
    expected = math.log(n_classes)
    details = {"value": value, "expected": expected}
    if value > 2 * expected:
        return CheckResult(
            "initial_loss",
            Status.WARN,
            f"initial loss {value:.3f} is much larger than ln({n_classes}) = {expected:.3f} "
            "expected from an untrained classifier; the network is over-confident at init "
            "(scale down the last layer's init, check input normalisation)",
            details,
        )
    if value < expected / 2:
        return CheckResult(
            "initial_loss",
            Status.WARN,
            f"initial loss {value:.3f} is suspiciously low compared to ln({n_classes}) = "
            f"{expected:.3f}; do the targets leak into the inputs?",
            details,
        )
    return CheckResult(
        "initial_loss",
        Status.PASS,
        f"initial loss {value:.3f} is close to ln({n_classes}) = {expected:.3f}",
        details,
    )


def _check_optimizer_coverage(ctx: _Context) -> CheckResult:
    if not ctx.user_optimizer:
        raise _Skip("no optimizer given (a default Adam(lr=1e-3) is used)")
    assert ctx.optimizer is not None
    in_optimizer = {id(p) for group in ctx.optimizer.param_groups for p in group["params"]}
    model_params = {id(p) for p in ctx.model.parameters()}
    missing = [
        n for n, p in ctx.named_params() if not ctx.is_frozen(n, p) and id(p) not in in_optimizer
    ]
    foreign = len(in_optimizer - model_params)
    if missing:
        hint = (
            f" The optimizer holds {foreign} tensor(s) that are not (or no longer) parameters of "
            "this model - were layers replaced after creating it, or was it created for another "
            "model instance?"
            if foreign
            else " Was the optimizer created before these layers were added/replaced?"
        )
        return CheckResult(
            "optimizer_coverage",
            Status.FAIL,
            f"{len(missing)} trainable parameter(s) are not in the optimizer: "
            f"{_names(missing)}.{hint}",
            {"missing": missing, "foreign": foreign},
        )
    if foreign:
        return CheckResult(
            "optimizer_coverage",
            Status.WARN,
            f"the optimizer holds {foreign} tensor(s) that are not parameters of this model",
            {"foreign": foreign},
        )
    return CheckResult(
        "optimizer_coverage", Status.PASS, "the optimizer covers every trainable parameter"
    )


def _check_gradient_flow(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    _require_train(ctx)
    trainable = ctx.trainable_names()
    if not trainable:
        raise _Skip("the model has no trainable parameters")
    grads = _gradients(ctx)
    none = [n for n in trainable if grads[n] is None]
    zero = [n for n in trainable if (g := grads[n]) is not None and not bool(_dense(g).any())]
    objective = ctx.objective_name()
    stochastic = sorted(_gradient_under_other_seeds(ctx, none)) if none else []
    none = [n for n in none if n not in stochastic]
    stochastic_note = (
        f"; {len(stochastic)} other(s) get a gradient only in some passes (stochastic depth / "
        f"LayerDrop skips their layer at random): {_names(stochastic)}"
        if stochastic
        else ""
    )
    if none:
        return CheckResult(
            "gradient_flow",
            Status.FAIL,
            f"{len(none)}/{len(trainable)} trainable parameter(s) get no gradient from the "
            f"{objective} (unused in forward, or the graph is cut by .detach()/.item()/"
            f"no_grad): {_names(none)}{stochastic_note}",
            {"no_grad": none, "zero_grad": zero, "stochastic": stochastic},
        )
    still = _zero_after_step(ctx) if zero else set()
    dead = [n for n in zero if n in still]
    transient = [n for n in zero if n not in still]
    details = {"zero_grad": dead, "zero_at_init": transient}
    if dead:
        note = f" ({len(transient)} more only at initialisation)" if transient else ""
        return CheckResult(
            "gradient_flow",
            Status.WARN,
            f"{len(dead)}/{len(trainable)} trainable parameter(s) get an all-zero gradient, also "
            f"after an optimizer step (dead ReLUs, multiplied by zero, masked out?): "
            f"{_names(dead)}{note}{stochastic_note}",
            details,
        )
    if transient:
        return CheckResult(
            "gradient_flow",
            Status.PASS,
            f"all {len(trainable)} trainable parameter tensors get non-zero gradients; "
            f"{len(transient)} only after one optimizer step (behind a zero-initialised layer): "
            f"{_names(transient)}{stochastic_note}",
            details,
        )
    return CheckResult(
        "gradient_flow",
        Status.PASS,
        f"all {len(trainable)} trainable parameter tensors get non-zero gradients ({objective})"
        + stochastic_note,
        {"stochastic": stochastic} if stochastic else {},
    )


def _check_gradient_magnitude(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    _require_train(ctx)
    grads = {n: g for n, g in _gradients(ctx).items() if g is not None}
    if not grads:
        raise _Skip("no parameter received a gradient")
    non_finite = [n for n, g in grads.items() if not bool(torch.isfinite(_dense(g)).all())]
    if non_finite:
        return CheckResult(
            "gradient_magnitude",
            Status.FAIL,
            f"NaN/inf gradients in {len(non_finite)} parameter(s): {_names(non_finite)} "
            "(log/sqrt/division of zero or negative values, overflow in half precision?)"
            + (_HALF_HINT if _half_precision(ctx) else ""),
            {"non_finite": non_finite},
        )
    norms = {n: float(_dense(g).float().norm()) for n, g in grads.items()}
    total = math.sqrt(sum(v * v for v in norms.values()))
    threshold = ctx.options.grad_norm_threshold
    params = dict(ctx.named_params())
    # relative to the parameter's own norm (at least 1): a gradient norm means little without
    # the scale of the parameter it updates
    relative = {n: v / max(float(params[n].detach().float().norm()), 1.0) for n, v in norms.items()}
    large = sorted(
        (n for n, v in relative.items() if v > threshold), key=lambda n: relative[n], reverse=True
    )
    details = {"total_norm": total, "norms": norms, "relative": relative}
    if large:
        worst = large[0]
        return CheckResult(
            "gradient_magnitude",
            Status.WARN,
            f"exploding gradients: the gradient norm of {worst} is {norms[worst]:.3g}, "
            f"{relative[worst]:.3g}x its parameter norm (> {threshold:g}); {len(large)} "
            "parameter(s) above the threshold. Check init scale, normalisation, learning rate or "
            "add gradient clipping",
            details,
        )
    return CheckResult(
        "gradient_magnitude", Status.PASS, f"total gradient norm {total:.3g}", details
    )


def _check_params_change(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    _require_train(ctx)
    trainable = ctx.trainable_names()
    if not trainable:
        raise _Skip("the model has no trainable parameters")
    changed = _changed_after_step(ctx)
    unchanged = [n for n in trainable if not changed[n]]
    if not unchanged:
        return CheckResult(
            "params_change",
            Status.PASS,
            f"all {len(trainable)} trainable parameter tensors changed after one optimizer step",
        )
    # a parameter whose gradient is exactly zero cannot change; that is gradient_flow's (WARN)
    # finding, not an optimizer problem
    grads = _gradients(ctx)
    zero_grad = [
        n for n in unchanged if (g := grads.get(n)) is not None and not bool(_dense(g).any())
    ]
    params = dict(ctx.named_params())

    def vanishing(name: str) -> bool:
        # far too small to move the parameter in its precision (e.g. behind a layer scale
        # initialised to 1e-6); gradient_flow and gradient_magnitude own this finding
        grad = grads.get(name)
        if grad is None:
            return False
        largest = float(params[name].detach().abs().max()) if params[name].numel() else 0.0
        return float(_dense(grad).abs().max()) < 1e-4 * max(largest, 1e-30)

    tiny = [n for n in unchanged if n not in zero_grad and vanishing(n)]
    real = [n for n in unchanged if n not in zero_grad and n not in tiny]
    # skipped by stochastic depth / LayerDrop in this pass: they change in other steps
    no_grad = [n for n in real if grads.get(n) is None]
    skipped = sorted(_gradient_under_other_seeds(ctx, no_grad)) if no_grad else []
    real = [n for n in real if n not in skipped]
    # behind a zero-initialised layer: the gradient is non-zero from the second step on
    still = _zero_after_step(ctx) if zero_grad else set()
    at_init = [n for n in zero_grad if n not in still]
    zero_grad = [n for n in zero_grad if n in still]
    details = {
        "unchanged": unchanged,
        "zero_gradient": zero_grad,
        "zero_gradient_at_init": at_init,
        "vanishing_gradient": tiny,
        "skipped_layer": skipped,
    }
    if real:
        others = len(zero_grad) + len(tiny) + len(at_init) + len(skipped)
        note = f" ({others} more with a zero or vanishing gradient)" if others else ""
        return CheckResult(
            "params_change",
            Status.FAIL,
            f"{len(real)}/{len(trainable)} trainable parameter(s) did not change after an "
            f"optimizer step: {_names(real)}{note}",
            details,
        )
    if not zero_grad and not tiny:
        return CheckResult(
            "params_change",
            Status.PASS,
            f"{len(trainable) - len(at_init) - len(skipped)}/{len(trainable)} trainable "
            f"parameter tensors changed after one optimizer step; {len(at_init)} are behind a "
            f"zero-initialised layer (a gradient from the second step on), {len(skipped)} in a "
            "layer that stochastic depth / LayerDrop skipped in this pass",
            details,
        )
    parts = []
    if zero_grad:
        parts.append(f"{len(zero_grad)} with an exactly zero gradient: {_names(zero_grad)}")
    if tiny:
        parts.append(
            f"{len(tiny)} with a gradient too small to change them (vanishing): {_names(tiny)}"
        )
    return CheckResult(
        "params_change",
        Status.WARN,
        f"{len(zero_grad) + len(tiny)}/{len(trainable)} trainable parameter(s) did not change - "
        + "; ".join(parts),
        details,
    )


def _check_frozen_params(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    _require_train(ctx)
    frozen = [n for n, p in ctx.named_params() if ctx.is_frozen(n, p)]
    all_names = [n for n, _ in ctx.model.named_parameters(remove_duplicate=False)]
    unmatched = [
        f for f in ctx.options.frozen if not any(_matches_prefix(n, (f,)) for n in all_names)
    ]
    unmatched_note = (
        f"frozen={unmatched} match(es) no parameter name - a typo, or a renamed module?"
        if unmatched
        else ""
    )
    if not frozen:
        if unmatched:
            return CheckResult(
                "frozen_params", Status.WARN, unmatched_note, {"unmatched": unmatched}
            )
        raise _Skip("no frozen parameters (requires_grad=False or frozen=[...])")
    if ctx.optimizer is None:
        return CheckResult("frozen_params", Status.PASS, "every parameter is frozen")
    changed = _changed_after_step(ctx)
    moved = [n for n in frozen if changed[n]]
    if moved:
        return CheckResult(
            "frozen_params",
            Status.FAIL,
            f"{len(moved)}/{len(frozen)} parameter(s) expected to be frozen changed after an "
            f"optimizer step: {_names(moved)} (requires_grad still True? still in the "
            "optimizer? modified in-place in forward()?)"
            + (f"; {unmatched_note}" if unmatched else ""),
            {"changed": moved, "unmatched": unmatched},
        )
    if unmatched:
        return CheckResult("frozen_params", Status.WARN, unmatched_note, {"unmatched": unmatched})
    return CheckResult(
        "frozen_params", Status.PASS, f"all {len(frozen)} frozen parameter tensors stayed fixed"
    )


def _check_input_dependency(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    leaves = tree_leaves(ctx.inputs, "inputs")
    float_paths = {p for p, t in leaves if t.is_floating_point()}
    problems: list[str] = []
    notes: list[str] = []
    checked = 0
    originals = dict(leaves)

    warnings_: list[str] = []
    if float_paths:
        paths = [p for p, _ in leaves if p in float_paths]
        bd = ctx.batch_dim
        # accumulated |gradient| per input: a feature influences the model if any probe sees it
        influence = {
            p: _absolute(g, originals[p])
            for p, g in _input_gradients(ctx, paths, train=False).items()
        }

        def dead_mask(path: str) -> torch.Tensor:
            grad = influence[path]
            batched = (
                bd is not None
                and grad.ndim >= 2
                and grad.ndim > bd
                and grad.shape[bd] == ctx.batch_size
            )
            per_feature = grad.sum(bd) if batched else grad
            value = originals[path]
            # features that are 0 in every sample are indistinguishable from padding
            padding = (value == 0).all(bd) if batched else value == 0
            return (per_feature == 0) & ~padding.to(per_feature.device)

        def unresolved() -> list[str]:
            return [p for p in paths if bool(dead_mask(p).any()) or not bool(influence[p].any())]

        def through_training_objective(todo: list[str]) -> dict[str, torch.Tensor | None]:
            # eval outputs can be piecewise constant (e.g. detections after NMS): the training
            # objective is what the inputs must influence during training
            return _input_gradients(ctx, todo, train=True)

        def after_one_step(todo: list[str]) -> dict[str, torch.Tensor | None]:
            # zero-initialised output layers or residual branches (torchvision ViT, timm) hide
            # inputs at initialisation; one optimizer step makes them non-zero
            _one_step(ctx)
            return _input_gradients(ctx, todo, train=False)

        fallbacks = (
            ("the training objective", through_training_objective),
            ("the outputs after one optimizer step (zero-initialised layers?)", after_one_step),
        )
        for label, probe in fallbacks:
            todo = unresolved()
            if not todo or (label.startswith("the outputs after") and ctx.optimizer is None):
                continue
            before = {p: int(dead_mask(p).sum()) for p in todo}
            for p, g in probe(todo).items():
                influence[p] = influence[p] + _absolute(g, originals[p])
            helped = [p for p in todo if int(dead_mask(p).sum()) < before[p]]
            if helped:
                notes.append(f"{', '.join(helped)}: some influence only through {label}")

        for path in paths:
            checked += 1
            if not bool(influence[path].any()):
                problems.append(f"{path} has no influence on the output")
                continue
            dead = dead_mask(path)
            n_dead = int(dead.sum())
            if n_dead:
                example = [tuple(i) for i in dead.nonzero()[:3].tolist()]
                message = (
                    f"{n_dead}/{dead.numel()} features of {path} "
                    f"(per-sample shape {tuple(dead.shape)}) have no influence on the "
                    f"output, e.g. at {example}"
                )
                if _only_trailing_border(dead):
                    warnings_.append(
                        message + " - only at the trailing border: kernel size/stride do not "
                        "cover the input size"
                    )
                else:
                    problems.append(message)

    # integer/bool inputs (token ids, masks): roll them along the batch - values stay valid
    bd = ctx.batch_dim
    for path, tensor in leaves:
        if bd is None or path in float_paths:
            continue
        if tensor.ndim <= bd or tensor.shape[bd] != ctx.batch_size:
            continue
        if torch.equal(tensor.roll(1, bd), tensor):
            continue
        inputs = tree_map(_roll_leaf(path, bd), ctx.inputs, "inputs")
        try:
            rolled_outputs = _eval_forward(ctx, inputs)
        except Exception as e:  # the model validates this input (e.g. sorted lengths)
            reason = type(e).__name__
            notes.append(
                f"{path} could not be probed: the model rejects shuffled values ({reason})"
            )
            continue
        checked += 1
        if structure_difference(ctx.eval_outputs, rolled_outputs):
            continue  # the outputs changed shape or structure: the input has an influence
        if all(
            bool(_close(a, b).all()) for _, a, b in paired_leaves(ctx.eval_outputs, rolled_outputs)
        ):
            problems.append(f"{path} has no influence on the output")

    if not checked:
        raise _Skip("no input could be probed" + (f"; {notes[0]}" if notes else ""))
    if problems:
        return CheckResult(
            "input_dependency",
            Status.FAIL,
            "; ".join(problems) + " (wrong slice/split, a variable reused by mistake?)",
            {"problems": problems, "warnings": warnings_, "notes": notes},
        )
    if warnings_:
        return CheckResult(
            "input_dependency",
            Status.WARN,
            "; ".join(warnings_),
            {"warnings": warnings_, "notes": notes},
        )
    return CheckResult(
        "input_dependency",
        Status.PASS,
        f"all {checked} input tensor(s) influence the output"
        + (f" ({'; '.join(notes)})" if notes else ""),
        {"notes": notes} if notes else {},
    )


def _absolute(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return (
        torch.zeros_like(like, dtype=torch.float32) if grad is None else grad.detach().abs().float()
    )


def _only_trailing_border(dead: torch.Tensor) -> bool:
    """Whether every dead position lies in a trailing band of one of the last two dimensions of
    a spatial input: pixels that a strided convolution or a patch embedding never covers (up to
    stride - 1 of them, e.g. 8 pixels for 14x14 patches of a 64x64 image). The band is at most a
    quarter of the dimension. Only for inputs with at least 3 dimensions per sample (e.g.
    C x H x W): in a sequence, an ignored last time step is more likely an off-by-one bug."""
    if dead.ndim < 3 or not bool(dead.any()):
        return False
    border = torch.zeros_like(dead)
    for dim in range(max(0, dead.ndim - 2), dead.ndim):
        size = dead.shape[dim]
        width = max(2, size // 4)
        if size <= 2 * width:
            continue
        index = [slice(None)] * dead.ndim
        index[dim] = slice(size - width, size)
        border[tuple(index)] = True
    return bool(border.any()) and not bool((dead & ~border).any())


def _input_gradients(
    ctx: _Context, paths: Sequence[str], *, train: bool
) -> dict[str, torch.Tensor | None]:
    """Gradients of an objective w.r.t. the float inputs at ``paths``: the surrogate of the
    eval-mode outputs, or the training objective in train mode."""
    grad_inputs: dict[str, torch.Tensor] = {}
    wanted = set(paths)

    def make_leaf(path: str, tensor: torch.Tensor) -> torch.Tensor:
        if path in wanted:
            grad_inputs[path] = tensor.detach().clone().requires_grad_(True)
            # a non-leaf copy, so that models may modify their inputs in-place
            return grad_inputs[path].clone()
        return tensor

    inputs = tree_map(make_leaf, ctx.inputs, "inputs")
    try:
        if train:
            _reseed(ctx.options.seed, ctx.cuda_devices)
            ctx.model.train()
            if ctx.forward_fn is not None:
                outputs = ctx.forward_fn(ctx.model, inputs, ctx.targets)
            else:
                outputs = call_model(ctx.model, inputs)
            objective = _objective(ctx, outputs)
        else:
            # cuDNN refuses backward passes through RNNs in eval mode
            with torch.backends.cudnn.flags(enabled=False):
                objective = _surrogate(_eval_forward(ctx, inputs, grad=True), ctx.options.seed)
    except Exception:
        if not train:
            raise
        return dict.fromkeys(paths)
    if objective is None or not objective.requires_grad:
        return dict.fromkeys(paths)
    grads = torch.autograd.grad(objective, [grad_inputs[p] for p in paths], allow_unused=True)
    return dict(zip(paths, grads))


def _one_step(ctx: _Context) -> None:
    """One optimizer step on the objective, from the snapshot (the next check restores it)."""
    assert ctx.optimizer is not None
    ctx.reset()
    optimizer = ctx.optimizer

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = _train_objective(ctx)
        if loss is not None and loss.requires_grad:
            loss.backward()
        return loss

    if needs_closure(optimizer):
        optimizer.step(closure)
    else:
        closure()
        optimizer.step()


def _roll_leaf(target: str, dim: int) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """``tree_map`` function that rolls the leaf at path ``target`` along ``dim``."""

    def roll(path: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.roll(1, dim) if path == target else tensor

    return roll


def _check_batch_independence(ctx: _Context) -> CheckResult:
    _require_forward(ctx)
    batch_size = _require_batch(ctx)
    bd, out_bd = ctx.batch_dim, ctx.output_batch_dim
    assert bd is not None and out_bd is not None  # guaranteed by _require_batch
    batched = dict(_batched(tree_leaves(ctx.inputs, "inputs"), batch_size, bd))
    if all(torch.equal(t[0], t[1]) for t in batched.values()):
        raise _Skip("samples 0 and 1 are identical")

    def replace_first(path: str, tensor: torch.Tensor) -> torch.Tensor:
        if path not in batched:
            return tensor
        tensor = tensor.clone()
        front = tensor.movedim(bd, 0)  # a view: writing to it modifies the clone
        front[0] = front[1]
        return tensor

    base_outputs = _eval_forward(ctx)
    base = _batched(tree_leaves(base_outputs, "output"), batch_size, out_bd)
    if not base:
        raise _Skip(f"no output has the batch dimension (dim {out_bd}){_OUT_HINT}")
    after_step = False
    float_outputs = [t for _, t in base if t.is_floating_point()]
    if (
        float_outputs
        and ctx.optimizer is not None
        and all(torch.allclose(t, t[:1].expand_as(t), rtol=1e-6, atol=1e-8) for t in float_outputs)
    ):
        # outputs that are (almost) the same for every sample at initialisation (zero-initialised
        # head, untrained BatchNorm statistics) cannot reveal a coupling: take one step first
        _one_step(ctx)
        base_outputs = _eval_forward(ctx)
        base = _batched(tree_leaves(base_outputs, "output"), batch_size, out_bd)
        after_step = True
    try:
        perturbed = _eval_forward(ctx, tree_map(replace_first, ctx.inputs, "inputs"))
    except Exception as e:  # the model validates the batch (e.g. sorted lengths)
        raise _Skip(
            f"the model rejects the batch with sample 0 replaced by sample 1 ({type(e).__name__}: "
            f"{str(e)[:100]})"
        ) from e
    # a per-sample output without the batch dimension (e.g. the boxes of each image) may change
    # its shape with sample 0; a batched output or the set of outputs may not
    changed = structure_difference(
        base_outputs,
        perturbed,
        shape_matters=lambda t: t.ndim > out_bd and t.shape[out_bd] == batch_size,
    )
    if changed:
        return CheckResult(
            "batch_independence",
            Status.FAIL,
            f"changing sample 0 changed the structure of the outputs: {changed} - the outputs "
            "depend on the contents of other samples",
            {"structure": changed},
        )
    new = _batched(tree_leaves(perturbed, "output"), batch_size, out_bd)

    def compare(base, new):
        leaked = torch.zeros(batch_size - 1, dtype=torch.bool)
        weak = torch.zeros(batch_size - 1, dtype=torch.bool)
        largest = 0.0
        new_by_path = dict(new)
        for path, a in base:  # paired by path: dict keys may come back in another order
            if path not in new_by_path:
                continue
            b = new_by_path[path]
            same = _close(a[1:], b[1:]).reshape(batch_size - 1, -1).all(dim=1)
            leaked |= ~same.cpu()
            if a.is_floating_point() and a.numel():
                # untouched samples normally get bit-identical outputs: a change far below the
                # tolerance is still a (weak) coupling, e.g. near-uniform attention at init
                diff = (a[1:] - b[1:]).abs().float().reshape(batch_size - 1, -1).amax(dim=1)
                scale = float(a.detach().abs().float().mean()) or 1.0
                relative = diff.cpu() / scale
                weak |= relative > 1e-6
                largest = max(largest, float(relative.nan_to_num(0.0).max()))
        return leaked, weak, largest

    leaked, weak, largest = compare(base, new)
    tf32_note = ""
    on_cuda = any(t.is_cuda for _, t in base)
    tf32 = torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    if bool(leaked.any()) and on_cuda and tf32:
        # TF32 kernels may tile the batch, so that a sample's rounding depends on its neighbours
        # (seen on real models: ~1e-4 of the output scale); a real coupling survives without TF32
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            base_fp32 = _batched(tree_leaves(_eval_forward(ctx), "output"), batch_size, out_bd)
            new_fp32 = _batched(
                tree_leaves(
                    _eval_forward(ctx, tree_map(replace_first, ctx.inputs, "inputs")), "output"
                ),
                batch_size,
                out_bd,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        leaked_fp32, weak, largest_fp32 = compare(base_fp32, new_fp32)
        if not bool(leaked_fp32.any()):
            tf32_note = (
                f"; with TF32 the outputs of other samples changed by up to {largest:.1e} of "
                "the output scale - batch-dependent rounding of TF32 kernels, not of the model"
            )
            leaked, largest = leaked_fp32, largest_fp32
    if bool(leaked.any()):
        indices = (leaked.nonzero().flatten() + 1).tolist()
        return CheckResult(
            "batch_independence",
            Status.FAIL,
            f"changing sample 0 changed the outputs of {len(indices)}/{batch_size - 1} other "
            f"sample(s), e.g. {indices[:5]}: the model mixes information across the batch "
            "(view/reshape instead of transpose/permute, a batch_first mix-up, softmax or "
            "normalisation over the batch dimension?)",
            {"affected_samples": indices},
        )
    if bool(weak.any()):
        indices = (weak.nonzero().flatten() + 1).tolist()
        return CheckResult(
            "batch_independence",
            Status.WARN,
            f"changing sample 0 changed the outputs of {len(indices)}/{batch_size - 1} other "
            f"sample(s) slightly (up to {largest:.1e} of the output scale): a weak coupling "
            "between samples, or batch-dependent numerics of a kernel" + tf32_note,
            {"affected_samples": indices, "relative_change": largest},
        )
    return CheckResult(
        "batch_independence",
        Status.PASS,
        "outputs of a sample do not depend on other samples in the batch (eval mode"
        + (
            ", after one optimizer step: the outputs are constant at initialisation)"
            if after_step
            else ")"
        )
        + tf32_note,
    )


def _check_overfit_batch(ctx: _Context) -> CheckResult:
    steps = ctx.options.overfit_steps
    if steps <= 0:
        raise _Skip("disabled (overfit_steps=0)")
    _require_loss(ctx)
    if ctx.optimizer is None:
        raise _Skip("the model has no trainable parameters")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Anomaly Detection has been enabled")
        anomaly = (
            torch.autograd.detect_anomaly(check_nan=True)
            if ctx.options.detect_anomaly
            else contextlib.nullcontext()
        )
    with anomaly:
        try:
            return _overfit(ctx, steps)
        except RuntimeError as e:
            if not ctx.options.detect_anomaly:
                raise
            first_line = str(e).splitlines()[0]
            return CheckResult(
                "overfit_batch",
                Status.FAIL,
                f"torch.autograd.detect_anomaly: {first_line} (the warning printed above shows "
                "the forward-pass traceback of the offending operation)",
            )


def _overfit(ctx: _Context, steps: int) -> CheckResult:
    ratio, target = ctx.options.overfit_ratio, ctx.options.overfit_target
    optimizer = ctx.optimizer
    assert optimizer is not None
    _require_train(ctx)
    hint = _HALF_HINT if _half_precision(ctx) else ""

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_loss(ctx, _train_forward(ctx))
        loss.backward()
        return loss

    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_loss(ctx, _train_forward(ctx))
        value = loss.item()
        if not math.isfinite(value):
            return CheckResult(
                "overfit_batch",
                Status.FAIL,
                f"the loss became {value} after {step} step(s) (lr too high? unstable ops?){hint}",
                {"losses": losses},
            )
        losses.append(value)
        goal = target if target is not None else losses[0] * ratio if losses[0] > 0 else None
        if goal is not None and value <= goal:
            return CheckResult(
                "overfit_batch",
                Status.PASS,
                f"loss {losses[0]:.4g} -> {value:.4g} in {step} step(s)",
                {"losses": losses},
            )
        loss.backward()
        non_finite = [
            n
            for n, p in ctx.named_params()
            if p.grad is not None and not bool(torch.isfinite(_dense(p.grad)).all())
        ]
        if non_finite:
            return CheckResult(
                "overfit_batch",
                Status.FAIL,
                f"gradients became NaN/inf at step {step} (loss {value:.4g}) in "
                f"{_names(non_finite)}; rerun with detect_anomaly=True to find the operation{hint}",
                {"losses": losses, "non_finite": non_finite, "step": step},
            )
        if needs_closure(optimizer):
            optimizer.step(closure)
        else:
            optimizer.step()

    initial, best = losses[0], min(losses)
    details = {"losses": losses}
    if best >= initial - 1e-3 * abs(initial):
        return CheckResult(
            "overfit_batch",
            Status.FAIL,
            f"the loss did not decrease on a single batch in {steps} steps ({initial:.4g} -> best "
            f"{best:.4g}); check the learning rate, the optimizer and that targets match outputs",
            details,
        )
    if target is None and initial <= 0:
        return CheckResult(
            "overfit_batch",
            Status.PASS,
            f"loss {initial:.4g} -> {best:.4g} in {steps} steps",
            details,
        )
    goal_text = f"the target {target:.4g}" if target is not None else f"{ratio:.0%}"
    return CheckResult(
        "overfit_batch",
        Status.WARN,
        f"the loss decreased ({initial:.4g} -> best {best:.4g}, {best / initial:.0%} of the "
        f"initial value) but did not reach {goal_text} within {steps} steps; the model may lack "
        "capacity, the lr may be too low, or it simply needs more overfit_steps",
        details,
    )


def _check_gradcheck(ctx: _Context) -> CheckResult:
    if not ctx.options.gradcheck:
        raise _Skip("disabled (gradcheck=True verifies custom autograd.Function backward passes)")
    _require_forward(ctx)
    float_paths = [p for p, t in tree_leaves(ctx.inputs, "inputs") if t.is_floating_point()]
    if not float_paths:
        raise _Skip("needs floating point inputs")
    n, bd = ctx.options.gradcheck_samples, ctx.batch_dim
    # float64 copies of the model and of the first samples; the user's model stays untouched
    model = copy.deepcopy(ctx.model).double()

    def shrink(path: str, tensor: torch.Tensor) -> torch.Tensor:
        if bd is not None and tensor.ndim > bd and tensor.shape[bd] == ctx.batch_size:
            tensor = tensor.narrow(bd, 0, min(n, tensor.shape[bd]))
        return tensor.double() if path in float_paths else tensor

    small = tree_map(shrink, ctx.inputs, "inputs")
    leaves = dict(tree_leaves(small, "inputs"))
    xs = tuple(leaves[p].detach().clone().requires_grad_(True) for p in float_paths)

    def function(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        by_path = dict(zip(float_paths, values))
        inputs = tree_map(lambda p, t: by_path.get(p, t), small, "inputs")
        outputs = _eval_forward(ctx, inputs, grad=True, model=model)
        return tuple(t for _, t in tree_leaves(outputs) if t.is_floating_point())

    with torch.backends.cudnn.flags(enabled=False):
        ok = _run_gradcheck(function, xs, ctx)
    if not ok:
        return CheckResult(
            "gradcheck",
            Status.FAIL,
            "analytical and numerical gradients w.r.t. the inputs disagree (float64); a custom "
            "autograd.Function has a wrong backward(), or the model is not differentiable here",
        )
    return CheckResult(
        "gradcheck",
        Status.PASS,
        f"analytical gradients match finite differences ({len(xs)} input tensor(s), "
        f"{min(n, ctx.batch_size or n)} sample(s), float64)",
    )


def _run_gradcheck(function: Callable, xs: tuple, ctx: _Context) -> bool:
    return torch.autograd.gradcheck(
        function,
        xs,
        eps=ctx.options.gradcheck_eps,
        atol=ctx.options.gradcheck_atol,
        raise_exception=False,
    )


def _check_gpu(ctx: _Context) -> CheckResult:
    if not ctx.options.require_gpu:
        raise _Skip("require_gpu=False")
    if not torch.cuda.is_available():
        return CheckResult("gpu", Status.FAIL, "CUDA is not available")
    if not ctx.cuda_devices:
        return CheckResult("gpu", Status.FAIL, "CUDA is available but the model is on the CPU")
    names = ", ".join(f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in ctx.cuda_devices)
    return CheckResult("gpu", Status.PASS, f"model is on {names}")


_CHECK_FUNCTIONS: dict[str, Callable[[_Context], CheckResult]] = {
    "forward": _check_forward,
    "output_shape": _check_output_shape,
    "output_finite": _check_output_finite,
    "output_range": _check_output_range,
    "output_varies": _check_output_varies,
    "eval_determinism": _check_eval_determinism,
    "loss": _check_loss,
    "initial_loss": _check_initial_loss,
    "optimizer_coverage": _check_optimizer_coverage,
    "gradient_flow": _check_gradient_flow,
    "gradient_magnitude": _check_gradient_magnitude,
    "params_change": _check_params_change,
    "frozen_params": _check_frozen_params,
    "input_dependency": _check_input_dependency,
    "batch_independence": _check_batch_independence,
    "overfit_batch": _check_overfit_batch,
    "gradcheck": _check_gradcheck,
    "gpu": _check_gpu,
}
assert tuple(_CHECK_FUNCTIONS) == CHECKS


def _run_check(ctx: _Context, name: str) -> CheckResult:
    """Run one check (after the ones it depends on), turning errors into FAIL results."""
    if name != "forward" and "forward" not in ctx.ran:
        _run_check(ctx, "forward")
    if CHECKS.index(name) > CHECKS.index("loss") and "loss" not in ctx.ran:
        _run_check(ctx, "loss")
    ctx.ran.add(name)
    start = time.perf_counter()
    try:
        ctx.reset()
        result = _CHECK_FUNCTIONS[name](ctx)
    except _Skip as skip:
        result = CheckResult(name, Status.SKIP, str(skip))
    except Exception as e:  # a crash inside user code is a finding
        result = CheckResult(name, Status.FAIL, f"raised {type(e).__name__}: {e}")
    return dataclasses.replace(result, seconds=time.perf_counter() - start)


def check_model(
    model: nn.Module,
    inputs: Any,
    targets: Any = None,
    *,
    loss_fn: LossFn | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    output_shape: Sequence[int | None] | None = None,
    output_range: tuple[float | None, float | None] | None = None,
    frozen: Iterable[str] = (),
    overfit_steps: int = 200,
    overfit_ratio: float = 0.1,
    overfit_target: float | None = None,
    detect_anomaly: bool = False,
    grad_norm_threshold: float = 1e3,
    gradcheck: bool = False,
    require_gpu: bool = False,
    plugin: PluginSpec | None = None,
    forward_fn: ForwardFn | None = None,
    eval_forward_fn: EvalForwardFn | None = None,
    batch_dim: int | None = DEFAULT_BATCH_DIM,
    output_batch_dim: int | None = None,
    skip: Iterable[str] = (),
    seed: int = 0,
    verbose: bool = False,
) -> Report:
    """Run every applicable sanity check on ``model`` with one batch and return a :class:`Report`.

    The PyTorch state of the model and optimizer (parameters, buffers, optimizer state,
    gradients, train/eval flags, uninitialised lazy modules), the caller's input tensors and the
    PyTorch, Python and NumPy global RNG states are left as they were. Python attributes that the
    model's own code changes (e.g. a call counter in ``forward``) are not restored.

    Arguments:
        model: the model to check. Inputs/targets are moved to the device of its parameters.
        inputs: one batch of inputs - a tensor, a tuple/list (passed as ``model(*inputs)``) or a
            dict (passed as ``model(**inputs)``). The batch dimension is ``batch_dim`` (first by
            default).
        targets: targets for ``loss_fn`` (any structure), or ``None``.
        loss_fn: ``loss_fn(outputs, targets)`` (or ``loss_fn(outputs)`` when ``targets`` is
            ``None``) returning a scalar tensor. If omitted, a ``loss`` returned by the model
            (``outputs.loss`` / ``outputs["loss"]``, Hugging Face style) is used; otherwise the
            loss-based checks are skipped and gradient checks use a random projection of the
            outputs.
        optimizer: the optimizer you train with. Defaults to ``Adam(lr=1e-3)`` over the trainable
            parameters (then ``optimizer_coverage`` is skipped).
        output_shape: expected shape of the (first) output tensor; ``None``/``-1`` = any size.
        output_range: ``(low, high)`` inclusive bounds for all outputs; either may be ``None``.
        frozen: names (or name prefixes such as ``"backbone"``) of parameters that must not
            change during training, in addition to those with ``requires_grad=False``.
        overfit_steps: maximum number of steps for ``overfit_batch``; ``0`` disables it.
        overfit_ratio: ``overfit_batch`` passes once the loss drops to this fraction of its
            initial value.
        overfit_target: absolute loss value for ``overfit_batch`` to reach (replaces
            ``overfit_ratio``).
        detect_anomaly: run ``overfit_batch`` under ``torch.autograd.detect_anomaly`` to name the
            operation that produces NaN gradients (slow).
        grad_norm_threshold: ratio of a gradient norm to its parameter norm (at least 1) above
            which ``gradient_magnitude`` warns.
        gradcheck: enable the ``gradcheck`` check - ``torch.autograd.gradcheck`` w.r.t. the float
            inputs on a float64 copy of the model and the first 2 samples. Use it when the model
            contains custom ``autograd.Function`` classes. Slow for large inputs.
        require_gpu: fail if CUDA is unavailable or the model is not on a GPU.
        plugin: a :class:`~torchtest2.plugin_base.Plugin` (instance, class or registered name
            such as ``"lightning"``, ``"huggingface"`` or ``"torchvision_detection"``) that
            knows how to run a family of models. It supplies ``forward_fn``, ``eval_forward_fn``,
            the loss, ``batch_dim`` and possibly the optimizer; explicit arguments win.
        forward_fn: ``forward_fn(model, inputs, targets) -> outputs`` for training-mode passes,
            for models whose training forward differs from ``model(inputs)`` (e.g. torchvision
            detection models, which take the targets and return a dict of losses).
        eval_forward_fn: ``eval_forward_fn(model, inputs) -> outputs`` for eval-mode passes.
        batch_dim: the batch dimension of the inputs and outputs (default 0, or the plugin's;
            e.g. 1 for sequence-first tensors). ``None`` means the inputs have no batch
            dimension; the batch checks are then skipped.
        output_batch_dim: the batch dimension of the outputs, when it differs from that of the
            inputs (e.g. 0 for a sequence-first RNN classifier with inputs ``[T, N, F]`` and
            outputs ``[N, C]``). Default: ``batch_dim``.
        skip: names of checks to skip (see ``torchtest2.CHECKS``).
        seed: seed of the forked RNG the checks run with (dropout masks, surrogate loss).
        verbose: print the report.

    Returns:
        a :class:`~torchtest2.report.Report`; call ``.raise_on_failure()`` to turn failures into
        an ``AssertionError``.
    """
    skip = {skip} if isinstance(skip, str) else set(skip)
    unknown = skip - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown check name(s) {sorted(unknown)}; valid names: {CHECKS}")
    if "forward" in skip:
        raise ValueError("the 'forward' check cannot be skipped - every other check needs it")
    if isinstance(frozen, str):
        frozen = (frozen,)
    hooks = _resolve_hooks(
        model,
        plugin,
        loss_fn=loss_fn,
        optimizer=optimizer,
        forward_fn=forward_fn,
        eval_forward_fn=eval_forward_fn,
        batch_dim=batch_dim,
        output_batch_dim=output_batch_dim,
    )
    options = _Options(
        output_shape=output_shape,
        output_range=output_range,
        frozen=tuple(frozen),
        overfit_steps=overfit_steps,
        overfit_ratio=overfit_ratio,
        overfit_target=overfit_target,
        detect_anomaly=detect_anomaly,
        grad_norm_threshold=grad_norm_threshold,
        gradcheck=gradcheck,
        require_gpu=require_gpu,
        batch_dim=hooks.batch_dim,
        output_batch_dim=hooks.output_batch_dim,
        loss_label=hooks.loss_label,
        seed=seed,
    )
    results = []
    with _session(model, inputs, targets, options, hooks) as ctx:
        for name in CHECKS:
            if name in skip:
                results.append(CheckResult(name, Status.SKIP, "skipped by the caller"))
                ctx.ran.add(name)  # a skipped 'loss' must not be run as a dependency
            else:
                results.append(_run_check(ctx, name))
        checked = ctx.model  # an initialised copy of a model with lazy modules
        info = {
            "parameters": sum(p.numel() for p in checked.parameters()),
            "trainable": sum(
                p.numel() for n, p in checked.named_parameters() if not ctx.is_frozen(n, p)
            ),
            "device": str(model_device(checked)),
            "batch size": ctx.batch_size,
            "objective": ctx.objective_name(),
        }
        if hooks.plugin is not None:
            info["plugin"] = hooks.plugin.name or type(hooks.plugin).__name__
        if ctx.notes:
            info["notes"] = "; ".join(ctx.notes)
    report = Report(type(model).__name__, results, info)
    if verbose:
        print(report)
    return report


def assert_model_ok(
    model: nn.Module, inputs: Any, targets: Any = None, *, strict: bool = False, **kwargs: Any
) -> Report:
    """:func:`check_model` followed by :meth:`Report.raise_on_failure` - one line in a pytest test.

    Arguments:
        strict: treat warnings as failures.
        **kwargs: forwarded to :func:`check_model`.

    Raises:
        ModelCheckError: (an ``AssertionError``) listing the failed checks and the full report.
    """
    return check_model(model, inputs, targets, **kwargs).raise_on_failure(strict=strict)
