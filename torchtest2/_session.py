"""The execution model of :func:`torchtest2.check_model`: how a model is run (the caller's
arguments completed by a plugin), the snapshot that is restored before every check and at the end,
the forked and reseeded random generators, and the context the checks share.

What is restored: parameters, buffers, optimizer state, gradients, train/eval flags, uninitialised
lazy modules (checked as an initialised copy), the PyTorch CPU/CUDA RNG and the Python and NumPy
global RNGs; inputs are copied. Other Python state changed by the model's own code is not.
"""

from __future__ import annotations

import contextlib
import copy
import random
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parameter import is_lazy

from torchtest2._tree import (
    batch_size_of,
    call_model,
    model_device,
    to_device,
)
from torchtest2.plugin_base import LossFn, Plugin, overrides

ForwardFn = Callable[[nn.Module, Any, Any], Any]
EvalForwardFn = Callable[[nn.Module, Any], Any]
PluginSpec = Plugin | type | str


class _DefaultBatchDim(int):
    """``0``, but recognisable as "not passed by the caller" so that a plugin can override it."""

    def __repr__(self) -> str:
        return "0"


DEFAULT_BATCH_DIM: int = _DefaultBatchDim(0)


@dataclass
class _Options:
    output_shape: Sequence[int | None] | None = None
    output_range: tuple[float | None, float | None] | None = None
    frozen: tuple[str, ...] = ()
    overfit_steps: int = 200
    overfit_ratio: float = 0.1
    overfit_target: float | None = None
    detect_anomaly: bool = False
    grad_norm_threshold: float = 1e3
    gradcheck: bool = False
    gradcheck_samples: int = 2
    gradcheck_eps: float = 1e-6
    gradcheck_atol: float = 1e-4
    require_gpu: bool = False
    batch_dim: int | None = 0
    output_batch_dim: int | None = None  # None: same as batch_dim
    loss_label: str = "loss_fn"
    seed: int = 0


@dataclass
class _Hooks:
    """How to run the model: the caller's arguments, completed by a plugin if one is given."""

    loss_fn: LossFn | None = None
    optimizer: torch.optim.Optimizer | None = None
    forward_fn: ForwardFn | None = None
    eval_forward_fn: EvalForwardFn | None = None
    batch_dim: int | None = 0
    loss_label: str = "loss_fn"
    plugin: Any = None
    output_batch_dim: int | None = None


def _resolve_hooks(
    model: nn.Module,
    plugin: PluginSpec | None,
    *,
    loss_fn: LossFn | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    forward_fn: ForwardFn | None = None,
    eval_forward_fn: EvalForwardFn | None = None,
    batch_dim: int | None = DEFAULT_BATCH_DIM,
    output_batch_dim: int | None = None,
) -> _Hooks:
    """Explicit arguments win; a plugin fills in the rest."""
    hooks = _Hooks(loss_fn, optimizer, forward_fn, eval_forward_fn, batch_dim)
    hooks.output_batch_dim = output_batch_dim
    if plugin is None:
        hooks.batch_dim = None if batch_dim is None else int(batch_dim)
        return hooks
    from torchtest2.plugins import get_plugin  # the plugins import this module's dependencies

    plugin = get_plugin(plugin)
    hooks.plugin = plugin
    hooks.forward_fn = forward_fn or plugin.forward
    hooks.eval_forward_fn = eval_forward_fn or plugin.eval_forward
    # input and output batch dimensions are resolved independently: an explicit batch_dim does
    # not discard the plugin's output_batch_dim
    if isinstance(batch_dim, _DefaultBatchDim):
        hooks.batch_dim = plugin.batch_dim
    else:
        hooks.batch_dim = None if batch_dim is None else int(batch_dim)
    if output_batch_dim is None:
        hooks.output_batch_dim = getattr(plugin, "output_batch_dim", None)
    if optimizer is None:
        hooks.optimizer = plugin.configure_optimizer(model)
    if overrides(plugin, "loss"):
        user_loss_fn = loss_fn

        def plugin_loss(outputs: Any, targets: Any = None) -> Any:
            return plugin.loss(outputs, targets, user_loss_fn)

        hooks.loss_fn = plugin_loss
        hooks.loss_label = f"{plugin.name or type(plugin).__name__} plugin"
    return hooks


class _Snapshot:
    """In-memory copy of everything a check may modify: parameters, buffers (including
    non-persistent ones, which ``state_dict()`` leaves out), optimizer state, gradients and
    train/eval flags."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
        self.model = model
        self.optimizer = optimizer
        # parameters and buffers are kept on the CPU to spare accelerator memory
        state_dict = model.state_dict()
        self.state = OrderedDict(
            (
                k,
                v.detach().to("cpu", copy=True)
                if isinstance(v, torch.Tensor)
                else copy.deepcopy(v),
            )
            for k, v in state_dict.items()
        )
        # per-module versions, which some modules' _load_from_state_dict rely on (e.g. MNASNet)
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            self.state._metadata = copy.deepcopy(metadata)  # type: ignore[attr-defined]
        self.params = {
            name: self.state[name] if name in self.state else p.detach().to("cpu", copy=True)
            for name, p in model.named_parameters()
        }
        self.optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer else None
        self.grads = [
            None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
        ]
        self.training = [m.training for m in model.modules()]
        # buffers registered with persistent=False are not in the state dict
        self.non_persistent = [
            (module, name, buffer.detach().to("cpu", copy=True), buffer.device)
            for module in model.modules()
            for name, buffer in module._buffers.items()
            if name in module._non_persistent_buffers_set and buffer is not None
        ]

    def restore(self) -> None:
        with torch.no_grad():
            self.model.load_state_dict(self.state)
            for module, name, saved, device in self.non_persistent:
                current = module._buffers.get(name)
                if (
                    isinstance(current, torch.Tensor)
                    and current.shape == saved.shape
                    and current.dtype == saved.dtype
                ):
                    current.copy_(saved)  # in place: references to the buffer stay valid
                else:  # the forward pass replaced or resized it
                    module._buffers[name] = saved.to(device, copy=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            # deep-copied again: load_state_dict may keep references to the given tensors
            self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        for p, grad in zip(self.model.parameters(), self.grads):
            p.grad = None if grad is None else grad.clone()
        for module, training in zip(self.model.modules(), self.training):
            module.training = training


@dataclass
class _Context:
    model: nn.Module
    inputs: Any
    targets: Any
    loss_fn: LossFn | None
    optimizer: torch.optim.Optimizer | None
    user_optimizer: bool
    options: _Options
    snapshot: _Snapshot
    batch_size: int | None
    cuda_devices: list[int]
    forward_fn: ForwardFn | None = None
    eval_forward_fn: EvalForwardFn | None = None
    # filled in while the checks run
    eval_outputs: Any = None
    forward_error: str | None = None
    train_error: str | None = None
    notes: list[str] = field(default_factory=list)
    _aliases: dict[int, list[str]] | None = None
    loss_source: str | None = None
    loss_error: str | None = None
    grads: dict[str, torch.Tensor | None] | None = None
    changed: dict[str, bool] | None = None
    still_zero: set[str] | None = None
    ran: set[str] = field(default_factory=set)

    def named_params(self) -> list[tuple[str, nn.Parameter]]:
        return list(self.model.named_parameters())

    def is_frozen(self, name: str, param: nn.Parameter) -> bool:
        """Frozen = ``requires_grad=False`` or listed in ``frozen=`` by the user, under any of
        the names of a shared (tied) parameter."""
        if not param.requires_grad:
            return True
        if not self.options.frozen:
            return False
        return any(
            _matches_prefix(alias, self.options.frozen) for alias in self.aliases(param, name)
        )

    def aliases(self, param: nn.Parameter, name: str) -> list[str]:
        if self._aliases is None:
            self._aliases = {}
            for alias, p in self.model.named_parameters(remove_duplicate=False):
                self._aliases.setdefault(id(p), []).append(alias)
        return self._aliases.get(id(param), [name])

    def trainable_names(self) -> list[str]:
        return [n for n, p in self.named_params() if not self.is_frozen(n, p)]

    def reset(self) -> None:
        self.snapshot.restore()
        _reseed(self.options.seed, self.cuda_devices)

    @property
    def batch_dim(self) -> int | None:
        """Batch dimension of the inputs."""
        return self.options.batch_dim

    @property
    def output_batch_dim(self) -> int | None:
        """Batch dimension of the outputs (that of the inputs unless given separately)."""
        if self.options.batch_dim is None or self.options.output_batch_dim is None:
            return self.options.batch_dim
        return self.options.output_batch_dim

    def objective_name(self) -> str:
        return "surrogate" if self.loss_source is None or self.loss_error else self.loss_source


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        prefix = prefix.rstrip(".")
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def _reseed(seed: int, cuda_devices: Sequence[int]) -> None:
    """Seed the PyTorch RNGs, and the Python and NumPy global RNGs that a ``forward`` may use, so
    that compared forward passes draw the same random numbers."""
    torch.default_generator.manual_seed(seed)
    for index in cuda_devices:
        with torch.cuda.device(index):
            torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy = sys.modules.get("numpy")  # only if the user's code imported it
    if numpy is not None:
        numpy.random.seed(seed)


@contextlib.contextmanager
def _fork_python_rngs() -> Iterator[None]:
    """Restore the Python and NumPy (if imported) global RNG states on exit."""
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy is not None:
            numpy.random.set_state(numpy_state)


def _cuda_devices(model: nn.Module) -> list[int]:
    indices = set()
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.device.type == "cuda":
            index = tensor.device.index
            indices.add(torch.cuda.current_device() if index is None else index)
    return sorted(indices)


@contextlib.contextmanager
def _session(
    model: nn.Module,
    inputs: Any,
    targets: Any,
    options: _Options,
    hooks: _Hooks,
) -> Iterator[_Context]:
    """Prepare a :class:`_Context`; on exit restore the model/optimizer and the global RNGs."""
    device = model_device(model)
    cuda_devices = _cuda_devices(model)
    inputs, targets = to_device(inputs, device), to_device(targets, device)
    notes = []
    optimizer = hooks.optimizer
    if any(is_lazy(p) for p in model.parameters()):
        # initialising lazy modules cannot be undone in place (they even change their class):
        # check an initialised copy and leave the caller's model uninitialised. An optimizer is
        # copied together with the model, so that the copy updates the copied parameters
        # (lazy parameters are materialised in place, so the copied references stay valid).
        try:
            model, optimizer = _copy_with_optimizer(model, optimizer)
        except Exception as e:
            raise TypeError(
                "the model has uninitialised lazy modules and the optimizer cannot be copied "
                f"together with it ({type(e).__name__}: {e}); run one forward pass before "
                "creating the optimizer, or call check_model without an optimizer"
            ) from e
        notes.append("lazy modules were initialised by one forward pass on a copy")
        _materialize(model, inputs, hooks, cuda_devices)
    user_optimizer = optimizer is not None
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3) if trainable else None
    snapshot = _Snapshot(model, optimizer)
    ctx = _Context(
        model=model,
        inputs=inputs,
        targets=targets,
        loss_fn=hooks.loss_fn,
        optimizer=optimizer,
        user_optimizer=user_optimizer,
        options=options,
        snapshot=snapshot,
        batch_size=batch_size_of(inputs, options.batch_dim),
        cuda_devices=cuda_devices,
        forward_fn=hooks.forward_fn,
        eval_forward_fn=hooks.eval_forward_fn,
        notes=notes,
    )
    with torch.random.fork_rng(devices=cuda_devices), _fork_python_rngs():
        try:
            yield ctx
        finally:
            snapshot.restore()


def _copy_with_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer | None
) -> tuple[nn.Module, torch.optim.Optimizer | None]:
    """Deep copies of ``model`` and ``optimizer`` in which the optimizer refers to the copied
    parameters. The optimizer's ``__dict__`` is copied as a whole, with the model's memo:
    ``Optimizer.__getstate__`` (used by a plain ``deepcopy``) keeps only the defaults, the state
    and the parameter groups, and would drop attributes such as LBFGS's ``_params`` or those of
    a custom optimizer."""
    memo: dict[int, Any] = {}
    model_copy = copy.deepcopy(model, memo)
    if optimizer is None:
        return model_copy, None
    optimizer_copy = copy.copy(optimizer)
    optimizer_copy.__dict__ = copy.deepcopy(optimizer.__dict__, memo)
    return model_copy, optimizer_copy


def _materialize(model: nn.Module, inputs: Any, hooks: _Hooks, cuda_devices: list[int]) -> None:
    """Run one eval-mode forward pass so that lazy modules create their parameters (on a forked
    RNG; the train/eval flags are restored)."""
    training = [m.training for m in model.modules()]
    with torch.random.fork_rng(devices=cuda_devices), _fork_python_rngs(), torch.no_grad():
        model.eval()
        if hooks.eval_forward_fn is not None:
            hooks.eval_forward_fn(model, inputs)
        else:
            call_model(model, inputs)
    for module, flag in zip(model.modules(), training):
        module.training = flag
