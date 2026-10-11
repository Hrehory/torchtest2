"""Granular, exception-raising API (the original torchtest2 interface).

For a one-call report use :func:`torchtest2.check_model`; :meth:`ModelTester.check` bridges both.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn

from torchtest2._session import (
    DEFAULT_BATCH_DIM,
    EvalForwardFn,
    ForwardFn,
    PluginSpec,
    _Context,
    _DefaultBatchDim,
    _Options,
    _resolve_hooks,
    _session,
)
from torchtest2._tree import model_device, to_device
from torchtest2.asserts import (
    assert_gpu_available,
    assert_never_inf,
    assert_never_nan,
    assert_shape,
    assert_tensor_range,
)
from torchtest2.checks import (
    _changed_after_step,
    _compute_loss,
    _eval_forward,
    _run_check,
    _train_forward,
    check_model,
    needs_closure,
)
from torchtest2.exceptions import (
    AnomalyDetectionError,
    BatchIndependencyException,
    GradCheckException,
    InputDependencyException,
    OverfitException,
    OverfitNaNException,
    ParametersChangeException,
)
from torchtest2.plugin_base import LossFn
from torchtest2.report import Report, Status
from torchtest2.utils import count_parameters

# default model output range used by the output range test
MODEL_OUT_LOW = -1
MODEL_OUT_HIGH = 1

DEFAULT_SUITE_CONFIG: dict[str, bool] = {
    "test_gpu_available": False,
    "test_parameters_change": True,
    "test_train_params_change": False,
    "test_nontrain_params_change": False,
    "test_output_consistency": False,
    "test_model_output_range": False,
    "test_model_output_nan": True,
    "test_model_output_inf": True,
    "test_input_dependency": True,
    "test_batch_independence": True,
    "test_overfit": False,
    "test_gradcheck": False,
    "test_anomaly": True,
}
"""The tests of :meth:`ModelTester.test_suite` and :class:`~torchtest2.test_suite.TestSuite`,
their default flags and run order. The keys are the names of the ``ModelTester`` methods."""

SUITE_CONFIG_ALIASES: dict[str, str] = {
    "test_params_change": "test_parameters_change",
    "test_output_range": "test_model_output_range",
    "test_nan_values": "test_model_output_nan",
    "test_inf_values": "test_model_output_inf",
}
"""Key names of the 2021 ``master`` version, accepted as aliases."""


def normalize_suite_config(config: Mapping[str, bool] | None) -> dict[str, bool]:
    """``None`` -> the defaults; otherwise resolve aliases, reject unknown keys, and treat
    missing keys as disabled (as both 2021 versions did)."""
    if config is None:
        return dict(DEFAULT_SUITE_CONFIG)
    resolved = {SUITE_CONFIG_ALIASES.get(k, k): bool(v) for k, v in config.items()}
    unknown = set(resolved) - set(DEFAULT_SUITE_CONFIG)
    if unknown:
        raise ValueError(
            f"unknown test_suite_config key(s) {sorted(unknown)}; "
            f"valid keys: {list(DEFAULT_SUITE_CONFIG)}"
        )
    return {name: resolved.get(name, False) for name in DEFAULT_SUITE_CONFIG}


ParamSpec = Iterable[tuple[str, nn.Parameter] | nn.Parameter | str]


class ModelTester:
    """Pre-train tests for one model / loss / optimizer / batch combination.

    Every ``test_*`` method raises a :class:`~torchtest2.exceptions.TorchTestError` (an
    ``AssertionError``) on failure. Each method works on a snapshot: the parameters, buffers,
    optimizer state, gradients and train/eval flags are restored afterwards - also when the test
    fails - and the global RNG state is not touched.

    Arguments:
        model: the model under test.
        loss_criterion: ``loss_criterion(outputs, targets)`` returning a scalar tensor.
        optimizer: the optimizer used for training.
        batch: ``[inputs, targets]``; ``inputs`` may be a tensor, tuple or dict.
        device: if given, the model is moved there (``model.to(device)``); the batch always
            follows the model's device. The list you pass is not modified.
        min_epochs, max_epochs, n_epochs_stop: early-stopping settings of
            :meth:`overfit_single_batch` (stop after ``n_epochs_stop`` steps without improvement,
            but not before ``min_epochs`` steps). ``min_epochs`` is also the default number of
            steps of :meth:`test_anomaly`.
        test_suite_config: which tests :meth:`test_suite` runs; keys are the names of the test
            methods (see ``DEFAULT_SUITE_CONFIG``, the default when ``None``). Missing keys are
            disabled. The 2021 ``master`` names (``test_params_change``, ``test_nan_values``,
            ...) are accepted as aliases.
        train_params, nontrain_params: parameters (``(name, param)`` pairs, parameters or names)
            for ``test_train_params_change`` / ``test_nontrain_params_change``;
            ``nontrain_params`` defaults to the parameters with ``requires_grad=False``.
        output_range: ``(low, high)`` for :meth:`test_model_output_range` (strict bounds, as in
            2021; ``check_model(output_range=...)`` uses inclusive bounds).
        expected_loss, overfit_relation: defaults of :meth:`test_overfit`.
        expected_output_shape: default of :meth:`test_output_consistency`.
        forward_fn, eval_forward_fn, batch_dim, output_batch_dim: see
            :func:`torchtest2.check_model`.
        plugin: a :class:`~torchtest2.plugin_base.Plugin` (instance, class or name); its
            ``split_batch`` turns ``batch`` into inputs and targets. 2021 plugins
            (``PluginBase`` subclasses or objects with its methods) work too.
        seed: seed of the forked RNG the tests run with.
        all_to_double: accepted for compatibility and ignored (``test_gradcheck`` works on a
            float64 copy); passing ``True`` emits a ``DeprecationWarning``.

    Note:
        Pass the arguments after ``n_epochs_stop`` by keyword: their positional order differs
        between the 2021 ``master`` and ``develop`` versions.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_criterion: LossFn,
        optimizer: torch.optim.Optimizer | None,
        batch: Sequence[Any],
        device: torch.device | str | None = None,
        min_epochs: int = 50,
        max_epochs: int = 10000,
        n_epochs_stop: int = 10,
        test_suite_config: Mapping[str, bool] | None = None,
        train_params: ParamSpec | None = None,
        nontrain_params: ParamSpec | None = None,
        output_range: tuple[float | None, float | None] = (MODEL_OUT_LOW, MODEL_OUT_HIGH),
        expected_loss: float | None = None,
        overfit_relation: Literal["greater", "less"] | None = None,
        expected_output_shape: Sequence[int | None] | None = None,
        forward_fn: ForwardFn | None = None,
        eval_forward_fn: EvalForwardFn | None = None,
        batch_dim: int | None = DEFAULT_BATCH_DIM,
        plugin: PluginSpec | None = None,
        seed: int = 0,
        all_to_double: bool = False,
        output_batch_dim: int | None = None,
    ) -> None:
        if all_to_double:
            warnings.warn(
                "all_to_double is no longer needed and is ignored: test_gradcheck runs on a "
                "float64 copy of the model, the model itself is not converted",
                DeprecationWarning,
                stacklevel=2,
            )
        self.suite_config = normalize_suite_config(test_suite_config)
        if self.suite_config["test_train_params_change"] and not train_params:
            raise ValueError("test_train_params_change is enabled but train_params is not given")
        if plugin is not None:
            from torchtest2.plugins import get_plugin

            plugin = get_plugin(plugin)
            spec = plugin.split_batch(batch)
            batch = [spec["inputs"], spec["targets"]]
            if isinstance(batch_dim, _DefaultBatchDim) and "batch_dim" in spec:
                batch_dim = spec["batch_dim"]
        if len(batch) != 2:
            raise ValueError("batch must be [inputs, targets]")
        if device is not None:
            model.to(device)
        self.model = model
        self.loss_criterion = loss_criterion
        self.forward_fn = forward_fn
        self.eval_forward_fn = eval_forward_fn
        self.batch_dim = batch_dim
        self.hooks = _resolve_hooks(
            model,
            plugin,
            loss_fn=loss_criterion,
            optimizer=optimizer,
            forward_fn=forward_fn,
            eval_forward_fn=eval_forward_fn,
            batch_dim=batch_dim,
            output_batch_dim=output_batch_dim,
        )
        self.output_batch_dim = output_batch_dim
        if self.hooks.optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.hooks.optimizer = torch.optim.Adam(trainable, lr=1e-3) if trainable else None
        self.optimizer = self.hooks.optimizer
        self.plugin = self.hooks.plugin
        self.device = model_device(model)
        self.batch = [to_device(item, self.device) for item in batch]
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.n_epochs_stop = n_epochs_stop
        self.train_params = train_params
        self.nontrain_params = nontrain_params
        self.output_range = output_range
        self.expected_loss = expected_loss
        self.overfit_relation = overfit_relation
        self.expected_output_shape = expected_output_shape
        self.seed = seed

    # the 2021 attribute names of two suite flags
    @property
    def test_output_range(self) -> bool:
        return self.suite_config["test_model_output_range"]

    @property
    def test_gpu_available(self) -> bool:
        return self.suite_config["test_gpu_available"]

    # ----------------------------------------------------------------------------------------

    def _session(self, **options: Any):
        return _session(
            self.model,
            self.batch[0],
            self.batch[1],
            _Options(
                seed=self.seed,
                batch_dim=self.hooks.batch_dim,
                output_batch_dim=self.hooks.output_batch_dim,
                loss_label=self.hooks.loss_label,
                **options,
            ),
            self.hooks,
        )

    def _names(self, params: ParamSpec) -> list[str]:
        by_id = {id(p): n for n, p in self.model.named_parameters()}
        names = []
        for item in params:
            if isinstance(item, str):
                names.append(item)
            else:
                param = item[1] if isinstance(item, tuple) else item
                if id(param) not in by_id:
                    raise ValueError("a given parameter does not belong to the model")
                names.append(by_id[id(param)])
        return names

    def _forward_step(self) -> Any:
        """Forward the batch in eval mode (restoring the train/eval flags afterwards)."""
        with self._session() as ctx:
            return _eval_forward(ctx)

    def _raise_if_failed(self, name: str, exception: type[Exception], **options: Any) -> None:
        with self._session(**options) as ctx:
            result = _run_check(ctx, name)
        if result.status is Status.FAIL:
            raise exception(result.message)

    # ----------------------------------------------------------------------------------------

    def suite_tests(self) -> list[tuple[str, Callable[[], Any]]]:
        """The tests enabled in ``test_suite_config``, in run order, as ``(name, callable)``."""
        nontrain = self.nontrain_params
        if nontrain is None:
            nontrain = [(n, p) for n, p in self.model.named_parameters() if not p.requires_grad]
        available: dict[str, Callable[[], Any]] = {
            "test_gpu_available": assert_gpu_available,
            "test_parameters_change": lambda: self.test_parameters_change(should_change=True),
            "test_train_params_change": lambda: self.test_parameters_change(
                should_change=True, params=self.train_params
            ),
            "test_nontrain_params_change": lambda: self.test_parameters_change(
                should_change=False, params=nontrain
            ),
            "test_output_consistency": self.test_output_consistency,
            "test_model_output_range": self.test_model_output_range,
            "test_model_output_nan": self.test_model_output_nan,
            "test_model_output_inf": self.test_model_output_inf,
            "test_input_dependency": self.test_input_dependency,
            "test_batch_independence": self.test_batch_independence,
            "test_overfit": self.test_overfit,
            "test_gradcheck": self.test_gradcheck,
            "test_anomaly": self.test_anomaly,
        }
        return [(name, available[name]) for name, on in self.suite_config.items() if on]

    def test_suite(self) -> None:
        """Run the tests enabled in ``test_suite_config`` and raise the first failure.

        :class:`~torchtest2.test_suite.TestSuite` runs the same tests but collects all results,
        and :meth:`check` returns a full :func:`~torchtest2.check_model` report.

        Raises:
            the exception of the first failing test (all are ``TorchTestError`` subclasses).
        """
        for _, test in self.suite_tests():
            test()

    def test_parameters_change(
        self, should_change: bool = True, params: ParamSpec | None = None
    ) -> None:
        """Check that ``params`` do (or do not) change after one training step.

        Arguments:
            should_change: whether the parameters are expected to change.
            params: ``(name, param)`` pairs (e.g. from ``named_parameters()``), parameters or
                names; defaults to all parameters with ``requires_grad=True``.

        Raises:
            ParametersChangeException: listing every offending parameter.
        """
        if params is None:
            names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        else:
            names = self._names(params)
        with self._session() as ctx:
            ctx.reset()
            changed = _changed_after_step(ctx)
        bad = [n for n in names if changed[n] != should_change]
        if bad:
            raise ParametersChangeException(bad, should_change)

    def test_model_output_range(self) -> None:
        """Check that all outputs lie strictly within ``output_range`` (``None`` = unbounded).

        Raises:
            RangeException: e.g. "all elements of tensor must be less than 1!"
        """
        low, high = self.output_range
        outputs = self._forward_step()
        if low is not None:
            assert_tensor_range(outputs, low, "greater", "all")
        if high is not None:
            assert_tensor_range(outputs, high, "less", "all")

    def test_model_output_nan(self) -> None:
        """Raises NaNTensorException if the model outputs contain NaN."""
        assert_never_nan(self._forward_step(), "model output")

    def test_model_output_inf(self) -> None:
        """Raises InfTensorException if the model outputs contain +/-inf."""
        assert_never_inf(self._forward_step(), "model output")

    def test_input_dependency(self) -> None:
        """Check that every input (and every feature of float inputs) influences the output.

        Raises:
            InputDependencyException: if (part of) an input is ignored by the model.
        """
        self._raise_if_failed("input_dependency", InputDependencyException)

    def test_batch_independence(self) -> None:
        """Check that the output of one sample does not depend on other samples in the batch
        (in eval mode, so BatchNorm is not an issue).

        Raises:
            BatchIndependencyException: if information leaks between samples.
        """
        self._raise_if_failed("batch_independence", BatchIndependencyException)

    def test_output_consistency(
        self, expected_output_shape: Sequence[int | None] | None = None
    ) -> None:
        """Check the output shape; ``None``/``-1`` matches any size. Defaults to the
        ``expected_output_shape`` given to the constructor.

        Raises:
            ConsistencyException: if the shape differs.
        """
        expected = expected_output_shape or self.expected_output_shape
        if expected is None:
            raise ValueError("test_output_consistency needs expected_output_shape")
        assert_shape(self._forward_step(), expected)

    def test_gradcheck(self, eps: float = 1e-6, atol: float = 1e-4) -> None:
        """``torch.autograd.gradcheck`` w.r.t. the float inputs, on a float64 *copy* of the model
        and the first two samples. Useful when the model contains custom
        ``autograd.Function`` s.

        Raises:
            GradCheckException: if analytical and numerical gradients disagree.
        """
        self._raise_if_failed(
            "gradcheck", GradCheckException, gradcheck=True, gradcheck_eps=eps, gradcheck_atol=atol
        )

    def test_anomaly(self, iter_number: int | None = None) -> None:
        """Train for ``iter_number`` steps (default ``min_epochs``) under
        ``torch.autograd.detect_anomaly``, which raises as soon as a backward operation
        produces NaN.

        Raises:
            AnomalyDetectionError: with the original ``RuntimeError`` as ``__cause__``.
        """
        steps = self.min_epochs if iter_number is None else iter_number
        optimizer = self._require_optimizer()
        with self._session() as ctx, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Anomaly Detection has been enabled")
            ctx.reset()
            try:
                with torch.autograd.detect_anomaly(check_nan=True):
                    for _ in range(steps):
                        _training_step(ctx, optimizer)
            except RuntimeError as e:
                raise AnomalyDetectionError(str(e).splitlines()[0]) from e

    def test_overfit(
        self,
        expected_loss: float | None = None,
        overfit_relation: Literal["greater", "less"] | None = None,
    ) -> float:
        """Overfit the batch (see :meth:`overfit_single_batch`) and compare the best loss with
        ``expected_loss`` (default: the constructor value, else 0.01).

        Arguments:
            expected_loss: the loss to compare with.
            overfit_relation: ``"greater"`` (default) raises if the best loss is greater than
                ``expected_loss``; ``"less"`` raises if it is smaller.

        Returns:
            the best loss.

        Raises:
            OverfitNaNException: if the loss became NaN/inf.
            OverfitException: if the expectation is not met.
        """
        expected = next(v for v in (expected_loss, self.expected_loss, 0.01) if v is not None)
        relation = overfit_relation or self.overfit_relation or "greater"
        if relation not in ("greater", "less"):
            raise ValueError("overfit_relation must be 'greater' or 'less'")
        loss = self.overfit_single_batch()
        if not math.isfinite(loss):
            raise OverfitNaNException()
        if (relation == "greater" and loss > expected) or (relation == "less" and loss < expected):
            raise OverfitException(loss, expected, relation)
        return loss

    def overfit_single_batch(self) -> float:
        """Train on the single batch until the loss stops improving for ``n_epochs_stop`` steps
        (but at least ``min_epochs`` and at most ``max_epochs`` steps), then restore the model.

        Returns:
            the lowest loss reached, or the first non-finite loss.
        """
        with self._session() as ctx:
            ctx.reset()
            return self._overfit(ctx)

    def _require_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            raise ValueError("the model has no trainable parameters, so it cannot be trained")
        return self.optimizer

    def _overfit(self, ctx: _Context) -> float:
        optimizer = self._require_optimizer()
        best = math.inf
        epochs_no_improve = 0
        for epoch in range(self.max_epochs):
            value = _training_step(ctx, optimizer)
            if not math.isfinite(value):
                return value
            if value < best:
                best = value
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch + 1 >= self.min_epochs and epochs_no_improve >= self.n_epochs_stop:
                break
        return best

    def get_number_of_parameters(self, trainable: bool = True) -> int:
        """Number of trainable (default) or all scalar parameters."""
        return count_parameters(self.model, trainable_only=trainable)

    def check(self, **kwargs: Any) -> Report:
        """Run :func:`torchtest2.check_model` with this tester's model, batch, loss, optimizer,
        plugin and settings: ``nontrain_params`` become ``frozen``, ``expected_loss`` the
        ``overfit_target``, ``expected_output_shape`` the ``output_shape``, and the
        ``test_model_output_range``/``test_gradcheck``/``test_gpu_available`` flags enable the
        corresponding checks. ``kwargs`` are forwarded and take precedence."""
        if self.test_output_range:
            kwargs.setdefault("output_range", self.output_range)
        if self.expected_output_shape is not None:
            kwargs.setdefault("output_shape", self.expected_output_shape)
        if self.nontrain_params is not None:
            kwargs.setdefault("frozen", self._names(self.nontrain_params))
        if self.expected_loss is not None and self.overfit_relation != "less":
            kwargs.setdefault("overfit_target", self.expected_loss)
        kwargs.setdefault("gradcheck", self.suite_config["test_gradcheck"])
        kwargs.setdefault("require_gpu", self.test_gpu_available)
        kwargs.setdefault("plugin", self.plugin)
        kwargs.setdefault("forward_fn", self.forward_fn)
        kwargs.setdefault("eval_forward_fn", self.eval_forward_fn)
        kwargs.setdefault("batch_dim", self.batch_dim)
        kwargs.setdefault("output_batch_dim", self.output_batch_dim)
        kwargs.setdefault("seed", self.seed)
        return check_model(
            self.model,
            self.batch[0],
            self.batch[1],
            loss_fn=self.loss_criterion,
            optimizer=self.optimizer,
            **kwargs,
        )


def _training_step(ctx: _Context, optimizer: torch.optim.Optimizer) -> float:
    """One optimizer step on the batch (with a closure for optimizers such as LBFGS); returns
    the loss before the step, or a non-finite value without stepping."""

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_loss(ctx, _train_forward(ctx))
        loss.backward()
        return loss

    loss = closure()
    value = loss.item()
    if math.isfinite(value):
        if needs_closure(optimizer):
            optimizer.step(closure)
        else:
            optimizer.step()
    return value


__all__ = [
    "DEFAULT_SUITE_CONFIG",
    "MODEL_OUT_HIGH",
    "MODEL_OUT_LOW",
    "SUITE_CONFIG_ALIASES",
    "ModelTester",
    "normalize_suite_config",
]
