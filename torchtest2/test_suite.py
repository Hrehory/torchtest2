"""``TestSuite``: run the :class:`~torchtest2.ModelTester` tests enabled in a config and collect
the results instead of stopping at the first failure (the 2021 ``TestSuite`` API).

:meth:`ModelTester.test_suite` runs the same tests with the same config but raises the first
failure; :func:`torchtest2.check_model` gives the fullest overview as a
:class:`~torchtest2.report.Report`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from torchtest2._session import PluginSpec
from torchtest2.exceptions import TorchTestError
from torchtest2.model_tester import (
    DEFAULT_SUITE_CONFIG,
    MODEL_OUT_HIGH,
    MODEL_OUT_LOW,
    ModelTester,
    ParamSpec,
)
from torchtest2.plugin_base import LossFn

logger = logging.getLogger(__name__)

DEFAULT_TEST_SUITE_CONFIG = DEFAULT_SUITE_CONFIG
"""The same configuration as :data:`torchtest2.model_tester.DEFAULT_SUITE_CONFIG`."""


@dataclass
class TestConfig:
    """One test of the suite and its outcome."""

    __test__ = False  # not a pytest test class

    test_name: str
    callback: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""


class TestSuite:
    """Run the tests enabled in ``test_suite_config`` and summarise them.

    The arguments are those of :class:`~torchtest2.ModelTester` (in the order of the 2021
    ``TestSuite``). ``test_suite_config`` uses the keys of ``DEFAULT_SUITE_CONFIG``; ``None``
    means the default configuration and missing keys are disabled.
    """

    __test__ = False  # not a pytest test class

    def __init__(
        self,
        model: nn.Module,
        loss_criterion: LossFn,
        optimizer: torch.optim.Optimizer | None,
        batch: Sequence[Any],
        device: torch.device | str | None = None,
        test_suite_config: Mapping[str, bool] | None = None,
        min_epochs: int = 50,
        max_epochs: int = 10000,
        n_epochs_stop: int = 10,
        expected_loss: float = 0.01,
        overfit_relation: Literal["greater", "less"] = "greater",
        expected_output_shape: Sequence[int | None] | None = None,
        train_params: ParamSpec | None = None,
        nontrain_params: ParamSpec | None = None,
        output_range: tuple[float | None, float | None] = (MODEL_OUT_LOW, MODEL_OUT_HIGH),
        plugin: PluginSpec | None = None,
        **model_tester_kwargs: Any,
    ) -> None:
        self.model_tester = ModelTester(
            model,
            loss_criterion,
            optimizer,
            batch,
            device,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            n_epochs_stop=n_epochs_stop,
            test_suite_config=test_suite_config,
            train_params=train_params,
            nontrain_params=nontrain_params,
            output_range=output_range,
            expected_loss=expected_loss,
            overfit_relation=overfit_relation,
            expected_output_shape=expected_output_shape,
            plugin=plugin,
            **model_tester_kwargs,
        )
        self.test_suite_config = self.model_tester.suite_config
        self.test_list = [TestConfig(name, test) for name, test in self.model_tester.suite_tests()]
        self.non_performed_test = [n for n, on in self.test_suite_config.items() if not on]

    def test_suite(self) -> dict[str, bool]:
        """Run the enabled tests, log each failure, print a summary.

        Returns:
            ``{test name: passed}``.
        """
        for test in self.test_list:
            try:
                test.callback(**test.kwargs)
                test.success, test.message = True, ""
            except TorchTestError as e:
                test.success, test.message = False, str(e)
                logger.error("%s failed: %s", test.test_name, e)
        self.display_results()
        return {test.test_name: test.success for test in self.test_list}

    def display_results(self) -> None:
        """Print one line per test (✓ / ×) and the tests that were not performed."""
        print("SUMMARY")
        for test in self.test_list:
            mark = "✓" if test.success else f"×  {test.message.splitlines()[0]}"
            print(f"  {test.test_name:<28} {mark}")
        print("not performed tests: " + ", ".join(self.non_performed_test))
