"""torchtest2 - sanity checks (pre-train) and behavioural tests (post-train) for PyTorch models.

Quick start::

    import torchtest2

    report = torchtest2.check_model(model, inputs, targets, loss_fn=loss_fn, optimizer=optimizer)
    print(report)
"""

from torchtest2.asserts import (
    assert_all_finite,
    assert_gpu_available,
    assert_in_range,
    assert_never_inf,
    assert_never_nan,
    assert_shape,
    assert_tensor_range,
)
from torchtest2.checks import CHECKS, assert_model_ok, check_model
from torchtest2.exceptions import ModelCheckError, TorchTestError
from torchtest2.model_tester import ModelTester
from torchtest2.plugin_base import Plugin, PluginBase
from torchtest2.plugins import get_plugin, plugin_discovery
from torchtest2.post_train import (
    assert_directional,
    assert_invariance,
    assert_minimum_functionality,
)
from torchtest2.report import CheckResult, Report, Status
from torchtest2.test_suite import TestSuite
from torchtest2.utils import count_parameters, seed_everything

__version__ = "0.1.0"

__all__ = [
    "CHECKS",
    "CheckResult",
    "ModelCheckError",
    "ModelTester",
    "Plugin",
    "PluginBase",
    "Report",
    "Status",
    "TestSuite",
    "TorchTestError",
    "__version__",
    "assert_all_finite",
    "assert_directional",
    "assert_gpu_available",
    "assert_in_range",
    "assert_invariance",
    "assert_minimum_functionality",
    "assert_model_ok",
    "assert_never_inf",
    "assert_never_nan",
    "assert_shape",
    "assert_tensor_range",
    "check_model",
    "count_parameters",
    "get_plugin",
    "plugin_discovery",
    "seed_everything",
]
