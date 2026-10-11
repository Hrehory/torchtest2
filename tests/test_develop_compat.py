"""Compatibility with the API of the 2021 develop branch. Tests marked "develop:" are ported from
that branch's test suite with only the imports adapted."""

import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtest2 import ModelTester, check_model
from torchtest2.asserts import assert_tensor_range
from torchtest2.exceptions import NaNLossException, RangeException, TorchTestError
from torchtest2.plugin_base import DefaultPlugin, PluginBase
from torchtest2.test_suite import DEFAULT_TEST_SUITE_CONFIG, TestSuite
from torchtest2.utils import plugin_discovery


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x.reshape(-1, 784))


@pytest.fixture
def standard_deeplearning_components():
    torch.manual_seed(0)
    model = MnistModel()
    return {
        "model": model,
        "optimizer": optim.SGD(model.parameters(), lr=1e-3),
        "loss_criterion": F.cross_entropy,
        "batch": [torch.randn((32, 1, 28, 28)), torch.randint(0, 10, (32,))],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }


class ExamplePlugin(PluginBase):
    """torchtest2/plugins/example_plugin.py from develop (mask_outputs is no longer needed)."""

    def compute_loss(self, loss_criterion, model_outputs, targets):
        return loss_criterion(model_outputs, targets)

    def train_inference(self, model, inputs):
        return model(inputs[0], inputs[1])

    def test_inference(self, model, inputs):
        return model(inputs[0], inputs[1])

    def split_batch(self, batch):
        return {
            "inputs": (batch[0][0], batch[0][1]),
            "targets": batch[1],
            "batch_size": batch[1].shape[0],
            "batch_dim": 0,
        }


class TwoInputModel(nn.Module):
    def __init__(self, use_x2: bool):
        super().__init__()
        self.linear = nn.Linear(30, 10)
        self.linear2 = nn.Linear(30, 10)
        self.use_x2 = use_x2

    def forward(self, x1, x2):
        x1 = x1.reshape(-1, 30)
        x2 = x2.reshape(-1, 30)
        return torch.cat((self.linear(x1), self.linear2(x2 if self.use_x2 else x1)), 1)


def two_input_components(use_x2: bool):
    model = TwoInputModel(use_x2)
    batch = [(torch.randn(32, 1, 30), torch.randn(32, 1, 30)), torch.randint(0, 10, (32,))]
    return {
        "model": model,
        "optimizer": optim.SGD(model.parameters(), lr=0.01),
        "loss_criterion": F.cross_entropy,
        "batch": batch,
    }


# ------------------------------------------------------------------------------------------------
# assertions and exceptions


def test_assert_tensor_range():  # develop: tests/test_asserts.py
    t = torch.tensor([1, 2, 3])
    assert_tensor_range(t, 0, relation="greater", comparison_condition="all")
    assert_tensor_range(t, 4, relation="less", comparison_condition="all")
    assert_tensor_range(t, 2, relation="greater", comparison_condition="any")
    assert_tensor_range(t, 2, relation="less", comparison_condition="any")

    with pytest.raises(RangeException):
        assert_tensor_range(t, 4, relation="greater", comparison_condition="all")
    with pytest.raises(RangeException):
        assert_tensor_range(t, 5, relation="greater", comparison_condition="any")
    with pytest.raises(RangeException):
        assert_tensor_range(t, 0, relation="less", comparison_condition="any")
    with pytest.raises(RangeException, match="all elements of tensor must be less than 2!"):
        assert_tensor_range(t, 2, relation="less", comparison_condition="all")
    with pytest.raises(ValueError):
        assert_tensor_range(t, 2, relation="bigger")


def test_nan_loss_exception_exists():
    assert issubclass(NaNLossException, TorchTestError)
    assert str(NaNLossException()) == "The loss is NaN!"


# ------------------------------------------------------------------------------------------------
# ModelTester


class OutOfRangeModel(MnistModel):
    def __init__(self, low: float):
        super().__init__()
        self.low = low

    def forward(self, x):
        out = torch.rand_like(super().forward(x)) + self.low  # range [low, low + 1]
        out[0] = 0.5  # something can be in range, but all values must be
        return out


def test_model_output_range(standard_deeplearning_components):  # develop
    class InRange(MnistModel):
        def forward(self, x):
            return 1.9 * (torch.rand_like(super().forward(x)) - 0.5)  # [-0.95, 0.95]

    standard_deeplearning_components["model"] = InRange()
    ModelTester(**standard_deeplearning_components).test_model_output_range()


@pytest.mark.parametrize(
    ("low", "message"),
    [
        (1.0, "all elements of tensor must be less than 1!"),
        (-2.0, "all elements of tensor must be greater than -1!"),
    ],
)
def test_model_output_range_raise_exception(standard_deeplearning_components, low, message):
    # develop: test_test_model_output_range_raise_exception / ..._exception2
    standard_deeplearning_components["model"] = OutOfRangeModel(low)
    with pytest.raises(RangeException) as exc:
        ModelTester(**standard_deeplearning_components).test_model_output_range()
    assert str(exc.value) == message


def test_all_to_double_is_accepted_and_ignored(standard_deeplearning_components):
    with pytest.warns(DeprecationWarning, match="all_to_double"):
        tester = ModelTester(**standard_deeplearning_components, all_to_double=True)
    assert standard_deeplearning_components["model"].linear.weight.dtype is torch.float32
    tester.test_parameters_change()


def test_nested_inputs_follow_the_model_device():
    # develop: test_transfer2device (skipped there: "there is a need to implement a plugin")
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, main, more, another):
            assert all(t.device == self.fc.weight.device for t in (main, *more.values(), *another))
            extra = more["input1"].sum() + more["input2"].sum() + sum(t.sum() for t in another)
            return self.fc(main) + extra

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nested().to(device)
    inputs = {
        "main": torch.randn(8, 10),
        "more": {"input1": torch.randn(3, 3), "input2": torch.randn(2, 2)},
        "another": [torch.randn(2, 2), torch.randn(4, 4)],
    }
    targets = (torch.randint(0, 10, (8,)), torch.randint(0, 50, (8,)))
    report = check_model(model, inputs, targets, loss_fn=lambda out, t: F.cross_entropy(out, t[0]))
    assert report["forward"].status.value == "PASS", str(report)
    assert report["loss"].status.value == "PASS", str(report)
    tester = ModelTester(
        model,
        lambda out, t: F.cross_entropy(out, t[0]),
        optim.SGD(model.parameters(), lr=0.1),
        [inputs, targets],
    )
    assert tester.batch[0]["more"]["input1"].device.type == device.type
    assert tester.batch[1][1].device.type == device.type
    assert inputs["main"].device.type == "cpu"  # the caller's tensors are not moved


# ------------------------------------------------------------------------------------------------
# plugins


def test_plugin_base_subclass(standard_deeplearning_components):
    tester = ModelTester(**two_input_components(use_x2=False), plugin=ExamplePlugin())
    tester.test_parameters_change()
    with pytest.raises(TorchTestError, match=r"inputs\[1\]"):
        tester.test_input_dependency()
    ModelTester(**two_input_components(use_x2=True), plugin=ExamplePlugin()).test_input_dependency()
    ModelTester(**standard_deeplearning_components, plugin=DefaultPlugin()).test_suite()


def test_plugin_base_is_abstract():
    with pytest.raises(TypeError):
        PluginBase()


def test_plugin_discovery(monkeypatch):
    class FakeEntryPoint:
        name = "example_plugin"

        @staticmethod
        def load():
            return ExamplePlugin

    monkeypatch.setattr(
        "torchtest2.plugins.entry_points",
        lambda group: [FakeEntryPoint()] if group == "plugins" else [],
    )
    assert plugin_discovery(group="plugins") == {"example_plugin": ExamplePlugin}
    plugins = plugin_discovery()  # built-in plugins + both entry-point groups
    assert plugins["example_plugin"] is ExamplePlugin
    assert {"lightning", "huggingface", "torchvision_detection"} <= set(plugins)
    assert isinstance(plugins["example_plugin"](), PluginBase)


# ------------------------------------------------------------------------------------------------
# TestSuite (develop: tests/test_test_suite.py)


def test_suite_batch_independence(standard_deeplearning_components):  # develop
    test_suite = TestSuite(
        **standard_deeplearning_components, test_suite_config={"test_batch_independence": True}
    )
    assert test_suite.test_suite() == {"test_batch_independence": True}


def test_suite_default_tests(standard_deeplearning_components, capsys):  # develop
    results = TestSuite(**standard_deeplearning_components, test_suite_config=None).test_suite()
    assert results == {name: True for name, on in DEFAULT_TEST_SUITE_CONFIG.items() if on}
    out = capsys.readouterr().out
    assert out.startswith("SUMMARY")
    assert "test_anomaly" in out and "✓" in out
    assert "not performed tests: test_gpu_available, test_train_params_change" in out


@pytest.mark.filterwarnings("ignore:Error detected in")
def test_suite_nan_outputs(standard_deeplearning_components, capsys):  # develop
    class ModelNanOutputs(MnistModel):
        def forward(self, x):
            out = super().forward(x)
            with torch.no_grad():
                out *= math.nan
            return out

    standard_deeplearning_components["model"] = ModelNanOutputs()
    standard_deeplearning_components["optimizer"] = optim.SGD(
        standard_deeplearning_components["model"].parameters(), lr=0.001
    )
    results = TestSuite(**standard_deeplearning_components).test_suite()  # does not raise
    assert results["test_model_output_nan"] is False
    assert results["test_model_output_inf"] is True
    assert "×" in capsys.readouterr().out


def test_suite_all_tests_with_frozen_layer(standard_deeplearning_components):
    # develop demo.ipynb, "second example": a frozen layer and every test enabled
    class TwoLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(784, 64)
            self.linear2 = nn.Linear(64, 10)
            self.linear.requires_grad_(False)

        def forward(self, x):
            return self.linear2(F.relu(self.linear(x.reshape(-1, 784))))

    model = TwoLayers()
    standard_deeplearning_components.update(
        model=model, optimizer=optim.Adam(model.parameters(), lr=1e-2)
    )
    # every test that develop's TestSuite had (it had no GPU test)
    config = {name: name != "test_gpu_available" for name in DEFAULT_TEST_SUITE_CONFIG}
    suite = TestSuite(
        **standard_deeplearning_components,
        test_suite_config=config,
        train_params=list(model.linear2.named_parameters(prefix="linear2")),
        nontrain_params=list(model.linear.named_parameters(prefix="linear")),
        expected_output_shape=(32, 10),
        output_range=(-100, 100),
        expected_loss=0.1,
        max_epochs=500,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        results = suite.test_suite()
    assert all(results.values()), results
    assert suite.non_performed_test == ["test_gpu_available"]


def test_suite_validation(standard_deeplearning_components):
    with pytest.raises(ValueError, match="unknown test_suite_config"):
        TestSuite(**standard_deeplearning_components, test_suite_config={"test_foo": True})
    with pytest.raises(ValueError, match="train_params is not given"):
        TestSuite(
            **standard_deeplearning_components,
            test_suite_config={"test_train_params_change": True},
        )


def test_suite_with_plugin():  # develop demo.ipynb, "fourth example"
    results = TestSuite(**two_input_components(use_x2=False), plugin=ExamplePlugin()).test_suite()
    assert results["test_input_dependency"] is False
    results = TestSuite(**two_input_components(use_x2=True), plugin=ExamplePlugin()).test_suite()
    assert all(results.values()), results
