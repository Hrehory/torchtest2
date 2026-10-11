"""The granular ModelTester API, including regression tests for the 2021 bugs."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtest2 import ModelTester, Status
from torchtest2.exceptions import (
    AnomalyDetectionError,
    BatchIndependencyException,
    ConsistencyException,
    GpuUnusedException,
    InputDependencyException,
    NaNTensorException,
    OverfitException,
    OverfitNaNException,
    ParametersChangeException,
    RangeException,
)


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x.reshape(-1, 784))


class WrongMnistModel(MnistModel):
    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(10, 10)  # not used in forward!


@pytest.fixture
def components():
    torch.manual_seed(0)
    model = MnistModel()
    return {
        "model": model,
        "optimizer": optim.SGD(model.parameters(), lr=1e-2),
        "loss_criterion": F.cross_entropy,
        "batch": [torch.randn((32, 1, 28, 28)), torch.randint(0, 10, (32,))],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }


def test_get_number_of_parameters(components):
    components["model"] = nn.Linear(10, 10, bias=False)
    components["optimizer"] = optim.SGD(components["model"].parameters(), lr=0.1)
    tester = ModelTester(**components)
    assert tester.get_number_of_parameters() == 100
    components["model"].requires_grad_(False)
    assert tester.get_number_of_parameters() == 0
    assert tester.get_number_of_parameters(trainable=False) == 100


def test_overfit_single_batch(components):
    components["optimizer"] = optim.Adam(components["model"].parameters(), lr=1e-2)
    tester = ModelTester(**components, max_epochs=500)
    loss = tester.overfit_single_batch()
    assert loss < 0.05


def test_overfit_early_stopping_works(components):
    # regression: the improvement test was inverted, so training always ran max_epochs steps
    calls = []
    loss_fn = components["loss_criterion"]

    def counting_loss(out, y):
        calls.append(1)
        return loss_fn(out, y) * 0  # constant loss: no improvement at all

    components["loss_criterion"] = counting_loss
    ModelTester(
        **components, min_epochs=5, n_epochs_stop=3, max_epochs=10000
    ).overfit_single_batch()
    assert len(calls) == 5


def test_overfit_restores_model(components):
    model = components["model"]
    tester = ModelTester(**components, max_epochs=20)
    before = model.linear.weight.clone()
    tester.overfit_single_batch()
    assert torch.equal(model.linear.weight, before)


def test_parameters_change(components):
    ModelTester(**components).test_parameters_change(should_change=True)


def test_parameters_change_specified_params(components):
    model = components["model"]
    model.linear.bias.requires_grad = False
    train = [(n, p) for n, p in model.named_parameters() if n != "linear.bias"]
    frozen = [(n, p) for n, p in model.named_parameters() if n == "linear.bias"]
    tester = ModelTester(**components)
    tester.test_parameters_change(should_change=True, params=train)
    tester.test_parameters_change(should_change=False, params=frozen)
    tester.test_parameters_change(should_change=False, params=["linear.bias"])
    with pytest.raises(ParametersChangeException, match="linear.bias"):
        tester.test_parameters_change(should_change=True, params=[model.linear.bias])


def test_parameters_change_bad_model(components):
    model = WrongMnistModel()
    components.update(model=model, optimizer=optim.Adadelta(model.parameters(), lr=1e-3))
    with pytest.raises(ParametersChangeException) as info:
        ModelTester(**components).test_parameters_change(should_change=True)
    assert info.value.param_names == ["linear2.weight", "linear2.bias"]


def test_parameters_change_bad_optimizer(components):
    components["model"] = WrongMnistModel()  # optimizer still holds the old model's parameters
    with pytest.raises(ParametersChangeException):
        ModelTester(**components).test_parameters_change(should_change=True)


def test_model_is_restored_after_failed_test(components):
    # regression: the reset decorator did not run when the test raised
    model = components["model"]
    tester = ModelTester(**components)
    before = model.linear.weight.clone()
    with pytest.raises(ParametersChangeException):
        tester.test_parameters_change(should_change=False)
    assert torch.equal(model.linear.weight, before)


def test_two_testers_coexist(components):
    # regression: every tester deleted the shared tmp_data folder in __del__
    first = ModelTester(**components)
    second = ModelTester(**components)
    del first
    second.test_parameters_change()


def test_batch_list_is_not_mutated(components):
    batch = components["batch"]
    original = list(batch)
    ModelTester(**components)
    assert all(a is b for a, b in zip(batch, original))


def test_input_dependency_raises():
    class WrongInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(30, 10)
            self.linear2 = nn.Linear(30, 10)

        def forward(self, x):
            x1, x2 = torch.split(x, 30, dim=2)
            out1 = self.linear(x1.reshape(-1, 30))
            out2 = self.linear2(x1.reshape(-1, 30))  # x1 instead of x2!
            return torch.cat((out1, out2), 1)

    model = WrongInputModel()
    tester = ModelTester(
        model,
        F.cross_entropy,
        optim.SGD(model.parameters(), lr=0.01),
        [torch.randn((32, 1, 60)), torch.randint(0, 10, (32,))],
    )
    with pytest.raises(InputDependencyException, match="30/60 features"):
        tester.test_input_dependency()


def test_input_dependency_multiple_inputs():
    class Model(nn.Module):
        def __init__(self, use_x2: bool):
            super().__init__()
            self.linear = nn.Linear(30, 10)
            self.linear2 = nn.Linear(30, 10)
            self.use_x2 = use_x2

        def forward(self, x1, x2):
            other = x2 if self.use_x2 else x1
            return torch.cat((self.linear(x1), self.linear2(other)), 1)

    batch = [(torch.randn(32, 30), torch.randn(32, 30)), torch.randint(0, 20, (32,))]
    good, bad = Model(True), Model(False)
    ModelTester(
        good, F.cross_entropy, optim.SGD(good.parameters(), lr=0.1), batch
    ).test_input_dependency()
    with pytest.raises(InputDependencyException, match=r"inputs\[1\]"):
        ModelTester(
            bad, F.cross_entropy, optim.SGD(bad.parameters(), lr=0.1), batch
        ).test_input_dependency()


def test_input_dependency_ignores_relu_zeros(components):
    # regression: a single zero gradient element (common with ReLU) used to fail the test
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    components.update(model=model, optimizer=optim.SGD(model.parameters(), lr=0.1))
    ModelTester(**components).test_input_dependency()


def test_batch_independence(components):
    ModelTester(**components).test_batch_independence()

    class Leaky(MnistModel):
        def forward(self, x):
            out = super().forward(x)
            return out - out.mean(dim=0)  # batch statistics in eval mode

    components["model"] = Leaky()
    with pytest.raises(BatchIndependencyException):
        ModelTester(**components).test_batch_independence()


def test_output_consistency(components):
    tester = ModelTester(**components)
    tester.test_output_consistency(torch.Size([32, 10]))
    tester.test_output_consistency((None, 10))
    with pytest.raises(ConsistencyException):
        tester.test_output_consistency((32, 11))


def test_suite_default_config_runs_on_cpu(components):
    # regression: the default config required a GPU
    components["device"] = torch.device("cpu")
    ModelTester(**components).test_suite()


def test_suite_output_range_flag_is_honoured(components):
    # regression: test_output_range was read from the "test_nontrain_params_change" key
    tester = ModelTester(**components, test_suite_config={"test_output_range": True})
    assert tester.test_output_range
    with pytest.raises(RangeException):
        tester.test_suite()
    ModelTester(
        **components, test_suite_config={"test_output_range": True}, output_range=(-1e3, 1e3)
    ).test_suite()


def test_suite_nontrain_params(components):
    components["model"].linear.bias.requires_grad_(False)
    ModelTester(**components, test_suite_config={"test_nontrain_params_change": True}).test_suite()


@pytest.mark.parametrize("cuda_available", [True, False])
def test_suite_gpu_flag(components, monkeypatch, cuda_available):
    components["device"] = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    tester = ModelTester(**components, test_suite_config={"test_gpu_available": True})
    if cuda_available:
        tester.test_suite()
    else:
        with pytest.raises(GpuUnusedException):
            tester.test_suite()


def test_suite_rejects_unknown_keys(components):
    with pytest.raises(ValueError, match="unknown test_suite_config"):
        ModelTester(**components, test_suite_config={"test_nan": True})


def test_check_bridges_to_report(components):
    report = ModelTester(**components).check(overfit_steps=50)
    assert report["optimizer_coverage"].status is Status.PASS
    assert report.ok, str(report)


# ------------------------------------------------------------------------------------------------
# API of the 2021 develop branch


def test_output_tests(components):
    tester = ModelTester(**components, expected_output_shape=(32, 10))
    tester.test_model_output_nan()
    tester.test_model_output_inf()
    tester.test_output_consistency()
    with pytest.raises(RangeException):
        tester.test_model_output_range()
    ModelTester(**components, output_range=(-1e3, 1e3)).test_model_output_range()
    with torch.no_grad():
        components["model"].linear.bias[0] = float("nan")
    with pytest.raises(NaNTensorException, match="model output"):
        ModelTester(**components).test_model_output_nan()


def test_test_overfit(components):
    components["optimizer"] = optim.Adam(components["model"].parameters(), lr=1e-2)
    tester = ModelTester(**components, max_epochs=300)
    assert tester.test_overfit(expected_loss=0.05) < 0.05
    with pytest.raises(OverfitException, match="above the expected"):
        tester.test_overfit(expected_loss=1e-9)
    with pytest.raises(OverfitException, match="below the expected"):
        tester.test_overfit(expected_loss=10.0, overfit_relation="less")


def test_test_overfit_nan(components):
    class NanModel(MnistModel):
        def forward(self, x):
            return super().forward(x) * float("nan")

    components["model"] = NanModel()
    with pytest.raises(OverfitNaNException):
        ModelTester(**components).test_overfit()


def test_test_anomaly(components):
    ModelTester(**components).test_anomaly(iter_number=5)


@pytest.mark.filterwarnings("ignore:Error detected in")
def test_test_anomaly_nan_parameter(components):
    with torch.no_grad():
        components["model"].linear.weight[0][0] = float("nan")
    with pytest.raises(AnomalyDetectionError) as info:
        ModelTester(**components).test_anomaly(iter_number=5)
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.filterwarnings("ignore:Error detected in")
def test_test_anomaly_nan_input(components):
    components["batch"][0][0, 0, 0, 0] = float("nan")
    with pytest.raises(AnomalyDetectionError):
        ModelTester(**components).test_anomaly(iter_number=5)


def test_test_gradcheck():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 3)

        def forward(self, x):
            return self.linear(x).tanh()

    model = Model()
    batch = [torch.randn(4, 6), torch.randint(0, 3, (4,))]
    ModelTester(
        model, F.cross_entropy, optim.SGD(model.parameters(), lr=0.1), batch
    ).test_gradcheck()
    assert model.linear.weight.dtype is torch.float32


def test_suite_accepts_2021_key_names(components):
    config = {"test_params_change": True, "test_nan_values": True}  # 2021 master names
    tester = ModelTester(**components, test_suite_config=config)
    assert tester.suite_config["test_parameters_change"]
    assert tester.suite_config["test_model_output_nan"]
    assert [name for name, _ in tester.suite_tests()] == [
        "test_parameters_change",
        "test_model_output_nan",
    ]  # missing keys are disabled
    tester.test_suite()


def test_suite_runs_all_default_tests(components):
    ModelTester(**components).test_suite()


def test_2021_plugin_adapter():
    class ExamplePlugin:
        """torchtest2/plugins/example_plugin.py from the develop branch, verbatim."""

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

        def mask_outputs(self, model_outputs, mask_idx, batch_dimension):
            raise NotImplementedError  # not needed any more

    class WrongInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(30, 10)
            self.linear2 = nn.Linear(30, 10)

        def forward(self, x1, x2):
            return torch.cat((self.linear(x1), self.linear2(x1)), 1)  # x1 instead of x2!

    model = WrongInputModel()
    batch = [(torch.randn(32, 30), torch.randn(32, 30)), torch.randint(0, 20, (32,))]
    tester = ModelTester(
        model, F.cross_entropy, optim.SGD(model.parameters(), lr=0.1), batch, plugin=ExamplePlugin()
    )
    tester.test_parameters_change()
    tester.test_batch_independence()
    with pytest.raises(InputDependencyException, match=r"inputs\[1\]"):
        tester.test_input_dependency()
    assert tester.check()["input_dependency"].status is Status.FAIL
