"""check_model: one test per bug class the README promises to catch, plus side-effect tests."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtest2
from torchtest2 import CHECKS, ModelCheckError, Status, check_model


class MLP(nn.Module):
    def __init__(self, n_in: int = 20, n_hidden: int = 32, n_out: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(16, 20), torch.randint(0, 5, (16,))


def run(model, data, **kwargs):
    x, y = data
    kwargs.setdefault("loss_fn", F.cross_entropy)
    kwargs.setdefault("optimizer", torch.optim.Adam(model.parameters(), lr=1e-2))
    return check_model(model, x, y, **kwargs)


def statuses(report):
    return {r.name: r.status for r in report}


# ------------------------------------------------------------------------------------------------
# healthy models


def test_healthy_model_passes(data):
    report = run(MLP(), data)
    assert report.ok, str(report)
    assert not report.warnings, str(report)
    assert [r.name for r in report] == list(CHECKS)
    skipped = {r.name for r in report.skipped}
    assert skipped == {"output_shape", "output_range", "frozen_params", "gradcheck", "gpu"}


def test_healthy_model_without_loss_and_optimizer(data):
    report = check_model(MLP(), data[0])
    assert report.ok, str(report)
    assert report["loss"].status is Status.SKIP
    assert report["optimizer_coverage"].status is Status.SKIP
    assert report["overfit_batch"].status is Status.SKIP
    assert report["gradient_flow"].status is Status.PASS
    assert report.info["objective"] == "surrogate"


def test_softmax_output_still_gets_gradients_with_surrogate(data):
    model = nn.Sequential(nn.Linear(20, 5), nn.Softmax(dim=-1))  # softmax(x).sum() == 1
    assert check_model(model, data[0])["gradient_flow"].status is Status.PASS


def test_conv_net_with_batchnorm_and_dropout():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )
    x, y = torch.randn(8, 3, 16, 16), torch.randint(0, 4, (8,))
    report = check_model(
        model, x, y, loss_fn=nn.CrossEntropyLoss(), output_shape=(None, 4), overfit_steps=300
    )
    assert report.ok, str(report)
    assert report["batch_independence"].status is Status.PASS  # BatchNorm is fine in eval mode


# ------------------------------------------------------------------------------------------------
# no side effects


def test_check_model_has_no_side_effects(data):
    model = MLP()
    model.fc1.weight.grad = torch.ones_like(model.fc1.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = data
    x_before, params_before = x.clone(), {n: p.clone() for n, p in model.named_parameters()}
    model.eval()
    rng_before = torch.get_rng_state()

    check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer)

    assert all(torch.equal(p, params_before[n]) for n, p in model.named_parameters())
    assert not model.training and not model.fc1.training
    assert torch.equal(model.fc1.weight.grad, torch.ones_like(model.fc1.weight))
    assert model.fc2.weight.grad is None
    assert optimizer.state_dict()["state"] == {}
    assert torch.equal(x, x_before) and not x.requires_grad
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_inplace_input_modification_does_not_leak(data):
    class Model(MLP):
        def forward(self, x):
            x.mul_(2)  # modifies its input in-place
            return super().forward(x)

    x, x_before = data[0], data[0].clone()
    check_model(Model(), x)
    assert torch.equal(x, x_before)


def test_module_with_extra_state(data):
    class Model(MLP):
        def get_extra_state(self):
            return {"calls": 3}

        def set_extra_state(self, state):
            self.restored = state

    assert check_model(Model(), data[0]).ok


def test_batchnorm_running_stats_are_restored():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    before = model[1].running_mean.clone()
    check_model(model, torch.randn(8, 4) * 10 + 3)
    assert torch.equal(model[1].running_mean, before)


def test_results_are_reproducible(data):
    a = run(MLP(), data).to_dict()
    torch.manual_seed(123)
    b = run(MLP(), data).to_dict()  # different init, same verdicts
    assert [r["status"] for r in a["results"]] == [r["status"] for r in b["results"]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_model_with_cpu_inputs(data):
    model = MLP().cuda()
    report = run(model, data, require_gpu=True)
    assert report.ok, str(report)
    assert report["gpu"].status is Status.PASS
    assert report.info["device"].startswith("cuda")


def test_require_gpu_fails_without_cuda(data, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # simulate a CPU-only machine
    result = run(MLP(), data, require_gpu=True)["gpu"]
    assert result.status is Status.FAIL
    assert result.message == "CUDA is not available"


def test_require_gpu_fails_for_cpu_model(data, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # a GPU exists, but is unused
    result = run(MLP(), data, require_gpu=True)["gpu"]
    assert result.status is Status.FAIL
    assert result.message == "CUDA is available but the model is on the CPU"


# ------------------------------------------------------------------------------------------------
# bug catalogue


def test_layer_not_used_in_forward(data):
    class Model(MLP):
        def __init__(self):
            super().__init__()
            self.extra = nn.Linear(32, 32)  # defined but forgotten in forward()

    report = run(Model(), data)
    assert report["gradient_flow"].status is Status.FAIL
    assert report["gradient_flow"].details["no_grad"] == ["extra.weight", "extra.bias"]
    assert report["params_change"].status is Status.FAIL


def test_optimizer_created_before_new_head(data):
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.fc2 = nn.Linear(32, 5)  # e.g. replacing the classifier for fine-tuning
    report = run(model, data, optimizer=optimizer)
    assert report["optimizer_coverage"].status is Status.FAIL
    assert set(report["optimizer_coverage"].details["missing"]) == {"fc2.weight", "fc2.bias"}
    assert report["params_change"].status is Status.FAIL


def test_optimizer_of_another_model_instance(data):
    optimizer = torch.optim.SGD(MLP().parameters(), lr=0.1)
    report = run(MLP(), data, optimizer=optimizer)
    assert report["optimizer_coverage"].status is Status.FAIL
    assert "another model instance" in report["optimizer_coverage"].message


def test_detached_graph(data):
    class Model(MLP):
        def forward(self, x):
            h = F.relu(self.fc1(x)).detach()  # cuts the graph: fc1 never trains
            return self.fc2(h)

    report = run(Model(), data)
    assert report["gradient_flow"].status is Status.FAIL
    assert report["gradient_flow"].details["no_grad"] == ["fc1.weight", "fc1.bias"]


def test_loss_computed_through_item(data):
    def loss_fn(out, y):
        return torch.tensor(F.cross_entropy(out, y).item())

    report = run(MLP(), data, loss_fn=loss_fn)
    assert report["loss"].status is Status.FAIL
    assert "not connected to the autograd graph" in report["loss"].message
    assert report["overfit_batch"].status is Status.SKIP


def test_loss_not_reduced(data):
    report = run(MLP(), data, loss_fn=nn.CrossEntropyLoss(reduction="none"))
    assert report["loss"].status is Status.FAIL
    assert "scalar" in report["loss"].message


@pytest.mark.filterwarnings("ignore:Using a target size")
def test_loss_fn_that_raises(data):
    report = run(MLP(), data, loss_fn=F.mse_loss)  # shape/dtype mismatch with class targets
    assert report["loss"].status is Status.FAIL
    assert report["gradient_flow"].status is Status.PASS  # falls back to the surrogate


def test_nan_output(data):
    class Model(MLP):
        def forward(self, x):
            return torch.log(super().forward(x))  # log of negative numbers

    report = run(Model(), data)
    assert report["output_finite"].status is Status.FAIL
    assert report["loss"].status is Status.FAIL


def test_nan_gradient():
    class PairwiseDistance(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            h = self.fc(x)
            # d/dx sqrt(x) is inf at 0 - the zero diagonal produces NaN gradients
            return (h.unsqueeze(1) - h.unsqueeze(0)).pow(2).sum(-1).sqrt()

    report = check_model(PairwiseDistance(), torch.randn(8, 4))
    assert report["output_finite"].status is Status.PASS  # the forward pass looks fine...
    assert report["gradient_magnitude"].status is Status.FAIL  # ...the backward pass is not


def test_exploding_initialisation(data):
    model = MLP()
    with torch.no_grad():
        model.fc2.weight.mul_(1e4)  # the last layer initialised far too large
    report = run(model, data)
    assert report["initial_loss"].status is Status.WARN
    assert report["gradient_magnitude"].status is Status.WARN


def test_lstm_batch_first_mixup():
    x = torch.randn(6, 10, 8)  # (batch, time, features)
    bad = check_model(nn.LSTM(8, 16), x)  # batch_first=False treats dim 0 as time
    good = check_model(nn.LSTM(8, 16, batch_first=True), x)
    assert bad["batch_independence"].status is Status.FAIL
    assert good["batch_independence"].status is Status.PASS


def test_view_instead_of_transpose():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):  # x: (batch, time, features)
            b, t, f = x.shape
            x = x.view(t, b, f)  # BUG: should be x.transpose(0, 1)
            x = self.proj(x).cumsum(dim=0)  # e.g. a recurrence over time
            return x.view(b, t, f)

    report = check_model(Model(), torch.randn(4, 5, 8))
    assert report["batch_independence"].status is Status.FAIL


def test_softmax_over_batch_dimension(data):
    class Model(MLP):
        def forward(self, x):
            return F.softmax(super().forward(x), dim=0)  # BUG: dim=0 is the batch

    assert check_model(Model(), data[0])["batch_independence"].status is Status.FAIL


def test_unused_input_features():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(30, 10)
            self.b = nn.Linear(30, 10)

        def forward(self, x):
            x1, x2 = torch.split(x, 30, dim=-1)
            return torch.cat([self.a(x1), self.b(x1)], dim=-1)  # BUG: x1 instead of x2

    report = check_model(Model(), torch.randn(16, 60))
    result = report["input_dependency"]
    assert result.status is Status.FAIL
    assert "30/60 features of inputs" in result.message


def test_unused_input_argument():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, image, metadata):
            return self.fc(image)  # metadata silently ignored

    report = check_model(Model(), (torch.randn(8, 4), torch.randn(8, 3)))
    assert report["input_dependency"].status is Status.FAIL
    assert "inputs[1] has no influence" in report["input_dependency"].message


def test_dict_inputs_and_integer_tokens():
    class TextModel(nn.Module):
        def __init__(self, use_mask: bool):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.head = nn.Linear(16, 3)
            self.use_mask = use_mask

        def forward(self, input_ids, attention_mask):
            h = self.emb(input_ids)
            if self.use_mask:
                m = attention_mask.unsqueeze(-1).float()
                h = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                h = h.mean(1)  # BUG: padding is averaged in
            return self.head(h)

    torch.manual_seed(0)
    ids = torch.randint(1, 100, (8, 12))
    mask = torch.ones(8, 12, dtype=torch.long)
    mask[::2, 6:] = 0
    inputs = {"input_ids": ids, "attention_mask": mask}
    y = torch.randint(0, 3, (8,))

    good = check_model(TextModel(True), inputs, y, loss_fn=F.cross_entropy)
    assert good.ok, str(good)
    bad = check_model(TextModel(False), inputs, y, loss_fn=F.cross_entropy)
    assert bad["input_dependency"].status is Status.FAIL
    assert "['attention_mask'] has no influence" in bad["input_dependency"].message


def test_huggingface_style_output_with_loss():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x, labels=None):
            logits = self.fc(x)
            out = {"logits": logits}
            if labels is not None:
                out["loss"] = F.cross_entropy(logits, labels)
            return out

    inputs = {"x": torch.randn(8, 10), "labels": torch.randint(0, 3, (8,))}
    report = check_model(Model(), inputs, optimizer=None)
    assert report.ok, str(report)
    assert report["loss"].status is Status.PASS
    assert report.info["objective"] == "output.loss"


def test_frozen_backbone(data):
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    report = run(model, data, optimizer=optimizer, frozen=["fc1"])
    assert report["frozen_params"].status is Status.FAIL  # forgot requires_grad_(False)
    assert report["frozen_params"].details["changed"] == ["fc1.weight", "fc1.bias"]

    model.fc1.requires_grad_(False)
    report = run(model, data, optimizer=optimizer, frozen=["fc1"])
    assert report["frozen_params"].status is Status.PASS
    assert report["optimizer_coverage"].status is Status.PASS
    assert report.ok, str(report)


def test_frozen_but_modified_in_forward(data):
    class Model(MLP):
        def forward(self, x):
            with torch.no_grad():
                self.fc1.weight.mul_(0.99)  # e.g. a hand-rolled EMA gone wrong
            return super().forward(x)

    model = Model()
    model.fc1.requires_grad_(False)
    assert run(model, data)["frozen_params"].status is Status.FAIL


def test_dropout_active_in_eval(data):
    class Model(MLP):
        def forward(self, x):
            return self.fc2(F.dropout(F.relu(self.fc1(x)), p=0.5))  # training=True by default

    report = run(Model(), data)
    assert report["eval_determinism"].status is Status.WARN
    assert report.ok, str(report)  # other checks must not raise false alarms because of it


def test_constant_output(data):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc(torch.zeros_like(x))  # input wiped out by mistake

    report = run(Model(), data)
    assert report["output_varies"].status is Status.WARN
    assert report["input_dependency"].status is Status.FAIL


def test_zero_gradient_warns():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.gate = nn.Linear(4, 4)

        def forward(self, x):
            return self.fc(x) + 0 * self.gate(x)

    assert check_model(Model(), torch.randn(8, 4))["gradient_flow"].status is Status.WARN


def test_learning_rate_zero(data):
    model = MLP()
    report = run(model, data, optimizer=torch.optim.SGD(model.parameters(), lr=0.0))
    assert report["params_change"].status is Status.FAIL
    assert report["overfit_batch"].status is Status.FAIL


def test_overfit_not_reaching_target(data):
    model = MLP()
    report = run(
        model, data, optimizer=torch.optim.SGD(model.parameters(), lr=1e-3), overfit_steps=20
    )
    assert report["overfit_batch"].status is Status.WARN


def test_overfit_diverges(data):
    model = MLP()
    report = run(model, data, optimizer=torch.optim.SGD(model.parameters(), lr=1e12))
    assert report["overfit_batch"].status is Status.FAIL
    assert "became" in report["overfit_batch"].message


def test_forward_crash_skips_everything_else(data):
    report = run(MLP(n_in=21), data)  # shape mismatch
    assert report["forward"].status is Status.FAIL
    assert "mat1 and mat2" in report["forward"].message
    others = {r.status for r in report if r.name not in ("forward", "optimizer_coverage")}
    assert others == {Status.SKIP}


def test_output_shape_and_range(data):
    assert run(MLP(), data, output_shape=(16, 5))["output_shape"].status is Status.PASS
    assert run(MLP(), data, output_shape=(None, 5))["output_shape"].status is Status.PASS
    assert run(MLP(), data, output_shape=(-1, 4))["output_shape"].status is Status.FAIL
    sigmoid = nn.Sequential(MLP(), nn.Sigmoid())
    assert run(sigmoid, data, output_range=(0, 1))["output_range"].status is Status.PASS
    assert run(MLP(), data, output_range=(0, 1))["output_range"].status is Status.FAIL
    assert run(MLP(), data, output_range=(None, 1e6))["output_range"].status is Status.PASS


def test_model_without_trainable_parameters(data):
    model = MLP().requires_grad_(False)
    report = check_model(model, data[0])
    assert report.ok, str(report)
    assert report["gradient_flow"].status is Status.SKIP
    assert report["frozen_params"].status is Status.PASS


def test_batch_of_one_skips_batch_checks(data):
    report = run(MLP(), (data[0][:1], data[1][:1]))
    assert report["batch_independence"].status is Status.SKIP
    assert report["output_varies"].status is Status.SKIP


# ------------------------------------------------------------------------------------------------
# API


def test_skip_and_validation(data):
    report = run(MLP(), data, skip=["overfit_batch", "loss"])
    assert report["overfit_batch"].message == "skipped by the caller"
    assert report["loss"].status is Status.SKIP
    with pytest.raises(ValueError, match="unknown check"):
        run(MLP(), data, skip=["gradient_flw"])
    with pytest.raises(ValueError, match="cannot be skipped"):
        run(MLP(), data, skip=["forward"])


def test_raise_on_failure_and_assert_model_ok(data):
    class Model(MLP):
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)).detach())

    with pytest.raises(ModelCheckError, match="gradient_flow") as info:
        run(Model(), data).raise_on_failure()
    assert isinstance(info.value, AssertionError)
    assert info.value.report["gradient_flow"].status is Status.FAIL

    torchtest2.assert_model_ok(MLP(), *data, loss_fn=F.cross_entropy)


def test_strict_mode_fails_on_warnings(data):
    class Model(MLP):
        def forward(self, x):
            return self.fc2(F.dropout(F.relu(self.fc1(x)), p=0.5))

    report = run(Model(), data)
    assert report.ok
    report.raise_on_failure()
    with pytest.raises(ModelCheckError, match="eval_determinism"):
        report.raise_on_failure(strict=True)


def test_report_formatting(data, capsys):
    report = run(MLP(), data, verbose=True)
    printed = capsys.readouterr().out
    assert printed.startswith("torchtest2 report for MLP")
    assert "PASS  gradient_flow" in printed
    assert report.summary() in printed
    assert repr(report).startswith("<Report MLP:")
    as_dict = report.to_dict()
    assert as_dict["ok"] is True
    assert {r["name"] for r in as_dict["results"]} == set(CHECKS)
    assert "gradient_flow" in report and "nope" not in report
    with pytest.raises(KeyError):
        report["nope"]


# ------------------------------------------------------------------------------------------------
# features ported from the 2021 develop branch


class _WrongSquare(torch.autograd.Function):
    """x**2 with a wrong backward (should be 2 * x * grad)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return x * grad


class _Square(_WrongSquare):
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return 2 * x * grad


def _custom_op_model(op):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 3)

        def forward(self, x):
            return self.fc(op.apply(x))

    return Model()


def test_gradcheck(data):
    x = torch.randn(4, 6)
    good = _custom_op_model(_Square)
    assert check_model(good, x, gradcheck=True)["gradcheck"].status is Status.PASS
    bad = _custom_op_model(_WrongSquare)
    assert check_model(bad, x, gradcheck=True)["gradcheck"].status is Status.FAIL
    assert bad.fc.weight.dtype is torch.float32  # runs on a float64 copy
    assert check_model(bad, x)["gradcheck"].status is Status.SKIP  # opt-in


class _SqrtGate(nn.Module):
    """sqrt inside torch.where: finite forward, NaN backward once the parameter turns negative."""

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        gate = torch.where(self.s > 0, torch.sqrt(self.s), torch.zeros_like(self.s))
        return gate * torch.ones(x.shape[0], 1)


@pytest.mark.filterwarnings("ignore:Error detected in SqrtBackward0")
def test_overfit_detects_nan_gradients_during_training():
    model = _SqrtGate()
    kwargs = dict(
        loss_fn=lambda out: out.mean(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.3),
        overfit_target=-1.0,  # unreachable: keep training until something breaks
        skip=["gradient_magnitude"],
    )
    x = torch.randn(4, 2)
    assert check_model(model, x, **kwargs)["gradient_magnitude"].status is Status.SKIP
    result = check_model(model, x, **kwargs)["overfit_batch"]
    assert result.status is Status.FAIL
    assert result.details["step"] > 0  # step 0 was fine
    assert "gradients became NaN/inf" in result.message

    result = check_model(model, x, detect_anomaly=True, **kwargs)["overfit_batch"]
    assert result.status is Status.FAIL
    assert "SqrtBackward" in result.message


def test_overfit_target(data):
    report = run(MLP(), data, overfit_target=0.5)
    assert report["overfit_batch"].status is Status.PASS
    assert report["overfit_batch"].details["losses"][-1] <= 0.5
    report = run(MLP(), data, overfit_target=-1.0, overfit_steps=20)
    assert report["overfit_batch"].status is Status.WARN
    assert "target" in report["overfit_batch"].message


def test_forward_fn_for_models_returning_losses_in_train_mode(data):
    class Detector(nn.Module):
        """torchvision-detection-like: train mode takes targets and returns a dict of losses."""

        def __init__(self, use_box_loss: bool = True):
            super().__init__()
            self.backbone = nn.Linear(20, 16)
            self.cls = nn.Linear(16, 5)
            self.box = nn.Linear(16, 4)
            self.use_box_loss = use_box_loss

        def forward(self, x, targets=None):
            h = F.relu(self.backbone(x))
            if self.training:
                losses = {"cls": F.cross_entropy(self.cls(h), targets)}
                if self.use_box_loss:
                    losses["box"] = self.box(h).pow(2).mean()
                return losses
            return {"scores": self.cls(h).softmax(-1), "boxes": self.box(h)}

    kwargs = dict(
        forward_fn=lambda model, x, y: model(x, y),
        loss_fn=lambda losses, y: sum(losses.values()),
    )
    report = check_model(Detector(), *data, **kwargs)
    assert report.ok, str(report)
    report = check_model(Detector(use_box_loss=False), *data, **kwargs)
    assert report["gradient_flow"].details["no_grad"] == ["box.weight", "box.bias"]


def test_eval_forward_fn(data):
    model = MLP()
    report = check_model(model, data[0], eval_forward_fn=lambda m, x: m(x).softmax(-1))
    assert report["output_range"].status is Status.SKIP
    report = check_model(
        model, data[0], eval_forward_fn=lambda m, x: m(x).softmax(-1), output_range=(0, 1)
    )
    assert report["output_range"].status is Status.PASS


def test_batch_dim_for_sequence_first_tensors():
    x = torch.randn(10, 4, 8)  # (time, batch, features)
    good = check_model(nn.LSTM(8, 16), x, batch_dim=1)
    assert good.info["batch size"] == 4
    assert good["batch_independence"].status is Status.PASS
    assert good["input_dependency"].status is Status.PASS
    bad = check_model(nn.LSTM(8, 16, batch_first=True), x, batch_dim=1)
    assert bad["batch_independence"].status is Status.FAIL


# ------------------------------------------------------------------------------------------------
# input/output structures promised by the README


def test_namedtuple_and_dataclass_outputs(data):
    from collections import namedtuple
    from dataclasses import dataclass

    Out = namedtuple("Out", "logits aux")

    @dataclass
    class DataOut:
        logits: torch.Tensor
        aux: torch.Tensor

    class Model(MLP):
        def __init__(self, kind):
            super().__init__()
            self.aux = nn.Linear(20, 2)
            self.kind = kind

        def forward(self, x):
            out = (super().forward(x), self.aux(x))
            return Out(*out) if self.kind == "namedtuple" else DataOut(*out)

    def loss_fn(out, y):
        return F.cross_entropy(out.logits, y) + out.aux.pow(2).mean()

    for kind in ("namedtuple", "dataclass"):
        report = run(Model(kind), data, loss_fn=loss_fn)
        assert report.ok, str(report)
        assert "output.logits: Tensor[16, 5]" in report["forward"].message
        assert report["batch_independence"].status is Status.PASS


def test_namedtuple_input_is_a_single_argument():
    from collections import namedtuple

    Pair = namedtuple("Pair", "a b")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, pair):
            return self.fc(pair.a)  # pair.b ignored

    report = check_model(Model(), Pair(torch.randn(6, 8), torch.randn(6, 8)))
    assert report["input_dependency"].status is Status.FAIL
    assert "inputs.b has no influence" in report["input_dependency"].message


def test_unusual_mapping_inputs(data):
    from collections import OrderedDict, defaultdict

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc(x)

    for mapping in (OrderedDict(x=data[0]), defaultdict(list, x=data[0])):
        assert check_model(Model(), mapping).ok


def test_seed_everything():
    import random

    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torchtest2.seed_everything(123)
    a = (random.random(), torch.rand(1))
    torchtest2.seed_everything(123, deterministic=True)
    b = (random.random(), torch.rand(1))
    try:
        assert a[0] == b[0] and torch.equal(a[1], b[1])
        assert torch.are_deterministic_algorithms_enabled()
    finally:  # global switches: restore them for the other tests
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
