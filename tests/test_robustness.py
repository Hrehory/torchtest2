"""Unusual but legitimate models and setups (found by an edge-case study): no
internal crash, no false FAIL, no side effects."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from torchtest2 import ModelTester, Status, check_model


def mlp(i=16, o=4):
    return nn.Sequential(nn.Linear(i, 32), nn.ReLU(), nn.Linear(32, o))


@pytest.fixture
def xy():
    g = torch.Generator().manual_seed(0)
    return torch.randn(8, 16, generator=g), torch.randint(0, 4, (8,), generator=g)


def fails(report):
    return {r.name for r in report if r.status is Status.FAIL}


def test_lazy_modules(xy):
    model = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.LazyLinear(4))
    rng = torch.get_rng_state()
    report = check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=20)
    assert not fails(report), str(report)
    assert "lazy modules were initialised" in report.info["notes"]
    assert torch.equal(torch.get_rng_state(), rng)


def test_complex_outputs(xy):
    class Spectral(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return torch.fft.rfft(self.fc(x), dim=-1)

    report = check_model(Spectral(), xy[0])
    assert not fails(report), str(report)
    assert report["gradient_flow"].status is Status.PASS

    class NaNSpectrum(Spectral):
        def forward(self, x):
            return super().forward(x) * float("nan")

    assert check_model(NaNSpectrum(), xy[0])["output_finite"].status is Status.FAIL


def test_model_modifying_its_input_in_place(xy):
    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = mlp()

        def forward(self, x):
            x.clamp_(-2, 2)
            return self.net(x)

    x, y = xy
    before = x.clone()
    report = check_model(Clamp(), x, y, loss_fn=F.cross_entropy, overfit_steps=20)
    assert report["input_dependency"].status is Status.PASS, str(report)
    assert torch.equal(x, before)


class Packed(nn.Module):
    def __init__(self, enforce_sorted):
        super().__init__()
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 4)
        self.enforce_sorted = enforce_sorted

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=self.enforce_sorted
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


def test_zero_padding_is_not_an_ignored_feature():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 10, 8, generator=g)
    lengths = torch.tensor([4, 9, 3, 6, 8, 5, 7, 2])
    for i, n in enumerate(lengths):
        x[i, n:] = 0  # zero padding
    report = check_model(Packed(enforce_sorted=False), (x, lengths))
    assert report["input_dependency"].status is Status.PASS, str(report)

    x[0, 9] = 1.0  # a non-zero value beyond the length of sample 0 really is ignored
    report = check_model(Packed(enforce_sorted=False), (x, lengths))
    assert report["input_dependency"].status is Status.FAIL


def test_ignored_features_are_still_found_next_to_padding():
    class HalfBlind(Packed):
        def forward(self, x, lengths):
            return super().forward(x * torch.tensor([1.0] * 4 + [0.0] * 4), lengths)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 10, 8, generator=g)
    lengths = torch.tensor([4, 9, 3, 6, 8, 5, 7, 10])
    for i, n in enumerate(lengths):
        x[i, n:] = 0
    report = check_model(HalfBlind(enforce_sorted=False), (x, lengths))
    assert report["input_dependency"].status is Status.FAIL


def test_inputs_the_model_validates_are_skipped_not_failed():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 10, 8, generator=g)
    lengths = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3])  # enforce_sorted=True needs this order
    report = check_model(Packed(enforce_sorted=True), (x, lengths))
    assert not fails(report), str(report)
    assert "could not be probed" in report["input_dependency"].message


def test_training_forward_failure_is_reported_once(xy):
    model = nn.Sequential(nn.Linear(16, 8), nn.BatchNorm1d(8), nn.Linear(8, 4))
    x, y = xy
    report = check_model(model, x[:1], y[:1], loss_fn=F.cross_entropy)
    assert fails(report) == {"loss"}
    assert "training-mode forward pass raised" in report["loss"].message
    for name in ("gradient_flow", "gradient_magnitude", "params_change", "overfit_batch"):
        assert report[name].status is Status.SKIP, name


def test_optimizer_that_needs_a_closure(xy):
    model = mlp()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5)
    report = check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=optimizer)
    assert not fails(report), str(report)
    assert report["params_change"].status is Status.PASS
    tester = ModelTester(model, F.cross_entropy, optimizer, list(xy), max_epochs=5, min_epochs=1)
    assert tester.overfit_single_batch() < 1.5
    tester.test_anomaly(iter_number=2)


def test_sparse_gradients():
    class Bag(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16, sparse=True)

        def forward(self, ids):
            return self.emb(ids).mean(1)

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 100, (8, 5), generator=g)
    model = Bag()
    optimizer = torch.optim.SparseAdam(list(model.parameters()))
    report = check_model(model, ids, optimizer=optimizer)
    assert not fails(report), str(report)
    assert report["gradient_magnitude"].status is Status.PASS


def test_rewrapped_loss_is_caught_by_the_loss_check(xy):
    def loss_fn(out, y):
        return torch.tensor(F.cross_entropy(out, y).item(), requires_grad=True)

    report = check_model(mlp(), *xy, loss_fn=loss_fn)
    assert report["loss"].status is Status.FAIL
    assert "leaf tensor" in report["loss"].message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pure_float16_training_is_flagged_with_an_explanation(xy):
    model = mlp().cuda().half()
    x, y = xy
    report = check_model(
        model, x.cuda().half(), y.cuda(), loss_fn=lambda o, t: F.cross_entropy(o.float(), t)
    )
    assert report["overfit_batch"].status is Status.FAIL
    assert "half precision" in report["overfit_batch"].message


def test_other_unusual_but_supported_setups(xy):
    x, y = xy
    tied = nn.Linear(16, 16, bias=False)
    shared = nn.Sequential(tied, nn.ReLU(), tied, nn.Linear(16, 4))  # the same layer twice
    ema = copy.deepcopy(shared).requires_grad_(False)
    cases = {
        "compile": torch.compile(mlp(), backend="eager"),
        "shared_layer": shared,
        "frozen_copy": nn.ModuleDict({"student": mlp(), "teacher": ema}),
        "float64": mlp().double(),
    }
    for name, model in cases.items():
        inputs = x.double() if name == "float64" else x
        if name == "frozen_copy":
            model.forward = lambda t, m=model: (
                m["student"](t) + 0 * m["teacher"](torch.zeros(t.shape[0], 16))[:, :4].sum()
            )
        report = check_model(model, inputs, y, loss_fn=F.cross_entropy, overfit_steps=20)
        assert not fails(report), f"{name}: {report}"


def test_zero_gradient_is_a_warning_not_an_optimizer_failure(xy):
    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
            self.gate = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc(x) + 0 * self.gate(x)  # the gate's gradient is exactly zero

    model = Gated()
    report = check_model(
        model, *xy, loss_fn=F.cross_entropy, optimizer=torch.optim.Adam(model.parameters())
    )
    assert report["gradient_flow"].status is Status.WARN
    assert report["params_change"].status is Status.WARN
    assert report["params_change"].details["zero_gradient"] == ["gate.weight", "gate.bias"]
    # a parameter with a real gradient that does not change is still an optimizer failure
    optimizer = torch.optim.Adam(model.fc.parameters())
    report = check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=optimizer)
    assert report["params_change"].status is Status.WARN  # only the zero-gradient gate is unchanged
    frozen_lr = torch.optim.SGD(model.parameters(), lr=0.0)
    assert (
        check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=frozen_lr)[
            "params_change"
        ].status
        is Status.FAIL
    )


def test_modules_that_version_their_state_dict(xy):
    class Versioned(nn.Module):  # like torchvision's MNASNet
        _version = 2

        def __init__(self):
            super().__init__()
            self.net = mlp()

        def forward(self, x):
            return self.net(x)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
            if local_metadata.get("version") not in (1, 2):
                raise ValueError("version should be set to 1 or 2")
            super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)

    report = check_model(Versioned(), *xy, loss_fn=F.cross_entropy, overfit_steps=10)
    assert not fails(report), str(report)


def test_zero_initialised_head(xy):
    model = mlp()
    nn.init.zeros_(model[2].weight)  # like torchvision's ViT head
    report = check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=10)
    assert not fails(report), str(report)
    assert "after one optimizer step" in report["input_dependency"].message
    assert "after one optimizer step" in report["output_varies"].message
    # zero gradients behind the zero-initialised head vanish after the first update
    assert report["gradient_flow"].status is Status.PASS, str(report)
    assert report["gradient_flow"].details["zero_at_init"] == ["0.weight", "0.bias"]
    assert report["params_change"].status is Status.PASS


def test_eval_outputs_that_are_piecewise_constant():
    class Detector(nn.Module):  # eval: thresholded outputs; train: a loss of the input
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            scores = self.fc(x)
            return scores.sum() if self.training else (scores > 1e6).float()

    x = torch.randn(8, 16)
    report = check_model(Detector(), x, loss_fn=lambda out: out, batch_dim=None)
    assert report["input_dependency"].status is Status.PASS, str(report)
    assert "training objective" in report["input_dependency"].message


def test_border_pixels_missed_by_a_stride_are_a_warning():
    conv = nn.Sequential(nn.Conv2d(3, 4, 3, stride=2), nn.Flatten())  # never sees column/row 7
    report = check_model(conv, torch.randn(4, 3, 8, 8))
    assert report["input_dependency"].status is Status.WARN
    assert "trailing border" in report["input_dependency"].message


def test_an_ignored_last_time_step_is_a_failure():
    class OffByOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(8, 16, batch_first=True)

        def forward(self, x):
            return self.rnn(x[:, :-1])[0].mean(1)  # drops the last time step

    report = check_model(OffByOne(), torch.randn(4, 10, 8))
    assert report["input_dependency"].status is Status.FAIL


def test_weak_batch_coupling_is_a_warning():
    class Weak(nn.Module):
        def __init__(self, strength):
            super().__init__()
            self.fc = nn.Linear(16, 4)
            self.strength = strength

        def forward(self, x):
            out = self.fc(x)
            return out + self.strength * out.roll(1, 0)

    x = torch.randn(8, 16)
    assert check_model(Weak(0.5), x)["batch_independence"].status is Status.FAIL
    assert check_model(Weak(1e-5), x)["batch_independence"].status is Status.WARN
    assert check_model(Weak(0.0), x)["batch_independence"].status is Status.PASS


def test_frozen_names_of_tied_parameters_and_typos(xy):
    class Tied(nn.Module):  # like T5/BART: shared embeddings, listed under their first name
        def __init__(self):
            super().__init__()
            self.shared = nn.Embedding(50, 16)
            self.encoder_embed = self.shared
            self.head = nn.Linear(16, 4)

        def forward(self, ids):
            return self.head(self.encoder_embed(ids).mean(1))

    ids = torch.randint(0, 50, (8, 6))
    y = torch.randint(0, 4, (8,))
    report = check_model(Tied(), ids, y, loss_fn=F.cross_entropy, frozen=["encoder_embed"])
    assert report["frozen_params"].status is Status.FAIL  # the alias is recognised
    report = check_model(Tied(), ids, y, loss_fn=F.cross_entropy, frozen=["encoder_embd"])
    assert report["frozen_params"].status is Status.WARN
    assert "match(es) no parameter" in report["frozen_params"].message


def test_vanishing_gradient_is_not_an_optimizer_failure(xy):
    class LayerScale(nn.Module):  # a branch behind a tiny layer scale (EdgeNeXt, CaiT, ...)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
            self.branch = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc(x) + 1e-30 * self.branch(x)

    report = check_model(LayerScale(), *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    assert report["params_change"].status is Status.WARN, str(report)
    assert "vanishing" in report["params_change"].message


def test_batch_coupling_behind_constant_initial_outputs(xy):
    model = mlp()
    nn.init.zeros_(model[2].weight)  # outputs identical for every sample at initialisation
    coupled = nn.Sequential(model, nn.Softmax(dim=0))  # BUG: softmax over the batch
    report = check_model(coupled, xy[0], overfit_steps=0)
    assert report["batch_independence"].status is Status.FAIL, str(report)
    report = check_model(nn.Sequential(model, nn.Softmax(dim=1)), xy[0], overfit_steps=0)
    assert report["batch_independence"].status is Status.PASS
    assert "after one optimizer step" in report["batch_independence"].message


def test_patch_embedding_border_is_a_warning():
    # 14x14 patches of a 64x64 image never cover the last 8 rows/columns (as in DINOv2, ViTs)
    model = nn.Sequential(nn.Conv2d(3, 8, 14, stride=14), nn.Flatten(), nn.Linear(8 * 16, 4))
    x = torch.randn(4, 3, 64, 64)
    report = check_model(model, x, torch.randint(0, 4, (4,)), loss_fn=F.cross_entropy)
    assert report["input_dependency"].status is Status.WARN, str(report)


def test_layerdrop_is_not_an_unused_parameter():
    class LayerDrop(nn.Module):  # skips its second layer in some training passes
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(16, 16), nn.Linear(16, 16)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            x = torch.relu(self.a(x))
            if not self.training or torch.rand(()) > 0.5:
                x = x + torch.relu(self.b(x))
            return self.head(x)

    x, y = torch.randn(8, 16), torch.randint(0, 4, (8,))
    for seed in range(4):  # whichever pass skips the layer
        report = check_model(LayerDrop(), x, y, loss_fn=F.cross_entropy, seed=seed)
        assert report["gradient_flow"].status is not Status.FAIL, str(report)
        assert report["params_change"].status is not Status.FAIL, str(report)


def test_unused_layer_is_still_reported_with_layerdrop_recheck(xy):
    class Unused(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.unused = nn.Linear(16, 4), nn.Linear(16, 4)

        def forward(self, x):
            return self.fc(x)

    report = check_model(Unused(), *xy, loss_fn=F.cross_entropy)
    assert report["gradient_flow"].status is Status.FAIL
    assert report["params_change"].status is Status.FAIL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tf32_rounding_is_not_a_batch_coupling():
    torch.manual_seed(0)
    model = nn.Sequential(
        *[m for _ in range(6) for m in (nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 4),
    ).cuda()
    model.insert(0, nn.Conv2d(3, 32, 3, padding=1).cuda())
    x = torch.randn(2, 3, 64, 64, device="cuda")
    report = check_model(model, x, overfit_steps=0)
    assert report["batch_independence"].status is not Status.FAIL, str(report)
