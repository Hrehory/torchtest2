"""The side-effect contract of check_model and the structural comparisons of its checks: what is
restored, what is documented as not restored, and outputs whose structure changes."""

import dataclasses
import random
import threading
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtest2 import Status, check_model
from torchtest2._tree import structure_difference, to_device, tree_leaves, tree_map
from torchtest2.exceptions import InvarianceException
from torchtest2.plugins import get_plugin, plugin_discovery
from torchtest2.post_train import assert_invariance


def mlp(i=16, o=4):
    return nn.Sequential(nn.Linear(i, 32), nn.ReLU(), nn.Linear(32, o))


@pytest.fixture
def xy():
    g = torch.Generator().manual_seed(0)
    return torch.randn(8, 16, generator=g), torch.randint(0, 4, (8,), generator=g)


# ------------------------------------------------------------------------------------------------
# dataclass inputs are copied


@dataclass
class Batch:
    x: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class FrozenBatch:
    x: torch.Tensor
    scale: float = 1.0
    cache: torch.Tensor = field(init=False, default_factory=lambda: torch.zeros(3))


@dataclass
class SubBatch(Batch):
    extra: list = field(default_factory=list)


def test_tree_map_rebuilds_dataclasses():
    frozen = FrozenBatch(torch.ones(2, 3), 2.0)
    copied = tree_map(lambda _, t: t.clone(), frozen)
    assert type(copied) is FrozenBatch and copied.scale == 2.0
    assert copied.x is not frozen.x and torch.equal(copied.x, frozen.x)
    assert copied.cache is not frozen.cache  # a field with init=False is mapped too
    nested = SubBatch(torch.ones(2), torch.zeros(2), [torch.ones(1), {"a": torch.ones(1)}])
    copied = to_device(nested, torch.device("cpu"))
    assert type(copied) is SubBatch
    original_ids = {id(t) for _, t in tree_leaves(nested)}
    assert [p for p, _ in tree_leaves(copied)] == [p for p, _ in tree_leaves(nested)]
    assert not original_ids & {id(t) for _, t in tree_leaves(copied)}


def test_dataclass_inputs_modified_in_place_do_not_leak():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, batch):
            batch.x.add_(1)  # in-place modification of both tensors of the input
            batch.mask.mul_(0)
            return self.fc(batch.x * batch.mask.unsqueeze(-1))

    batch = Batch(torch.randn(8, 16), torch.ones(8))
    before = Batch(batch.x.clone(), batch.mask.clone())
    check_model(Model(), batch, overfit_steps=5)
    assert torch.equal(batch.x, before.x) and torch.equal(batch.mask, before.mask)


# ------------------------------------------------------------------------------------------------
# lazy modules, RNGs and Python attributes


def test_lazy_modules_stay_uninitialised(xy):
    model = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.LazyLinear(4))
    report = check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=20)
    assert report.ok, str(report)
    assert "on a copy" in report.info["notes"]
    assert report.info["parameters"] == 16 * 32 + 32 + 32 * 4 + 4
    assert isinstance(model[0], nn.LazyLinear)
    assert isinstance(model[0].weight, nn.parameter.UninitializedParameter)
    model(xy[0])  # the caller can still initialise it
    assert model[0].weight.shape == (32, 16)


def _lazy_mlp():
    return nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.LazyLinear(4))


def _uninitialised(model):
    return all(isinstance(p, nn.parameter.UninitializedParameter) for p in model.parameters())


class _ScaledSGD(torch.optim.SGD):
    """A custom optimizer with an extra attribute."""

    def __init__(self, params, lr, scale=1.0):
        super().__init__(params, lr=lr)
        self.scale = scale

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:  # uses the extra attribute: it must survive the copy
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-group["lr"] * self.scale)


@pytest.mark.parametrize(
    "make_optimizer",
    [
        pytest.param(lambda m: torch.optim.SGD(m.parameters(), lr=0.1), id="sgd"),
        pytest.param(lambda m: torch.optim.Adam(m.parameters(), lr=1e-2), id="adam"),
        pytest.param(
            lambda m: torch.optim.SGD(
                [{"params": m[0].parameters()}, {"params": m[2].parameters(), "lr": 0.01}],
                lr=0.1,
                momentum=0.9,
            ),
            id="param-groups",
        ),
        pytest.param(lambda m: torch.optim.LBFGS(m.parameters(), lr=0.1), id="closure"),
        pytest.param(lambda m: _ScaledSGD(m.parameters(), lr=0.1, scale=2.0), id="custom"),
    ],
)
def test_lazy_modules_with_an_optimizer_stay_uninitialised(xy, make_optimizer):
    model = _lazy_mlp()
    optimizer = make_optimizer(model)
    groups = [list(g["params"]) for g in optimizer.param_groups]
    report = check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=optimizer, overfit_steps=20)
    assert report.ok, str(report)
    # the model and the optimizer were copied together: the copy's optimizer trained the copy
    assert report["optimizer_coverage"].status is Status.PASS, str(report)
    assert report["params_change"].status is Status.PASS, str(report)
    assert "on a copy" in report.info["notes"]
    assert _uninitialised(model) and isinstance(model[0], nn.LazyLinear)
    assert [list(g["params"]) for g in optimizer.param_groups] == groups
    assert all(
        p is q for g, h in zip(optimizer.param_groups, groups) for p, q in zip(g["params"], h)
    )
    assert optimizer.state_dict()["state"] == {}
    model(xy[0])  # the caller can still initialise both and train
    F.cross_entropy(model(xy[0]), xy[1]).backward()
    if not isinstance(optimizer, torch.optim.LBFGS):
        optimizer.step()


def test_lazy_modules_with_tied_parameters(xy):
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = nn.LazyLinear(16)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            return self.head(torch.relu(self.encode(x)))

    model = Tied()
    model.head2 = model.head  # the same module under two names
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report = check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=optimizer, overfit_steps=5)
    assert report.ok, str(report)
    assert isinstance(model.encode.weight, nn.parameter.UninitializedParameter)


def test_lazy_model_with_an_uncopyable_optimizer_is_rejected(xy):
    class Uncopyable(torch.optim.SGD):
        def __init__(self, params, lr):
            super().__init__(params, lr=lr)
            self.lock = threading.Lock()  # an attribute that cannot be deep-copied

    model = _lazy_mlp()
    optimizer = Uncopyable(model.parameters(), lr=0.1)
    with pytest.raises(TypeError, match="lazy modules and the optimizer cannot be copied"):
        check_model(model, *xy, loss_fn=F.cross_entropy, optimizer=optimizer)
    assert _uninitialised(model)


def test_rngs_are_restored(xy):
    class Noisy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            noise = random.random()  # Python global RNG used inside forward
            return self.fc(x) + 0 * noise + 0 * torch.rand(1, device=x.device)

    model = Noisy()
    random.seed(123)
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_numpy_rng_is_restored(xy):
    np = pytest.importorskip("numpy")

    class Noisy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc(x) * float(1 + 0 * np.random.rand())

    np.random.seed(7)
    state = np.random.get_state()
    check_model(Noisy(), *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    after = np.random.get_state()
    assert state[0] == after[0] and (state[1] == after[1]).all() and state[2:] == after[2:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_rng_is_restored(xy):
    state = torch.cuda.get_rng_state()
    model = nn.Sequential(nn.Linear(16, 32), nn.Dropout(0.5), nn.Linear(32, 4)).cuda()
    check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_python_rng_in_forward_is_not_a_false_alarm(xy):
    class RandomScale(nn.Module):
        """A sample-wise random augmentation drawn with the Python RNG in eval mode."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            scales = torch.tensor([random.uniform(0.5, 1.5) for _ in range(len(x))])
            return self.fc(x) * scales[:, None]

    report = check_model(RandomScale(), *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    # the compared passes see the same Python random numbers: no batch coupling is reported
    assert report["batch_independence"].status is Status.PASS
    assert report["eval_determinism"].status is Status.WARN  # it is random in eval mode


class _CountingBuffer(nn.Module):
    """Counts its forward passes in a non-persistent buffer (not in the state dict)."""

    def __init__(self, fail=False, replace=False):
        super().__init__()
        self.fc = nn.Linear(16, 4)
        self.register_buffer("counter", torch.zeros(()), persistent=False)
        self.fail, self.replace = fail, replace

    def forward(self, x):
        if self.replace:
            self.counter = self.counter + 1  # assigns a new tensor to the registered buffer
        else:
            self.counter.add_(1)
        if self.fail:
            raise RuntimeError("forward failed after changing the buffer")
        return self.fc(x)


@pytest.mark.parametrize("replace", [False, True], ids=["in-place", "reassigned"])
def test_non_persistent_buffers_are_restored(xy, replace):
    from torchtest2 import CHECKS

    model = _CountingBuffer(replace=replace)
    buffer = model.counter
    assert "counter" not in model.state_dict()
    check_model(model, xy[0], skip=[c for c in CHECKS if c != "forward"])
    assert float(model.counter) == 0.0
    check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=5)  # every check
    assert float(model.counter) == 0.0
    if not replace:
        assert model.counter is buffer  # restored in place


def test_non_persistent_buffers_are_restored_when_forward_raises(xy):
    model = _CountingBuffer(fail=True)
    report = check_model(model, *xy, loss_fn=F.cross_entropy)
    assert report["forward"].status is Status.FAIL
    assert float(model.counter) == 0.0


def test_python_attributes_are_outside_the_contract(xy):
    class Counting(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.fc(x)

    model = Counting()
    check_model(model, *xy, loss_fn=F.cross_entropy, overfit_steps=5)
    assert model.calls > 0  # documented: plain Python attributes are not restored


def test_pytorch_state_is_restored(xy):
    model = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.Dropout(0.1), nn.Linear(32, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = xy
    F.cross_entropy(model(x), y).backward()
    optimizer.step()  # non-empty optimizer state and gradients
    model[2].eval()  # mixed train/eval flags
    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [p.grad.clone() for p in model.parameters()]
    optimizer_state = {
        k: {n: t.clone() for n, t in v.items()} for k, v in optimizer.state_dict()["state"].items()
    }
    flags = [m.training for m in model.modules()]
    x_before = x.clone()

    check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer, overfit_steps=10)

    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    after = optimizer.state_dict()["state"]
    assert all(
        torch.equal(after[k][n], t) for k, v in optimizer_state.items() for n, t in v.items()
    )
    assert [m.training for m in model.modules()] == flags
    assert torch.equal(x, x_before)


# ------------------------------------------------------------------------------------------------
# structural comparisons: no silent truncation


def test_structure_difference():
    a = (torch.zeros(2, 3), torch.zeros(2))
    assert structure_difference(a, (torch.ones(2, 3), torch.ones(2))) is None
    assert "missing output[1]" in structure_difference(a, a[:1])
    assert "new output[2]" in structure_difference(a, (*a, torch.zeros(1)))
    assert "changed shape" in structure_difference(a, (torch.zeros(2, 4), torch.zeros(2)))
    assert "changed dtype" in structure_difference(a, (a[0].double(), a[1]))
    changed_key = structure_difference({"a": torch.zeros(1)}, {"b": torch.zeros(1)})
    assert "missing output['a']" in changed_key and "new output['b']" in changed_key
    t1, t2 = torch.zeros(2, 3), torch.ones(2)
    # dict insertion order is not structure; list order is (the index is part of the path)
    assert structure_difference({"a": t1, "b": t2}, {"b": t2, "a": t1}) is None
    assert "new output['c']" in structure_difference({"a": t1, "b": t2}, {"a": t1, "c": t2})
    assert "changed shape" in structure_difference([t1, t2], [t2, t1])


def test_reordered_dict_outputs_are_compared_by_key(xy):
    class Reordering(nn.Module):
        """Returns the same dict with its keys in an order that depends on the batch."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            logits = self.fc(x)
            out = {"logits": logits, "score": logits.sum(1)}
            return out if bool(torch.equal(x[0], x[1])) else dict(reversed(list(out.items())))

    report = check_model(Reordering(), *xy, loss_fn=lambda o, y: F.cross_entropy(o["logits"], y))
    assert report["batch_independence"].status is Status.PASS, str(report)
    assert report["eval_determinism"].status is Status.PASS, str(report)

    class Mixing(Reordering):
        def forward(self, x):
            out = super().forward(x)
            return {**out, "logits": out["logits"] - out["logits"].mean(0)}  # couples the batch

    report = check_model(Mixing(), *xy, loss_fn=lambda o, y: F.cross_entropy(o["logits"], y))
    assert report["batch_independence"].status is Status.FAIL, str(report)


def _sample_dependent(make_output):
    """A model whose output structure depends on whether sample 0 equals sample 1."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            y = self.fc(x)
            return make_output(y, bool(torch.equal(x[0], x[1])))

    return Model()


@pytest.mark.parametrize(
    "make_output",
    [
        pytest.param(lambda y, same: (y,) if same else (y, y.sum(1)), id="missing second leaf"),
        pytest.param(lambda y, same: (y, y.sum(1), y) if same else (y, y.sum(1)), id="extra leaf"),
        pytest.param(lambda y, same: {"b" if same else "a": y}, id="changed dict key"),
        pytest.param(lambda y, same: [y] * (3 if same else 2), id="changed tuple length"),
        pytest.param(lambda y, same: y[:, :2] if same else y, id="changed output shape"),
    ],
)
def test_batch_independence_detects_structural_changes(xy, make_output):
    report = check_model(_sample_dependent(make_output), *xy, overfit_steps=0)
    result = report["batch_independence"]
    assert result.status is Status.FAIL, str(report)
    assert "structure" in result.message


def test_per_sample_outputs_may_change_shape(xy):
    class PerSample(nn.Module):
        """Returns one tensor per sample whose length depends on that sample only."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            y = self.fc(x)
            return [row[: 2 + int(sample[0] > 0)] for row, sample in zip(y, x)]

    x = xy[0].clone()
    x[0, 0], x[1, 0] = -1.0, 1.0  # replacing sample 0 by sample 1 changes the length of output 0
    report = check_model(PerSample(), x, overfit_steps=0)
    assert report["batch_independence"].status is not Status.FAIL, str(report)


def test_eval_determinism_detects_structural_changes(xy):
    class Flaky(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
            self.register_buffer("calls", torch.zeros(()))

        def forward(self, x):
            self.calls += 1
            y = self.fc(x)
            return (y, y.sum(1)) if int(self.calls) % 2 else (y,)

    result = check_model(Flaky(), xy[0], overfit_steps=0)["eval_determinism"]
    assert result.status is Status.WARN and "differently structured" in result.message


def test_input_dependency_counts_structural_changes_as_influence():
    class TokenCount(nn.Module):
        """The tokens only decide how many outputs there are."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x, tokens):
            y = self.fc(x)
            return (y,) if int(tokens[0]) == 0 else (y, y.sum(1))

    x, tokens = torch.randn(4, 8), torch.tensor([0, 1, 2, 3])
    result = check_model(TokenCount(), (x, tokens), overfit_steps=0)["input_dependency"]
    assert result.status is Status.PASS, result.message


def test_invariance_detects_structural_changes():
    model = _sample_dependent(lambda y, same: (y,) if same else (y, y.sum(1)))
    x = torch.randn(4, 16)

    def perturb(inputs):
        inputs = inputs.clone()
        inputs[0] = inputs[1]
        return inputs

    with pytest.raises(InvarianceException, match="changed structure.*missing output\\[1\\]"):
        assert_invariance(model, x, perturb)


# ------------------------------------------------------------------------------------------------
# separate batch dimensions of inputs and outputs


class SequenceClassifier(nn.Module):
    """Sequence-first inputs [T, N, F], outputs [N, C]."""

    def __init__(self, mix_batch=False):
        super().__init__()
        self.rnn = nn.GRU(8, 16)
        self.head = nn.Linear(16, 3)
        self.mix_batch = mix_batch

    def forward(self, x):
        _, h = self.rnn(x)
        h = h[-1]
        if self.mix_batch:
            h = h - h.mean(0, keepdim=True)  # normalisation over the batch by mistake
        return self.head(h)


def test_separate_output_batch_dim():
    x = torch.randn(5, 6, 8)
    report = check_model(SequenceClassifier(), x, batch_dim=1, overfit_steps=0)
    for name in ("output_varies", "batch_independence"):
        assert report[name].status is Status.SKIP
        assert "output_batch_dim" in report[name].message
    report = check_model(SequenceClassifier(), x, batch_dim=1, output_batch_dim=0, overfit_steps=0)
    assert report["output_varies"].status is Status.PASS, str(report)
    assert report["batch_independence"].status is Status.PASS, str(report)
    buggy = check_model(
        SequenceClassifier(mix_batch=True), x, batch_dim=1, output_batch_dim=0, overfit_steps=0
    )
    assert buggy["batch_independence"].status is Status.FAIL


def test_plugin_output_batch_dim():
    from torchtest2 import Plugin

    class SeqFirstClassifier(Plugin):
        batch_dim = 1
        output_batch_dim = 0

    x = torch.randn(5, 6, 8)
    for kwargs in ({}, {"batch_dim": 1}):  # an explicit batch_dim keeps the plugin's output dim
        report = check_model(SequenceClassifier(), x, plugin=SeqFirstClassifier(), **kwargs)
        assert report["output_varies"].status is Status.PASS, (kwargs, str(report))
        assert report["batch_independence"].status is Status.PASS, (kwargs, str(report))
        buggy = check_model(
            SequenceClassifier(mix_batch=True), x, plugin=SeqFirstClassifier(), **kwargs
        )
        assert buggy["batch_independence"].status is Status.FAIL, kwargs
    # an explicit output_batch_dim wins over the plugin's
    report = check_model(SequenceClassifier(), x, plugin=SeqFirstClassifier(), output_batch_dim=1)
    assert report["batch_independence"].status is Status.SKIP


# ------------------------------------------------------------------------------------------------
# a broken third-party plugin does not break the lookup


def test_broken_third_party_plugin_is_isolated(monkeypatch):
    loaded = []

    class Broken:
        name = "broken"

        @staticmethod
        def load():
            loaded.append("broken")
            raise ImportError("missing dependency")

    monkeypatch.setattr(
        "torchtest2.plugins.entry_points",
        lambda group: [Broken()] if group == "torchtest2.plugins" else [],
    )
    for builtin in ("lightning", "huggingface", "torchvision_detection"):
        assert get_plugin(builtin).name == builtin
    assert not loaded  # a built-in name does not load any entry point
    with pytest.warns(RuntimeWarning, match="'broken' could not be loaded"):
        plugins = plugin_discovery()
    assert "broken" not in plugins and "lightning" in plugins
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="unknown plugin 'broken'"):
        get_plugin("broken")


def test_dataclass_helpers_ignore_unset_fields():
    @dataclass
    class Partial:
        x: torch.Tensor
        y: torch.Tensor = field(init=False)

    partial = Partial(torch.ones(1))  # y is never set
    assert [p for p, _ in tree_leaves(partial)] == [".x"]
    assert not hasattr(tree_map(lambda _, t: t.clone(), partial), "y")
    assert dataclasses.is_dataclass(tree_map(lambda _, t: t, partial))
