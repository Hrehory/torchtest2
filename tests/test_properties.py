"""Property-based tests (hypothesis): invariants that must hold for *any* input structure and
any small model, not just for hand-picked examples."""

import os
from collections import namedtuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from torchtest2 import check_model  # noqa: E402
from torchtest2._tree import tree_leaves, tree_map  # noqa: E402

Pair = namedtuple("Pair", "first second")
# TT2_PROPERTY_EXAMPLES=400 pytest tests/test_properties.py for a longer bug hunt
MODEL_EXAMPLES = int(os.environ.get("TT2_PROPERTY_EXAMPLES", 25))

tensors = st.builds(
    lambda shape, seed: torch.randn(shape, generator=torch.Generator().manual_seed(seed)),
    st.lists(st.integers(1, 3), min_size=0, max_size=3).map(tuple),
    st.integers(0, 1000),
)
scalars = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=3))
trees = st.recursive(
    st.one_of(tensors, scalars),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(st.text("abc", min_size=1, max_size=2), children, max_size=3),
        st.tuples(children, children).map(lambda t: Pair(*t)),
    ),
    max_leaves=10,
)


def same_structure(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            type(a) is type(b)
            and a.keys() == b.keys()
            and all(same_structure(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(same_structure, a, b))
    return a == b


@given(trees)
@settings(max_examples=200, deadline=None)
def test_tree_map_identity_preserves_structure(tree):
    assert same_structure(tree, tree_map(lambda _, t: t, tree))


@given(trees)
@settings(max_examples=200, deadline=None)
def test_tree_leaves_are_the_mapped_tensors_in_order(tree):
    leaves = tree_leaves(tree, "x")
    assert len({path for path, _ in leaves}) == len(leaves)  # paths are unique
    doubled = tree_map(lambda _, t: t * 2, tree, "x")
    for (p1, t1), (p2, t2) in zip(leaves, tree_leaves(doubled, "x")):
        assert p1 == p2 and torch.equal(t1 * 2, t2)


# ------------------------------------------------------------------------------------------------
# check_model on random small models

ACTIVATIONS = [nn.ReLU, nn.Tanh, nn.GELU, nn.SiLU]


@st.composite
def small_setups(draw):
    features = draw(st.integers(2, 12))
    widths = draw(st.lists(st.integers(2, 16), min_size=1, max_size=3))
    classes = draw(st.integers(2, 5))
    batch = draw(st.integers(2, 8))
    extras = draw(st.lists(st.sampled_from(["layernorm", "batchnorm", "dropout"]), max_size=2))
    seed = draw(st.integers(0, 10_000))
    torch.manual_seed(seed)
    layers, width = [], features
    for w in widths:
        layers += [nn.Linear(width, w), draw(st.sampled_from(ACTIVATIONS))()]
        width = w
    for extra in extras:
        layers.append(
            {
                "layernorm": nn.LayerNorm(width),
                "batchnorm": nn.BatchNorm1d(width),
                "dropout": nn.Dropout(0.2),
            }[extra]
        )
    layers.append(nn.Linear(width, classes))
    model = nn.Sequential(*layers)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, features, generator=g)
    y = torch.randint(0, classes, (batch,), generator=g)
    if draw(st.booleans()):
        model.train()
    else:
        model.eval()
    optimizer = draw(st.sampled_from(["adam", "sgd", "none"]))
    opt = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=1e-2),
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        "none": lambda: None,
    }[optimizer]()
    return model, x, y, opt


def fingerprint(model, x, opt):
    return (
        {k: v.clone() for k, v in model.state_dict().items()},
        [m.training for m in model.modules()],
        torch.get_rng_state(),
        x.clone(),
        None if opt is None else str(opt.state_dict()),
    )


def unchanged(a, b):
    return (
        a[0].keys() == b[0].keys()
        and all(torch.equal(a[0][k], b[0][k]) for k in a[0])
        and a[1] == b[1]
        and torch.equal(a[2], b[2])
        and torch.equal(a[3], b[3])
        and a[4] == b[4]
    )


@given(small_setups())
@settings(max_examples=MODEL_EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_no_side_effects_and_determinism(setup):
    model, x, y, opt = setup
    before = fingerprint(model, x, opt)
    first = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=opt, overfit_steps=15)
    assert unchanged(before, fingerprint(model, x, opt))
    second = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=opt, overfit_steps=15)
    assert [(r.name, r.status, r.message) for r in first] == [
        (r.name, r.status, r.message) for r in second
    ]
    assert all(r.seconds >= 0 for r in first)


BATCH_ORDER_INVARIANT = (
    "forward",
    "output_finite",
    "loss",
    "optimizer_coverage",
    "gradient_flow",
    "params_change",
    "frozen_params",
    "input_dependency",
    "batch_independence",
)


@given(small_setups(), st.integers(0, 1000))
@settings(max_examples=MODEL_EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_verdicts_do_not_depend_on_the_order_of_the_samples(setup, perm_seed):
    model, x, y, opt = setup
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(perm_seed))
    kwargs = dict(loss_fn=F.cross_entropy, optimizer=opt, overfit_steps=0)
    a = check_model(model, x, y, **kwargs)
    b = check_model(model, x[perm], y[perm], **kwargs)
    for name in BATCH_ORDER_INVARIANT:
        assert a[name].status == b[name].status, (name, a[name].message, b[name].message)
