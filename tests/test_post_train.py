import pytest
import torch
import torch.nn as nn

from torchtest2 import assert_directional, assert_invariance, assert_minimum_functionality
from torchtest2.exceptions import (
    DirectionalExpectationException,
    InvarianceException,
    MinimumFunctionalityException,
)


class SumClassifier(nn.Module):
    """'Trained' model: class 1 iff the sum of the features is positive."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        score = self.dropout(x).sum(-1)
        return torch.stack([-score, score], dim=-1)


def test_minimum_functionality():
    model = SumClassifier()
    x = torch.tensor([[1.0, 2.0], [-3.0, -1.0], [5.0, -1.0]])
    assert assert_minimum_functionality(model, x, torch.tensor([1, 0, 1])) == 1.0
    assert model.training  # train/eval flag restored
    with pytest.raises(MinimumFunctionalityException, match="66.7%") as info:
        assert_minimum_functionality(model, x, torch.tensor([1, 1, 1]))
    assert info.value.failing_indices == [1]
    assert_minimum_functionality(model, x, torch.tensor([1, 1, 1]), min_accuracy=0.6)


def test_invariance():
    model = SumClassifier()
    x = torch.randn(10, 4)
    assert_invariance(model, x, lambda t: t.flip(-1))  # the sum does not depend on the order
    assert_invariance(model, x, lambda t: t * 2, output_fn=lambda out: out.argmax(-1))
    with pytest.raises(InvarianceException, match="10/10"):
        assert_invariance(model, x, lambda t: t * 2)


def test_directional():
    model = SumClassifier()
    x = torch.randn(10, 4)
    positive = lambda out: out.softmax(-1)[:, 1]  # noqa: E731
    delta = assert_directional(model, x, lambda t: t + 1, output_fn=positive, expect="increase")
    assert (delta > 0).all()
    assert_directional(model, x, lambda t: t - 1, output_fn=positive, expect="decrease")
    with pytest.raises(DirectionalExpectationException) as info:
        assert_directional(model, x, lambda t: t - 1, output_fn=positive, expect="increase")
    assert info.value.failing_indices == list(range(10))
    with pytest.raises(ValueError):
        assert_directional(model, x, lambda t: t, output_fn=positive, expect="up")
