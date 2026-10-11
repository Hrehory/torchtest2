import math

import pytest
import torch

from torchtest2 import (
    TorchTestError,
    assert_all_finite,
    assert_in_range,
    assert_never_inf,
    assert_never_nan,
    assert_shape,
)
from torchtest2.exceptions import (
    ConsistencyException,
    InfTensorException,
    NaNTensorException,
    RangeException,
)


def test_assert_in_range():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert_in_range(t, 1, 3)
    assert_in_range(t, low=0)
    assert_in_range(t, high=3)
    assert_in_range(t, 0, 4, inclusive=False)
    with pytest.raises(RangeException, match=r"\[1, 2\].*max = 3"):
        assert_in_range(t, 1, 2)
    with pytest.raises(RangeException):
        assert_in_range(t, 1, 3, inclusive=False)
    with pytest.raises(RangeException):
        assert_in_range(torch.tensor([math.nan]), 0, 1)
    with pytest.raises(ValueError):
        assert_in_range(t)


def test_assert_never_inf():
    assert_never_inf(torch.tensor([0.0, 1.0, 2.0]))
    assert_never_inf(torch.tensor([0.0, math.nan]))  # NaN is not inf
    with pytest.raises(InfTensorException, match="1 infinite"):
        assert_never_inf(torch.tensor([0.0, 1.0, math.inf, 2.0]))
    with pytest.raises(InfTensorException):
        assert_never_inf(torch.tensor([-math.inf]))


def test_assert_never_nan():
    assert_never_nan(torch.tensor([0, 1, 2]))
    with pytest.raises(NaNTensorException, match="2 NaN"):
        assert_never_nan(torch.tensor([0.0, math.nan, 1.0, math.nan]))


def test_tensor_name_in_messages():
    with pytest.raises(NaNTensorException, match="Tensor 'logits' contains 1 NaN"):
        assert_never_nan(torch.tensor([math.nan]), "logits")
    with pytest.raises(InfTensorException, match="Tensor 'logits' contains 1 infinite"):
        assert_all_finite(torch.tensor([math.inf]), tensor_name="logits")


def test_assert_all_finite():
    assert_all_finite(torch.ones(3))
    with pytest.raises(NaNTensorException):
        assert_all_finite(torch.tensor([math.nan]))
    with pytest.raises(InfTensorException):
        assert_all_finite(torch.tensor([math.inf]))


def test_assert_shape():
    t = torch.zeros(4, 10)
    assert_shape(t, (4, 10))
    assert_shape(t, (None, 10))
    assert_shape(t, (-1, 10))
    with pytest.raises(ConsistencyException, match=r"\(\*, 11\)"):
        assert_shape(t, (None, 11))
    with pytest.raises(ConsistencyException):
        assert_shape(t, (4, 10, 1))


def test_exceptions_are_assertion_errors():
    with pytest.raises(AssertionError):
        assert_never_nan(torch.tensor([math.nan]))
    assert issubclass(RangeException, TorchTestError)
