import numpy as np
import pytest

from DeZero.variable import Variable


def test_variable_initialization_with_ndarray():
    data = np.array([1.0, 2.0, 3.0])
    var = Variable(data)

    assert var.data is data
    assert var.grad is None
    assert var.creator is None


def test_variable_mul_supports_scalar_inputs():
    x = Variable(np.array(2.0))
    y = x * 3.0

    assert isinstance(y, Variable)
    assert y.data == pytest.approx(6.0)


def test_variable_mul_backward_populates_gradients():
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = x * y
    z.backward()

    assert x.grad == pytest.approx(3.0)
    assert y.grad == pytest.approx(2.0)

