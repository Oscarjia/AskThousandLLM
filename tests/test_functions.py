import numpy as np
import pytest

from DeZero.function import exp, square
from DeZero.variable import Variable


def test_square_forward_and_backward():
    x = Variable(np.array(3.0))
    y = square(x)
    assert y.data == pytest.approx(9.0)

    y.backward()
    assert x.grad == pytest.approx(2 * 3.0)


def test_exp_forward_and_backward():
    x = Variable(np.array(0.0))
    y = exp(x)
    assert y.data == pytest.approx(1.0)

    y.backward()
    # derivative of exp at 0 is 1
    assert x.grad == pytest.approx(1.0)

