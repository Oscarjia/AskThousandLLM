import pytest

from DeZero.config import Config


@pytest.fixture(autouse=True)
def enable_backprop():
    """Ensure backprop is on for every test and leave no side effects."""
    original = Config.enable_backprop
    Config.enable_backprop = True
    yield
    Config.enable_backprop = original

