import pytest

torch = pytest.importorskip("torch")

from cnn import DigitNet  # noqa: E402


def test_forward_returns_log_probabilities():
    out = DigitNet().eval()(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(4), atol=1e-5)
