import pytest

from tpu_as_gpu.device import get_torch_device


def test_get_torch_device_returns_a_device():
    torch = pytest.importorskip("torch")
    device = get_torch_device()
    assert isinstance(device, torch.device) or "xla" in str(device)
