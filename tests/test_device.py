import pytest

from tpu_as_gpu.device import device_kind, get_torch_device


def test_get_torch_device_returns_a_device():
    torch = pytest.importorskip("torch")
    device = get_torch_device()
    assert isinstance(device, torch.device) or "xla" in str(device)


def test_device_kind_classifies_strings():
    assert device_kind("xla:0") == "xla"
    assert device_kind("cuda:0") == "cuda"
    assert device_kind("cpu") == "cpu"
