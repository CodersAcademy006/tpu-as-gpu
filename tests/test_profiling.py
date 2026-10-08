import pytest

from tpu_as_gpu.profiling import clear_cache, get_memory_info


def test_get_memory_info_cpu_is_none():
    assert get_memory_info("cpu") is None


def test_get_memory_info_xla_real_hardware():
    pytest.importorskip("torch_xla")
    from tpu_as_gpu.device import get_torch_device
    info = get_memory_info("xla", get_torch_device())
    assert info["bytes_used"] > 0
    assert info["bytes_limit"] > info["bytes_used"]


def test_clear_cache_xla_is_noop():
    assert clear_cache("xla") is False


def test_clear_cache_cpu_is_noop():
    assert clear_cache("cpu") is False


def test_clear_cache_cuda_real_hardware():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert clear_cache("cuda") is True
