import numpy as np
import pytest

from tpu_as_gpu.compat import DeferredSync, StaticKVCache, bucket_length, where


def test_where_numpy_default():
    cond = np.array([True, False, True])
    out = where(cond, np.array([1, 1, 1]), np.array([0, 0, 0]))
    assert out.tolist() == [1, 0, 1]


def test_where_dispatches_on_torch():
    torch = pytest.importorskip("torch")
    cond = torch.tensor([True, False])
    out = where(cond, torch.tensor([1, 1]), torch.tensor([0, 0]))
    assert out.tolist() == [1, 0]


def test_where_dispatches_on_jax():
    jnp = pytest.importorskip("jax.numpy")
    cond = jnp.array([True, False])
    out = where(cond, jnp.array([1, 1]), jnp.array([0, 0]))
    assert out.tolist() == [1, 0]


def test_bucket_length_rounds_up():
    assert bucket_length(5, [8, 16, 32]) == 8
    assert bucket_length(8, [8, 16, 32]) == 8
    assert bucket_length(17, [8, 16, 32]) == 32


def test_bucket_length_overflow_raises():
    with pytest.raises(ValueError):
        bucket_length(100, [8, 16, 32])


def test_deferred_sync_collects_then_resolves():
    sync = DeferredSync()
    for v in [1, 2, 3]:
        sync.collect(v)
    assert sync.resolve() == [1, 2, 3]
    assert sync.resolve() == []  # buffer cleared after resolve


def test_static_kv_cache_writes_in_order():
    cache = StaticKVCache(max_len=3, shape_per_token=(2,), zeros_fn=np.zeros)
    buf, pos = cache.write(np.array([1.0, 2.0]))
    assert pos == 1
    assert buf[0].tolist() == [1.0, 2.0]

    cache.write(np.array([3.0, 4.0]))
    cache.write(np.array([5.0, 6.0]))
    with pytest.raises(IndexError):
        cache.write(np.array([7.0, 8.0]))
