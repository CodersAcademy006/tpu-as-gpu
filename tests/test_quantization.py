import pytest

from tpu_as_gpu.quantization import quantized_matmul


def test_quantized_matmul_rejects_non_jax():
    import numpy as np

    with pytest.raises(NotImplementedError):
        quantized_matmul(np.zeros((2, 2)), np.zeros((2, 2)))


def test_quantized_matmul_jax_real():
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("aqt")
    import jax

    a = jax.random.normal(jax.random.PRNGKey(0), (8, 8))
    b = jax.random.normal(jax.random.PRNGKey(1), (8, 8))
    out = quantized_matmul(a, b, bits=8)
    ref = jnp.matmul(a, b)
    rel_err = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    assert rel_err < 0.05
