"""JAX has native TPU support already — no compat layer needed, just the
right init order. This proves devices are visible and a basic op runs."""
import time

import jax
import jax.numpy as jnp


def main():
    print(f"jax devices: {jax.devices()}")
    assert jax.devices()[0].platform == "tpu", "no TPU backend picked up"

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (2048, 2048))
    b = jax.random.normal(key, (2048, 2048))

    matmul = jax.jit(jnp.matmul)
    start = time.time()
    c = matmul(a, b)
    c.block_until_ready()  # jit is async, force completion before timing
    elapsed = time.time() - start

    assert c.shape == (2048, 2048)
    print(f"matmul ok, shape={c.shape}, first-compile+run={elapsed:.2f}s")


if __name__ == "__main__":
    main()
