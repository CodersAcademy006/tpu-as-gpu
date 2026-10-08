import random

from tpu_as_gpu.seeding import seed_everything


def test_seed_everything_makes_stdlib_random_deterministic():
    seed_everything(42)
    first = [random.random() for _ in range(5)]
    seed_everything(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_seed_everything_returns_jax_key_if_jax_installed():
    try:
        import jax
    except ImportError:
        assert seed_everything(0) is None
        return

    key = seed_everything(0)
    assert key is not None
