"""Category H fix from SCOPE.md: seeding/determinism.

GPU code seeds via `torch.cuda.manual_seed`, which has no XLA equivalent —
torch_xla reuses torch's own RNG state, but JAX has no global RNG state at
all. This seeds every backend that's importable and hands back the one
thing that can't be made global: the JAX key.
"""
from __future__ import annotations

import random
from typing import Any


def seed_everything(seed: int) -> Any:
    """Seeds stdlib random, numpy, and torch (CPU+CUDA) globally.

    Returns a `jax.random.PRNGKey(seed)` if JAX is installed, else None.
    JAX has no global RNG state — thread the returned key through your
    code explicitly, seeding it here doesn't make it global.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import jax

        return jax.random.PRNGKey(seed)
    except ImportError:
        return None
