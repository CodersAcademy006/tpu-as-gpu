"""Category B fixes from SCOPE.md: static-shape/XLA-compile constraints.

TPU (XLA) compiles a graph for a fixed shape; a Python `if` or an append-
and-grow buffer forces a new shape, and a new shape forces a new compile.
Each helper here is the minimum code needed on top of what the frameworks
already provide natively.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence


def where(condition: Any, on_true: Any, on_false: Any) -> Any:
    """Framework-dispatching replacement for `if traced_value: ...`.

    XLA can't trace a Python `if` on a traced value — every framework
    already ships the fix (`torch.where`, `jnp.where`, `tf.where`), this
    just picks the right one so calling code doesn't need a per-backend
    import. Also the fix for "early-stopping" loops: compute the update
    for every row unconditionally, then use `where(done_mask, old, new)`
    to freeze rows that already finished instead of breaking the loop.
    """
    module_root = type(condition).__module__.split(".")[0]

    if module_root == "torch":
        import torch

        return torch.where(condition, on_true, on_false)
    if module_root in ("jax", "jaxlib"):
        import jax.numpy as jnp

        return jnp.where(condition, on_true, on_false)
    if module_root == "tensorflow":
        import tensorflow as tf

        return tf.where(condition, on_true, on_false)

    import numpy as np

    return np.where(condition, on_true, on_false)


def bucket_length(length: int, buckets: Sequence[int]) -> int:
    """Rounds `length` up to the smallest bucket that fits it.

    Each distinct sequence length triggers a new XLA compile. Padding every
    sequence to one of a small, fixed set of bucket lengths bounds the
    number of compiles instead of compiling once per distinct length.
    """
    fitting = [b for b in sorted(buckets) if length <= b]
    if not fitting:
        raise ValueError(f"length {length} exceeds largest bucket {max(buckets)}")
    return fitting[0]


class DeferredSync:
    """Buffers values instead of syncing (`.item()`/`.cpu()`) every step.

    GPU eager execution tolerates a sync every iteration; on TPU each one
    is a device round-trip that stalls the whole traced graph. Collect
    values across the hot loop, resolve them once at the end.
    """

    def __init__(self) -> None:
        self._pending: list[Any] = []

    def collect(self, value: Any) -> None:
        self._pending.append(value)

    def resolve(self, sync_fn: Callable[[Sequence[Any]], list[Any]] | None = None) -> list[Any]:
        if sync_fn is None:
            sync_fn = lambda values: [v.item() if hasattr(v, "item") else v for v in values]
        values = sync_fn(self._pending)
        self._pending = []
        return values


class StaticKVCache:
    """Fixed-size KV cache buffer, written in place instead of grown.

    Autoregressive generation that appends each new token's KV forces a
    new shape, hence a new compile, every step. Pre-allocate the full
    `max_len` buffer once and write into a known position instead.

    For torch/numpy only (both support in-place item assignment). JAX
    arrays are immutable — for JAX, use `jax.lax.dynamic_update_slice_in_dim`
    directly, it's already the native fix, wrapping it here would be the
    extra layer this class exists to avoid.
    """

    def __init__(self, max_len: int, shape_per_token: tuple[int, ...], zeros_fn: Callable[[tuple], Any]):
        self.max_len = max_len
        self.buffer = zeros_fn((max_len, *shape_per_token))
        self.pos = 0

    def write(self, token_kv: Any) -> tuple[Any, int]:
        if self.pos >= self.max_len:
            raise IndexError("KV cache full")
        self.buffer[self.pos] = token_kv
        self.pos += 1
        return self.buffer, self.pos
