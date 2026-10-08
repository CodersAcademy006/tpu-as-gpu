"""Category E fixes from SCOPE.md: memory/profiling tooling.

Deep kernel profiling (row E2) isn't wrapped here — use
`torch_xla.debug.profiler.start_trace()`/`stop_trace()` or JAX's own
`jax.profiler.start_trace()`/`stop_trace()` directly, they're already the
right API; a wrapper would just be an extra layer. See `PERFORMANCE.md`.
"""
from __future__ import annotations

from typing import Any


def get_memory_info(device_kind: str, device: Any = None) -> dict | None:
    """Memory usage for the given device kind ("xla"/"cuda"/"cpu").

    GPU convention is `nvidia-smi`/`pynvml`. TPU/XLA convention is
    `xm.get_memory_info(device)`, a different call entirely — this picks
    the right one. Returns None on CPU (nothing to report).
    """
    if device_kind == "xla":
        import torch_xla.core.xla_model as xm
        return dict(xm.get_memory_info(device))
    if device_kind == "cuda":
        import torch
        free, total = torch.cuda.mem_get_info(device)
        return {"bytes_used": total - free, "bytes_limit": total}
    return None


def clear_cache(device_kind: str) -> bool:
    """Clears the allocator cache for the given device kind, if possible.

    `torch.cuda.empty_cache()` has no XLA equivalent — XLA manages its own
    allocator differently and doesn't expose a manual clear. Returns
    whether anything was actually cleared, so callers can tell a no-op
    from a real clear instead of assuming both behave the same.
    """
    if device_kind == "cuda":
        import torch
        torch.cuda.empty_cache()
        return True
    return False
