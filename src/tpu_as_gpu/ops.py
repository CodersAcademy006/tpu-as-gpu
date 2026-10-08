"""Category A fix from SCOPE.md: custom CUDA ops (NMS row).

`torchvision.ops.nms` is a CUDA kernel. No TPU-native kernel exists, so
this is an XLA-native reimplementation, not a port: a pure-JAX greedy NMS.

Returns a length-N boolean keep mask instead of `torchvision.ops.nms`'s
variable-length index tensor — a data-dependent output *length* needs a
dynamic shape, which doesn't compile on TPU (SCOPE.md Category B: XLA
needs a bounded shape). Caller does `boxes[keep]` off-device, or uses the
mask with a fixed-size top-k when a bounded output count is required.

Doesn't cover ROI align or deformable conv (same SCOPE row, not started).
"""
from __future__ import annotations

from typing import Any


def _box_iou(boxes: Any, jnp: Any) -> Any:
    """Pairwise IoU, boxes as (N, 4) xyxy. Returns (N, N)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    ix1 = jnp.maximum(x1[:, None], x1[None, :])
    iy1 = jnp.maximum(y1[:, None], y1[None, :])
    ix2 = jnp.minimum(x2[:, None], x2[None, :])
    iy2 = jnp.minimum(y2[:, None], y2[None, :])

    inter = jnp.clip(ix2 - ix1, 0) * jnp.clip(iy2 - iy1, 0)
    union = area[:, None] + area[None, :] - inter
    return inter / jnp.clip(union, 1e-9)


def nms(boxes: Any, scores: Any, iou_threshold: float = 0.5) -> Any:
    """Greedy NMS, JAX arrays only. Returns a static-shape bool keep mask.

    Raises NotImplementedError for anything that isn't a JAX array — no
    torch_xla-native path identified yet, this doesn't silently fall back
    to a CPU loop.
    """
    module_root = type(boxes).__module__.split(".")[0]
    if module_root not in ("jax", "jaxlib"):
        raise NotImplementedError(
            "nms only has a JAX backend right now — no torch_xla-native "
            "path identified. See SCOPE.md Category A."
        )
    import jax
    import jax.numpy as jnp

    n = boxes.shape[0]
    order = jnp.argsort(-scores)
    ious = _box_iou(boxes[order], jnp)
    idx = jnp.arange(n)

    def body(i, keep):
        suppress = (ious[i] > iou_threshold) & (idx > i) & keep[i]
        return keep & ~suppress

    keep_sorted = jax.lax.fori_loop(0, n, body, jnp.ones(n, dtype=bool))
    return jnp.zeros(n, dtype=bool).at[order].set(keep_sorted)
