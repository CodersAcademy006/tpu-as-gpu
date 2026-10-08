"""Category D fixes from SCOPE.md: precision & numerics conventions.

Both rows here are "pick the right convention," not "run the numbers" —
the dispatch itself doesn't need TPU hardware to be correct, only the
training-loop behavior it feeds into does.
"""
from __future__ import annotations


def autocast_dtype(device_kind: str) -> str:
    """Mixed-precision dtype for the given device kind ("xla"/"cuda"/"cpu").

    GPU convention is fp16 + loss scaling (`torch.cuda.amp.GradScaler`) —
    fp16 has a narrow dynamic range and needs it. TPU/XLA convention is
    bf16, no loss scaling needed (wider dynamic range) — `GradScaler`
    itself is CUDA-only and won't even run on an XLA device.
    """
    return "bfloat16" if device_kind == "xla" else "float16"


def compile_backend(device_kind: str) -> str:
    """`torch.compile` backend for the given device kind.

    The default backend, Inductor, generates Triton/CUDA — no TPU path.
    `backend="openxla"` is the XLA-native compile path.
    """
    return "openxla" if device_kind == "xla" else "inductor"
