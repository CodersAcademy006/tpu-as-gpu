"""Category H fix from SCOPE.md: device placement.

Model code sprinkled with hardcoded `.cuda()` assumes CUDA is always
available. That breaks on TPU (no CUDA at all) and on CPU-only boxes. Pick
the best available device once; model code takes a `device` argument
instead of a backend assumption.
"""
from __future__ import annotations


def get_torch_device():
    """Best available torch device: XLA TPU > CUDA GPU > CPU."""
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except ImportError:
        pass

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_kind(device) -> str:
    """Returns "xla", "cuda", or "cpu" for any torch-like device object."""
    s = str(device)
    if "xla" in s:
        return "xla"
    if "cuda" in s:
        return "cuda"
    return "cpu"
