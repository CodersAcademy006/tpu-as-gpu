"""Category A fix from SCOPE.md: 4-bit/8-bit quantization.

`bitsandbytes` is CUDA-only, no TPU path at all. AQT (google/aqt, pip
package `aqtp`) is the JAX-native equivalent — quantized `dot_general`,
verified on real TPU hardware (int8, Colab v5e-1, 2026-10-08, max relative
error 0.0091 vs fp32 — see `results/colab_run_2026-10-08.md`).

No PyTorch/torch_xla-native quantization path has been identified yet —
this is JAX/AQT only. Doesn't cover SCOPE row A6 (8-bit optimizers,
`Adam8bit`-style) — that's optimizer-state compression, a different
problem AQT doesn't address.
"""
from __future__ import annotations

from typing import Any


def quantized_matmul(a: Any, b: Any, bits: int = 8) -> Any:
    """int8 (or other bit-width) quantized matmul via AQT, JAX arrays only.

    Raises NotImplementedError for anything that isn't a JAX array — no
    torch_xla-native quantization path exists yet, this doesn't silently
    fall back to an unquantized matmul.
    """
    module_root = type(a).__module__.split(".")[0]
    if module_root not in ("jax", "jaxlib"):
        raise NotImplementedError(
            "quantized_matmul only has a JAX/AQT backend right now — no "
            "torch_xla-native quantization path identified. See "
            "research/prior_art.md."
        )
    import aqt.jax.v2.aqt_dot_general as aqt_dg

    dg_fn = aqt_dg.dot_general_make(lhs_bits=bits, rhs_bits=bits)
    dimension_numbers = (((a.ndim - 1,), (0,)), ((), ()))
    return dg_fn(a, b, dimension_numbers)
