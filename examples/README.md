# Examples

| Dir | What | Run where |
|---|---|---|
| `torch_xla/`, `jax/`, `tensorflow/` | Per-framework smoke tests + the Whisper benchmark + `train_infer_demo.py` (train a model, then infer with it, same device) | Any TPU notebook, imports `tpu_as_gpu` from the installed package |
| `kaggle_verify.py` | Runs all of the above in one shot, writes `results/` | Kaggle TPU VM notebook, `pip install -e .` first |
| `kaggle/` | Single self-contained script (`verify_kernel.py`) + `kernel-metadata.json` for `kaggle kernels push` | Kaggle, via the `kaggle` CLI — script kernels only upload one file, so the library is inlined rather than imported |
| `colab/` | `tpu_as_gpu_verify.ipynb`, same checks as `kaggle/verify_kernel.py` | Colab, Runtime > Change runtime type > TPU, then Run all |

Why `kaggle/` and `colab/` duplicate `src/tpu_as_gpu` inline instead of
importing it: Kaggle script kernels only upload the single `code_file`
in `kernel-metadata.json`, and a Colab notebook distributed standalone
can't assume the private repo is cloned alongside it. `src/tpu_as_gpu/`
is the source of truth; these are verification snapshots of it, kept
in sync by hand when the library changes.

Real run so far: `results/colab_run_2026-10-08.md` — ran
`tpu_as_gpu_verify.ipynb`'s logic on a live Colab TPU v5e-1 session via
the `colab-mcp` tool. The matching Kaggle kernel
(`srijan1upadhyay/tpu-as-gpu-verify`) was pushed the same day but got
stuck in the TPU queue and was cancelled before it ran — Colab's TPU
queue was free, Kaggle's wasn't; the kernel is still there to retry.
