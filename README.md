# tpu-as-gpu

Make Kaggle's free TPU run normal ML workloads the way a GPU does, instead of
sitting idle because a given workload only has a CUDA path.

## Why

`faster-whisper`/CTranslate2 (used for the Anchor Of Life transcription job)
is CUDA-only. Kaggle's 2xT4 GPU quota covered that job, but the TPU quota
went unused because nothing in that pipeline runs on XLA. This repo is the
general fix: per-framework helper code + benchmark workloads, so future
Kaggle jobs can burn TPU hours instead of only GPU hours, whatever framework
they're written in.

## Framework support matrix

| Framework | TPU support | What's needed |
|---|---|---|
| JAX | Native, first-class | Just the right init order — `jax/` has a smoke test, no compat layer required |
| TensorFlow | Native (`TPUStrategy`, oldest TPU support of any framework) | Cluster-resolver boilerplate — `tensorflow/` has a smoke test, no compat layer required |
| PyTorch | Via `torch_xla` bridge | Real compat work — `torch_xla/` has the smoke test + the Whisper port |
| MLX | **Not possible** | Metal-only framework, no XLA/TPU backend exists, no Apple roadmap for one — see `UNSUPPORTED_MLX.md`. Not a Kaggle issue, not fixable from this repo |

## First benchmark: Whisper transcription on TPU

CTranslate2 has no XLA backend, so the port goes through
`transformers.WhisperForConditionalGeneration` (pure PyTorch) running on an
XLA device instead — see `torch_xla/whisper_tpu.py`. Run
`torch_xla/smoke_test.py` first on any fresh notebook; it proves the TPU
device is visible and a basic op/compile cycle works before trusting
anything heavier.

## Scope

Full list of GPU-only processes we're porting is in [`SCOPE.md`](SCOPE.md),
grouped by root cause (CUDA-only kernel, static-shape compile constraint,
distributed collectives, precision convention, profiling tooling, data
loading, dev workflow, code hygiene) rather than by framework, since the fix
pattern repeats within a group. Each row there has a matching
[GitHub issue](../../issues) so it can be picked up and closed on its own.

## Status

Scaffolded 2026-10-08. Currently working: Whisper transcription port
(`torch_xla/whisper_tpu.py`), not yet verified on real TPU hardware.
Everything else in `SCOPE.md` is tracked as an open issue, not started.

## Running on Kaggle

Kaggle's TPU VM notebooks ship `torch_xla`, `jax`, and `tensorflow` all
preinstalled and version-matched to the notebook's runtime — do not
`pip install` a different version of any of them, it will likely mismatch
and break. Set accelerator to "TPU VM v3-8" in notebook settings, then run
whichever framework's smoke test first:

```python
!python torch_xla/smoke_test.py
!python jax/smoke_test.py
!python tensorflow/smoke_test.py
!python torch_xla/whisper_tpu.py --audio path/to/clip.wav
```

## Structure

```
torch_xla/   # PyTorch-on-TPU bridge: smoke test + Whisper transcription benchmark
jax/         # JAX-on-TPU: smoke test (native support, no bridge needed)
tensorflow/  # TF-on-TPU: smoke test (native support, no bridge needed)
UNSUPPORTED_MLX.md  # why MLX is out of scope, permanently
```
