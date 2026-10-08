# tpu-as-gpu

Make Kaggle's free TPU run normal ML workloads the way a GPU does, instead of
sitting idle because a given workload only has a CUDA path.

## Why

`faster-whisper`/CTranslate2 (used for the Anchor Of Life transcription job)
is CUDA-only. Kaggle's 2xT4 GPU quota covered that job, but the TPU quota
went unused because nothing in that pipeline runs on XLA. This repo is the
general fix: a framework-agnostic helper library + benchmark workloads, so
future Kaggle jobs can burn TPU hours instead of only GPU hours, whatever
framework they're written in.

## Install

```bash
pip install -e ".[dev]"
```

```python
from tpu_as_gpu import get_torch_device, seed_everything, where, bucket_length, DeferredSync, StaticKVCache
```

## Framework support matrix

| Framework | TPU support | What's needed |
|---|---|---|
| JAX | Native, first-class | Just the right init order — `examples/jax/` has a smoke test, no compat layer required |
| TensorFlow | Native (`TPUStrategy`, oldest TPU support of any framework) | Cluster-resolver boilerplate — `examples/tensorflow/` has a smoke test, no compat layer required |
| PyTorch | Via `torch_xla` bridge | Real compat work — `examples/torch_xla/` has the smoke test + the Whisper port |
| MLX | **Not possible** | Metal-only framework, no XLA/TPU backend exists, no Apple roadmap for one — see `UNSUPPORTED_MLX.md`. Not a Kaggle issue, not fixable from this repo |

## First benchmark: Whisper transcription on TPU

CTranslate2 has no XLA backend, so the port goes through
`transformers.WhisperForConditionalGeneration` (pure PyTorch) running on an
XLA device instead — see `examples/torch_xla/whisper_tpu.py`. Run
`examples/torch_xla/smoke_test.py` first on any fresh notebook; it proves the
TPU device is visible and a basic op/compile cycle works before trusting
anything heavier.

## Scope

Full list of GPU-only processes we're porting is in [`SCOPE.md`](SCOPE.md),
grouped by root cause (CUDA-only kernel, static-shape compile constraint,
distributed collectives, precision convention, profiling tooling, data
loading, dev workflow, code hygiene) rather than by framework, since the fix
pattern repeats within a group. Each row there has a matching
[GitHub issue](../../issues) so it can be picked up and closed on its own.

## Status

As of 2026-10-08: Categories B (static-shape), G (debugging), and H (device
placement/seeding) are closed at the code level — `src/tpu_as_gpu/` + pure
control-flow logic, tested on CPU, no hardware needed to verify the logic
itself. Whisper transcription port (`examples/torch_xla/whisper_tpu.py`) is
scaffolded, not yet run on real TPU hardware. Category A rows have 2
upstream solutions identified (Splash Attention, Ragged Paged Attention — see
`research/prior_art.md`) but not yet adopted. Categories C/D/E/F are
untouched — they need real TPU hardware to measure (profiling, pipeline
throughput) or an upstream library to adopt (distributed, precision), not
just logic. Real-hardware numbers land in `results/` once they exist, never
claimed before that.

## Running on Kaggle

Kaggle's TPU VM notebooks ship `torch_xla`, `jax`, and `tensorflow` all
preinstalled and version-matched to the notebook's runtime — do not
`pip install` a different version of any of them, it will likely mismatch
and break. Set accelerator to "TPU VM v3-8" in notebook settings, install
this repo's own library (pure Python, no conflicting deps), then run
whichever framework's smoke test first:

```python
!pip install -e .
!python examples/torch_xla/smoke_test.py
!python examples/jax/smoke_test.py
!python examples/tensorflow/smoke_test.py
!python examples/torch_xla/whisper_tpu.py --audio path/to/clip.wav
```

## Structure

```
src/tpu_as_gpu/   # installable library: compat.py, device.py, seeding.py
tests/            # pytest, CPU-only, importorskip for torch/jax
examples/
  torch_xla/      # PyTorch-on-TPU bridge: smoke test + Whisper transcription benchmark
  jax/            # JAX-on-TPU: smoke test (native support, no bridge needed)
  tensorflow/     # TF-on-TPU: smoke test (native support, no bridge needed)
results/          # real-hardware run output, one file per run, empty until a run exists
DEBUGGING.md      # how to debug traced/lazy tensor values on TPU
UNSUPPORTED_MLX.md  # why MLX is out of scope, permanently
```
