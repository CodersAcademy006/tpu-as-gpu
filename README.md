# tpu-as-gpu

Make Kaggle's free TPU run normal ML workloads the way a GPU does, instead of
sitting idle because the workload only has a CUDA path.

## Why

`faster-whisper`/CTranslate2 (used for the Anchor Of Life transcription job)
is CUDA-only. Kaggle's 2xT4 GPU quota covered that job, but the TPU quota
went unused because nothing in that pipeline runs on XLA. This repo is the
general fix: a reusable compat layer + benchmark workloads, so future Kaggle
jobs can burn TPU hours instead of only GPU hours.

## Approach

`torch_xla` first. Keep model code as close to normal PyTorch as possible,
swap the device to `xm.xla_device()`. Lower rewrite cost than a full JAX
port. Fall back to JAX/Flax per-workload only if a specific op has no
working XLA lowering in `torch_xla`.

## First benchmark: Whisper transcription on TPU

CTranslate2 has no XLA backend, so the port goes through
`transformers.WhisperForConditionalGeneration` (pure PyTorch) running on an
XLA device instead. `scripts/whisper_tpu.py` is the first real workload;
`scripts/smoke_test.py` is the sanity check that runs before it (proves the
TPU device is visible and a basic op/compile cycle works at all).

## Status

Scaffolded 2026-10-08. Nothing benchmarked yet. Run `smoke_test.py` on a
Kaggle TPU notebook first, before trying `whisper_tpu.py`.

## Running on Kaggle

Kaggle's TPU VM notebooks ship `torch_xla` preinstalled (version pinned to
match the notebook's `torch` build) — do not `pip install torch_xla`
yourself, it will likely mismatch and break. Set accelerator to "TPU VM
v3-8" in notebook settings, then:

```python
!python scripts/smoke_test.py
!python scripts/whisper_tpu.py --audio path/to/clip.wav
```

## Structure

```
scripts/smoke_test.py   # tiny matmul on xla device, proves device + compile work
scripts/whisper_tpu.py  # Whisper transcription via transformers on xla device
requirements.txt        # non-torch_xla deps only (transformers, librosa, etc.)
```
