# Prior art: who's already done GPU→TPU porting, and how

Research pass 2026-10-08. No single "GPU-to-TPU porting methodology" survey
paper exists — confirmed by direct search, not inferred. The field is
fragmented into per-library point solutions, each solving one row of our
`SCOPE.md`, not a general methodology. Below is what's actually out there,
per thing we care about.

## 1. `rishabhindoria/GPU-to-TPU` (the repo asked about)

Real, public, created 2019, 7 stars, last touched 2019
(https://github.com/rishabhindoria/GPU-to-TPU).

**What it is:** a Colab tutorial, not a compat layer. Fine-tunes ResNet-50
on the flowers dataset using TF1's `TPUEstimator` (the pre-Keras,
pre-`TPUStrategy` API) against Google's own `tensorflow/tpu` reference
models repo.

**Problem it actually solves:** the input-pipeline-starves-the-TPU problem,
for the single easiest case there is (fixed-shape batched image
classification, TF-native, no custom kernels, no generation). They document
it themselves: an unoptimized `tf.data` pipeline leaves the TPU 95% idle
waiting on CPU preprocessing; switching to `parallel_map_and_batch` drops
that to 0.5% idle. This is our `SCOPE.md` Category F, solved for their one
case.

**Methodology:** TFRecord data staged on a GCS bucket (TPU can't read
Drive/local files directly — needs GCS + a service account with bucket
read/write roles), then `resnet_main.py` from Google's official repo handles
the `TPUEstimator` training loop. Claims 5-30x speedup vs a Colab K80 GPU
depending on scale, ~84%/81% val/test accuracy, up to 60% TPU MXU
utilization at 1M images.

**Code available:** yes, runs, but TF1/`TPUEstimator`-era (that API is
deprecated in TF2's Keras+`TPUStrategy` world). Addresses zero of our
Category A (CUDA-only kernel) or Category B (static-shape/generation)
problems — those didn't really exist as problems yet in 2019 (FlashAttention,
bitsandbytes, vLLM, and autoregressive-generation-at-scale all postdate this
repo). Useful as a historical data point, not as a template for anything in
`SCOPE.md` Category A/B/C.

## 2. PyTorch on TPU: two real competing approaches exist, not one

We picked `torch_xla` for the repo's first pass. Research turned up that
there's a second, newer, actively-promoted path — worth a conscious
decision, not a default:

### `torch_xla` (what we started with)
Lazy-tensor execution: ops build a graph, `torch_xla.sync()` /
`xm.mark_step()` is the barrier that compiles+runs it. GPU support exists in
`torch_xla` too (via an OpenXLA PJRT plugin) but is explicitly
"secondary status to TPU" in the project's own docs. Migration guide:
https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html

### `torchax` (google/torchax) — newer, different mechanism
A `torch.Tensor` subclass that holds a `jax.Array` and overrides
`__torch_dispatch__`: every PyTorch op gets swapped for its JAX equivalent
at dispatch time, rather than being traced into an XLA graph the way
`torch_xla` does it. Net effect: you write PyTorch, it executes as JAX, on
any hardware JAX supports (TPU, GPU, CPU). Repo:
https://github.com/google/torchax

**Why this matters for us:** `torchax` gets PyTorch code direct access to
the JAX ecosystem — including Pallas kernels like Splash Attention (below)
— without a separate reimplementation. `torch_xla` doesn't get you that for
free; a custom XLA op only exists if someone lowers it for `torch_xla`
specifically. This is a real fork in the road for how we close Category A
rows, not a settled question — flagging it, not switching unilaterally.

## 3. Fused attention on TPU — already solved, has a name

**Splash Attention**: JAX's FlashAttention-equivalent for TPU, built on
Pallas (JAX's TPU-kernel DSL, lets you write custom kernels while reusing
JAX syntax, explicit control over MXU/VMEM). Implementation based on the
original FlashAttention paper (arXiv:2205.14135), landed in JAX itself.
This directly answers `SCOPE.md` row A2 ("fused attention has no TPU path")
— the answer isn't "write one," it's "use Splash Attention," available
today via JAX (and reachable from PyTorch via `torchax`, see above).

## 4. Paged KV-cache attention on TPU — already solved, has a name

**Ragged Paged Attention (RPA), specifically v3**
(arXiv:2604.15464, "Ragged Paged Attention: A High-Performance and Flexible
LLM Inference Kernel for TPU"): this is vLLM's actual TPU attention kernel,
not a hypothetical. Handles variable-length sequences without padding waste
(ragged tensors) plus paged KV-cache memory management, adapted specifically
for TPU's memory hierarchy and DMA scheduling constraints (gathering
non-contiguous KV addresses is harder to schedule efficiently on TPU than
GPU — a real, named architectural difference, not just "different API").
RPA v3 fuses the KV-cache scatter into the kernel itself to hide that
latency, and compiles to three sub-kernels (prefill-only, decode-only,
mixed) for deployment flexibility. ~10% throughput gain over RPA v2.

**vLLM TPU unified backend** (https://vllm.ai/blog/2025-10-16-vllm-tpu,
package `tpu-inference`): as of this generation, vLLM TPU lowers *all*
models through JAX as the primary path — PyTorch models go through
`torchax` to the same JAX primitives/XLA IR, not a separate code path.
Architecture is SPMD (XLA-native), not the multi-worker GPU style. This
directly answers `SCOPE.md` row A4 ("LLM serving with paged KV cache") —
the fix isn't "port vLLM's CUDA kernel," it's "use vLLM's existing TPU
backend," which is what the row already said, now with the actual kernel
name and a live blog confirming it's shipped and open source.

## 5. A real, independently-documented TPU porting gotcha we hadn't listed

Found in passing (an ML-reproducibility paper, arXiv:2107.00821) while
searching for broader porting literature: TPU ops require `bfloat16` for
input parameters in some cases where GPU tolerates other dtypes, and at
least one specific op (upsampling) was reported to work correctly on the
forward pass on TPU but not the backward pass. Couldn't get clean body text
out of that PDF to pin down which op/framework version exactly — flagging
as a lead, not adding it to `SCOPE.md` as a confirmed row until we hit it
ourselves or verify the specific claim.

## 6. Environment gotcha confirmed on real hardware: plain `pip install
   tensorflow` has no TPU kernels on a Colab TPU VM runtime

Found while running `results/colab_run_2026-10-08.md`, not from a paper —
worth recording here since it's a second data point on top of the Kaggle
base-image pitfall and shapes what README's framework matrix can claim.

On a Colab TPU v5e-1 runtime: `torch_xla` and `jax` both find and use the
TPU fine (`libtpu.so` already present, matching `TPU_LIBRARY_PATH`). A
plain `pip install tensorflow` (2.21.0, exactly matching this runtime's own
`TPU_ML_PLATFORM_VERSION`) fails with `No OpKernel was registered to
support Op 'ConfigureDistributedTPU'` — `Registered devices: [CPU]`, i.e.
the installed wheel has no TPU op kernels compiled in at all, not a
version/env mismatch (ruled out both `TPUClusterResolver()` and
`TPUClusterResolver(tpu="local")`, same failure either way).

**Kaggle's TPU VM notebooks ship a TPU-enabled `tensorflow` preinstalled**
(per `reference_kaggle_api`-style notes and this repo's own
`examples/tensorflow/requirements.txt` comment) — that's a different,
Google-built TF wheel than what a plain `pip install tensorflow` gets you
on Colab. README's framework matrix claim "TF has native TPU support, no
compat layer needed" is true on Kaggle's image specifically, **not**
reproduced by a generic pip install elsewhere. Not pursued further this
session (needs either a TPU-specific TF package/build not identified yet,
or relying on whatever the host notebook platform preinstalls) — tracked
as an open question, not a SCOPE.md row (it's an installation/environment
gap, not a GPU-only code pattern).

## What this changes in `SCOPE.md`

- Row A2 (fused attention): has a concrete, shipped answer now — Splash
  Attention — not just "use XLA's own fused attention" as a vague gesture.
- Row A4 (vLLM paged KV cache): has a concrete, shipped answer — vLLM's
  TPU backend + RPA v3 — same situation, already exists, just needs
  adopting, not building.
- Open decision, not yet made: whether `torch_xla` or `torchax` is the
  right default bridge for *future* Category A rows (custom kernels,
  quantization, etc.), since `torchax` reaches JAX/Pallas kernels directly
  and `torch_xla` doesn't. Need the user's call before switching the
  project's stated approach.
