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

## 2. PyTorch on TPU: three real paths exist, not one

We picked `torch_xla` for the repo's first pass. Research turned up a
second, newer path (`torchax`), and — as of this round (2026-10-08,
verified via web search, not inferred) — a **third**, Google-announced
but not-yet-public native path. Worth a conscious decision, not a default:

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

### `TorchTPU` — Google's announced native path, not yet public/GA

Confirmed real via Google Developers Blog ("TorchTPU: Running PyTorch
Natively on TPUs at Google Scale") and PyTorch's own Sept/Oct 2026
conference material — not a rumor, but also **not something we can adopt
today**: "Once TorchTPU is public it will replace PyTorch/XLA," meaning
`torch_xla` remains the current supported path until that happens.

Mechanism, for when it matters: built on PyTorch's `PrivateUse1` device
extension (the same mechanism third-party backends like `cuda`-alternatives
use), not a compiler-bridge/lazy-tensor trace like `torch_xla` — maps
PyTorch ops directly to StableHLO, eager-first execution model, goal is
`tensor.to('tpu')` feeling like `tensor.to('cuda')` with minimal code
changes. If/when it ships, our `get_torch_device()`/`device_kind()`
dispatch (`src/tpu_as_gpu/device.py`) is exactly the chokepoint that would
need a third branch — not a rewrite, one dispatch function already isolates
this decision from everything else in the library.

**Action taken:** none yet, correctly — nothing to integrate against until
it's public. Tracked here so the next person doesn't re-discover this from
scratch. Sources: https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/,
https://github.com/pytorch/xla/blob/master/README.md

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

## 7. Whisper on real TPU: ran end to end, hit a real eager-vs-XLA slicing
   divergence in `transformers` itself

`examples/torch_xla/whisper_tpu.py` (`openai/whisper-base`) run on the same
Colab TPU v5e-1 session as §6, 2026-10-08. Full result in
`results/colab_run_2026-10-08.md`.

First call failed in `transformers`' own
`WhisperGenerationMixin._retrieve_segment` (not our code):

```
single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
RuntimeError: Value out of range (expected to be in range of [-1, 0], but got -2)
```

Root cause, confirmed not an environment issue: the test input (a 2s pure
440Hz sine tone, not real speech) decoded to a degenerate 1-token
`seek_sequence`. In eager PyTorch, `tensor[-2:]` on a length-1 tensor
silently clamps and returns the whole tensor — standard Python/NumPy slice
semantics. Under `torch_xla`'s compiled execution, the same negative-start
slice throws instead of clamping. This is a genuine GPU(eager)-vs-TPU(XLA)
semantic divergence, surfaced by real third-party code
(`transformers`'s own generation loop) hitting a degenerate edge case, not
something contrived to prove a point. Passing `min_new_tokens=5` to force
a longer generated sequence avoided the degenerate case and let generation
complete (text was still garbage — `"you you you you you"` — expected,
since the input wasn't real speech, that's an input-quality artifact, not
a correctness bug).

Doesn't get its own SCOPE.md row — it's a transformers-library-internal
edge case triggered by a specific (degenerate) input shape, not a GPU-only
*pattern* like the rest of Category B. Worth remembering if any future row
hits an unexplained XLA slicing error: check whether eager would have
silently clamped where XLA throws.

Also a real, quantified data point for SCOPE row B1 (growing KV cache
recompiles): first-call `generate()` took 253.74s — almost entirely XLA
compile time for the autoregressive generation loop, the exact cost
`StaticKVCache` exists to avoid. Not yet A/B'd against a fixed-buffer
generation loop in this repo — that comparison is the natural next
Category A/Whisper follow-up, not done this session.

## 8. Tried to actually adopt Splash Attention on real hardware — blocked by
   a Mosaic version mismatch, not our code

With live TPU access (Colab v5e-1, same session as §6/§7), went past
"solution identified" and tried to actually call Splash Attention, to see
if rows A2/A3 could move to `done`.

The API is real and matches the signatures found by introspection
(`jax==0.7.2`, `jaxlib==0.7.2`):

```python
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

mask = mask_lib.MultiHeadMask(masks=[mask_lib.CausalMask(shape=(seq_len, seq_len))] * num_heads)
kernel = splash.make_splash_mha_single_device(mask=mask)
out = jax.jit(kernel)(q, k, v)  # q/k/v: (num_heads, seq_len, head_dim)
```

The module even ships its own reference implementation
(`splash.make_attention_reference`) for correctness-checking the kernel
output — would have been the exact right A/B check.

It didn't run. `jax.jit(kernel)(q, k, v)` (q/k/v: `(4, 128, 64)`, bf16)
failed with:

```
XlaRuntimeError: INTERNAL: Failed to deserialize the Mosaic module:
INTERNAL: Unsupported version: expected <= 7 but got 8
```

This is a version-skew bug between this JAX build's Pallas/Mosaic compiler
(which serializes kernels at IR version 8) and the installed
`libtpu==0.0.21.1` runtime's Mosaic deserializer (which only accepts
`<= 7`) — confirmed via `pip show jax jaxlib libtpu` on the same session:
`jax==0.7.2`, `jaxlib==0.7.2`, `libtpu==0.0.21.1`. Not a bug in Splash
Attention's API or in how it was called — the kernel never got far enough
to run, it failed at module deserialization.

**Deliberately not attempted**: upgrading/pinning `libtpu` to chase a
matching version. This session's `torch_xla` and plain `jax` TPU access
(§6, §7, the B/H verification run) all depend on the exact `libtpu`
already proven working — risking a version bump mid-session to fix one
Pallas kernel would have put all of that already-verified work at risk for
an unconfirmed payoff. `ragged_paged_attention` (SCOPE row A4) lives under
the same `jax.experimental.pallas.ops.tpu` package and almost certainly
hits the identical Mosaic deserialization path — inferred, not
independently re-tested, to avoid the same risk twice for the same root
cause.

**Status for A2/A3/A4**: still "solution identified, not adopted" — but
now with the real API confirmed correct and the actual blocker identified
precisely (a specific version-skew bug, not an unknown). Next attempt
should happen on a fresh environment where `jax`/`jaxlib`/`libtpu` can be
pinned to mutually compatible versions from the start, rather than
patching a running session.

## 9. Quantization on TPU: AQT picked and verified on real hardware

SCOPE row A6 (4-bit/8-bit quantization): `bitsandbytes` is CUDA-only, no
TPU path, no near-term fix. The TPU-native answer is **AQT**
(`google/aqt`, PyPI package `aqtp`) — quantized `jax.lax.dot_general`,
built and maintained by Google specifically for TPU training/inference,
the closest thing to a direct `bitsandbytes` equivalent that actually runs
on this hardware.

Picked over alternatives considered: `optax` has no quantization support
of its own (just optimizers); `torchao` (PyTorch's own quantization
library) targets CUDA, no TPU/XLA backend found. No PyTorch/torch_xla-
native quantization path was found at all — AQT is JAX-only, which means
Category A's quantization story currently forces a JAX dependency even
for an otherwise-torch_xla workload. Flagging this as a second data point
for the open `torch_xla`-vs-`torchax` decision (§2): `torchax` would let a
PyTorch model reach AQT directly, `torch_xla` wouldn't.

**Verified on real TPU hardware** (Colab v5e-1, 2026-10-08):
`aqt.jax.v2.aqt_dot_general.dot_general_make(lhs_bits=8, rhs_bits=8)` run
on a 512×256 @ 256×512 matmul, max relative error 0.91% vs the fp32
reference — a real, measured number, not a claim from AQT's own docs.
Wrapped as `tpu_as_gpu.quantization.quantized_matmul`, full result in
`results/colab_run_2026-10-08.md`.

**Does not solve SCOPE row A7** (8-bit optimizers, `Adam8bit`-style) —
that's optimizer*-state* compression (the optimizer's own momentum/
variance buffers stored at lower precision), a different problem from
quantizing activations/weights for the forward/backward matmul. AQT
doesn't address it; still an open row.

## 10. Train-then-infer lifecycle verified on real TPU hardware

Separately from any single SCOPE row: confirmed the basic premise of the
whole repo — that a model can actually be *trained* on TPU via
`torch_xla`, not just run inference — since nothing before this session
had exercised a real backward pass + optimizer step on real hardware.

A 2-layer MLP (`nn.Linear(10,32) -> ReLU -> nn.Linear(32,2)`), 5 SGD steps
on a Colab v5e-1: loss went 0.705 -> 0.692 (monotonic decrease, confirming
gradients actually flowed and the optimizer actually updated weights, not
just that the code ran without erroring), then `model.eval()` +
`torch.no_grad()` inference on the same device in the same process.
Example: `examples/torch_xla/train_infer_demo.py`. Full numbers in
`results/colab_run_2026-10-08.md`.

## 11. External review (2026-10-08): four claims checked against primary
    sources, three confirmed and actioned, one confirmed-but-nuanced

A pasted external critique of `SCOPE.md` made several specific factual
claims. Per this project's verify-before-acting discipline, each
actionable one was checked against a primary source before touching the
repo — not taken on faith, not dismissed either.

**Confirmed, actioned:**

1. **AQT is end-of-life, Qwix (`google/qwix`) is Google's own replacement.**
   Confirmed via the `qwix` repo/docs directly (not the critique's framing
   of it) — "AQT is reaching end-of-life... Qwix['s] design was inspired by
   AQT." Real difference, not just a rename: AQT has quantized *training*
   (quantized forward+backward), Qwix's QAT is fake-quantization only, so
   it doesn't speed up training the way AQT's int8 `dot_general` does.
   **Not swapped yet** — `tpu_as_gpu.quantization.quantized_matmul`'s AQT
   path is verified on real TPU hardware (0.91% rel. error, §9); Qwix is
   not. Swapping a verified dependency for an unverified one on a maybe
   would violate this project's own "verify before acting" rule. Sources:
   https://github.com/google/qwix, https://qwix.readthedocs.io/

   **Follow-up attempted 2026-10-08 (Colab, v5e-1), blocked — not our
   bug.** `pip install qwix` (0.1.8) declares `flax>=0.12.0` as its own
   dependency floor. Installing the latest flax that satisfies that floor
   (0.12.10) still fails on `import qwix`:
   ```
   ImportError: cannot import name 'BaseConfigContext' from 'flax.typing'
   ```
   raised inside `qwix/__init__.py` → `qwix.contrib` → `qwix.contrib.gptq`
   → `flax.nnx.variablelib`, which does `from flax.typing import
   BaseConfigContext` unconditionally — that name doesn't exist anywhere
   in `flax.typing` on 0.12.10. Confirmed `BaseConfigContext` isn't a
   version-skew fluke of this environment: checked `dir(flax.typing)`
   directly, it's simply absent. Tried pinning flax back to 0.11.2 (below
   qwix's own stated floor) as a sanity check — different failure
   (`ImportError: cannot import name 'ndtr' from 'jax._src.lax.special'`,
   an unrelated transitive pin conflict), confirming 0.11.x isn't a usable
   downgrade path either. Same failure class as the Splash Attention
   Mosaic-version-skew block (§8): a real upstream break inside qwix's own
   declared-compatible dependency range, not something fixable from this
   repo. Qwix verification stays open; AQT remains the only working path.
   Environment left on `flax==0.12.10` (qwix's own stated minimum)
   afterward. Session ended on a Colab TPU-quota wall before a retry was
   possible.

2. **TorchTPU exists and is Google-announced** — see §2 above, added as a
   third branch in the `torch_xla`/`torchax` decision, explicitly not
   adopted since it isn't public yet.

3. **TPU7x (Ironwood) has native FP8 hardware.** Confirmed via Google
   Cloud's own TPU7x docs: "FP8 peak TFlops are emulated for v4 and v5p,
   but natively supported for Ironwood" — 4614 TFLOPS/chip peak theoretical.
   This means SCOPE.md row A ("fp8 training... no equivalent library") was
   **generation-dependent, not universally true** — fixed in SCOPE.md.
   Our actual test hardware this session was a v5e-1, not TPU7x/Ironwood —
   whether v5e-1 has any FP8 path (native or emulated) is **not yet
   checked**, left as an open question rather than assumed either way.
   Source: https://docs.cloud.google.com/tpu/docs/tpu7x

4. **Helion (PyTorch's kernel-authoring DSL) compiles to Pallas on TPU,
   with real measured numbers** — 838 TFLOPs (~79% MFU) on a flash
   attention workload on TPU v7, per PyTorch's own blog with a full
   methodology writeup, not a marketing number. This is a more
   PyTorch-native alternative to hand-writing Pallas directly for the
   Triton→Pallas row (SCOPE.md Category A) — added as a note on that row.
   Source: https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/

**Confirmed, but the critique overstated it:**

5. **Dynamic shapes**: the critique said SCOPE.md's "TPU compiles for a
   fixed shape" framing is outdated because `torch_xla` added dynamic-shape
   support. True, but incompletely — checked `pytorch/xla`'s own dynamic
   shape docs: only **bounded** dynamic shape is supported (a dimension can
   vary but must have a known static upper bound, which is still
   allocated), not **unbounded** dynamic shape (arbitrary growth, no bound
   — that's explicitly GPU-only, "out of scope of the current design
   phase" per PyTorch/XLA's own docs). So: bucketing to a fixed set of
   bounds is no longer the *only* option, but TPU still fundamentally needs
   a bound — this isn't the architectural constraint disappearing, it's
   PyTorch/XLA formalizing what bucketing was already doing by hand.
   SCOPE.md Category B wording adjusted to say this precisely instead of
   either the original absolute claim or the critique's overcorrection.
   Source: https://docs.pytorch.org/xla/release/r2.6/learn/dynamic_shape.html

**Explicitly not actioned — see `RESEARCH_DIRECTION.md`:** the critique's
remaining ~35 points (capability registry, strict no-fallback policy,
operator-coverage database, TPU-generation abstraction, cost/performance
benchmarking framework, portability scoring, etc.) are a legitimate
research-maturity roadmap, not verifiable-or-not factual claims — recorded
as forward direction in `RESEARCH_DIRECTION.md` rather than built now.
Building a capability registry for a library with ~10 functions is
premature abstraction; the registry earns its keep once there are enough
dispatch points to actually need one.

## What this changes in `SCOPE.md`

- Row A2 (fused attention): has a concrete, shipped answer now — Splash
  Attention — not just "use XLA's own fused attention" as a vague gesture.
- Row A4 (vLLM paged KV cache): has a concrete, shipped answer — vLLM's
  TPU backend + RPA v3 — same situation, already exists, just needs
  adopting, not building.
- Row A6 (quantization): AQT verified, Qwix noted as the upstream
  direction, not yet adopted (§11.1).
- Row A (fp8): no longer flatly "no equivalent library" — TPU7x/Ironwood
  has native hardware support, earlier gens emulate it (§11.3).
- Row A (Triton): Helion→Pallas added as the more PyTorch-native option
  alongside hand-written Pallas (§11.4).
- Category B intro: "fixed shape" wording corrected to "bounded shape"
  (§11.5) — bucketing is still the mechanism, just now also what
  `torch_xla`'s own bounded-dynamic-shape support does under the hood.
- Open decision, not yet made: whether `torch_xla`, `torchax`, or (once
  public) `TorchTPU` is the right default bridge for *future* Category A
  rows. Need the user's call before switching the project's stated
  approach.
