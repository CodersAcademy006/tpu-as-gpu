# Scope: GPU-only processes to port to TPU

**29 rows total (2026-10-08): 17 done / 3 solution identified, not adopted
/ 1 wontfix / 8 not started.**
Done = A1/Whisper, A6/quantization, B (5/5), D (2/2), E (3/3), F (2/2),
G (1/1), H (2/2). A1, A6, B1-B4, E1/E3, F1, and H1-H2 are **verified on
real TPU hardware** (Colab, TPU v5e-1, 2026-10-08 — see
`results/colab_run_2026-10-08.md`); D1-D2 are dispatch-only (the pick is
right, the loop-level behavior it feeds into isn't exercised by this run).
Not started = all of C (3, needs real multi-device hardware, not available
this session — a single-core v5e-1 isn't enough to test collectives) plus
5 of A's remaining rows (8-bit optimizers, custom kernels, fp8, Triton —
each a standalone adoption effort, not a dispatch function).

Everything below works on a normal NVIDIA GPU and either breaks, doesn't
exist, or silently falls back to CPU on a TPU. Grouped by *why* it's broken,
not by framework, since the fix pattern is the same within a group. Each row
becomes one helper script/module under the matching framework dir once we
tackle it.

Status values: `not started` / `in progress` / `done` / `wontfix (reason)`.
`done` means the code exists and passes its CPU-logic tests under
`tests/` — Categories B/G/H are pure control-flow/bookkeeping fixes, no
TPU hardware needed to verify the logic itself. It does NOT mean verified
against real TPU numbers end to end; that's still pending (see
`results/README.md`).

See `research/prior_art.md` for what's already been solved upstream (don't
reimplement a kernel that already ships) and the open architectural
decision it surfaced: `torch_xla` vs `torchax` vs (once public) `TorchTPU`
as the PyTorch-on-TPU bridge for Category A rows — not yet decided. See
`RESEARCH_DIRECTION.md` for what a research-maturity version of this
project needs beyond the 29 rows (capability registry, no-silent-fallback
policy, numerical/performance benchmarking) and why most of it isn't built
yet — this is a ~10-function library, not a mature compat layer.

The installable library living under `src/tpu_as_gpu/` is what the
"done" rows below actually point at; `examples/<framework>/` holds
runnable scripts for whatever still needs real TPU hardware to verify.

## A. CUDA-only kernels — no TPU equivalent exists, needs a different library entirely

These aren't "install the right package" gaps — the kernel is written in
CUDA C++ and only runs on NVIDIA hardware. Fix is always "swap to a
different implementation of the same idea," never "port the kernel."

| Process | GPU library | TPU replacement | Status |
|---|---|---|---|
| Whisper/speech-to-text inference | CTranslate2 (faster-whisper) | `transformers` Whisper on torch_xla | done, **verified on real TPU** (`examples/torch_xla/whisper_tpu.py`, Colab v5e-1 2026-10-08 — see `research/prior_art.md` §7 for a real eager-vs-XLA slicing gotcha hit along the way, and the 253.74s first-compile cost as a live data point for row B1) |
| Fused attention | FlashAttention / FlashAttention-2 | **Splash Attention** — JAX's FlashAttention equivalent, built on Pallas, already lands in JAX itself. See `research/prior_art.md` §3 | not started — API confirmed correct on real hardware (`research/prior_art.md` §8), execution blocked by a Mosaic IR version mismatch (`jax`/`jaxlib` 0.7.2 vs `libtpu` 0.0.21.1), not our code |
| Memory-efficient attention | xformers | same as Splash Attention row | not started — same blocker as the row above |
| LLM inference serving w/ paged KV cache | vLLM (CUDA PagedAttention) | **Ragged Paged Attention v3** inside vLLM's TPU backend (`tpu-inference`), shipped and open source. See `research/prior_art.md` §4 | not started — same Mosaic blocker expected (inferred, not independently re-tested — see `research/prior_art.md` §8) |
| Quantized local inference | llama.cpp / GGUF | no TPU path at all — needs re-export to a TPU-native format or a different runtime | not started |
| 4-bit/8-bit quantization | bitsandbytes | **AQT** (`google/aqt`, pip `aqtp`) — JAX-native quantized `dot_general`. See `research/prior_art.md` §9. AQT is end-of-life upstream; **Qwix** (`google/qwix`) is Google's own replacement, confirmed real but not yet verified on our hardware — see §11.1 | done, **verified on real TPU** (`tpu_as_gpu.quantization.quantized_matmul`, int8, 0.91% max rel. error vs fp32) — JAX only, no torch_xla-native path identified yet. Qwix migration is a follow-up, not done — swapping a verified dep for an unverified one isn't justified yet |
| 8-bit optimizers (Adam8bit, etc.) | bitsandbytes | standard `optax`/torch_xla optimizer, no 8-bit equivalent on TPU | not started — AQT (row above) is activation/weight quantization, not optimizer-state compression; a different problem, not solved by the same library |
| Custom CUDA ops (NMS, ROI align, deformable conv) | torchvision CUDA kernels | CPU fallback (slow) or XLA-native reimplementation | not started |
| Arbitrary custom CUDA extensions | `torch.utils.cpp_extension` | full rewrite as XLA op / Pallas kernel, case by case | not started |
| fp8 training | NVIDIA Transformer Engine (Hopper-only) | generation-dependent, not a flat "no" — TPU7x/Ironwood has **native FP8 hardware** (confirmed, Google Cloud docs, §11.3); v4/v5p emulate it in software. Our test hardware (v5e-1) FP8 status not yet checked | not started — needs checking what this session's actual hardware (v5e-1) supports before picking a path |
| Custom GPU kernels via Triton | OpenAI Triton | no mainstream TPU backend for Triton itself — options are hand-written Pallas (JAX), or **Helion** (PyTorch's kernel DSL, compiles to Pallas, real measured 838 TFLOPs flash-attention on TPU v7 — §11.4), more PyTorch-native than raw Pallas | not started |
| Physics sim for RL | Isaac Gym (CUDA/PhysX) | no TPU path — would need a JAX-native sim (e.g. Brax) instead | wontfix (different sim entirely, not a port) |

## B. Static-shape / compilation constraints — architectural, not a missing library

TPU (XLA) compiles a graph for a *bounded* shape — a dimension can vary but
needs a known static upper bound, which is still what gets allocated; it
cannot grow truly unbounded the way GPU eager execution allows (confirmed:
`torch_xla` added bounded dynamic-shape support, but unbounded dynamism is
explicitly out of scope of its design — `research/prior_art.md` §11.5,
not inferred). Anything that changes shape at runtime without a declared
bound either recompiles (slow) or breaks. Fix pattern: pad/bucket to a
fixed set of bounds, or restructure control flow to avoid data-dependent
branching — bucketing by hand and `torch_xla`'s bounded-dynamic-shape API
are two ways to express the same underlying bound, not competing fixes.

| Process | Why it breaks on TPU | Fix pattern | Status |
|---|---|---|---|
| Autoregressive generation w/ growing KV cache | every new token = new shape = recompile | static-size KV cache buffer + masking, not append-and-grow | done, **verified on real TPU** (`tpu_as_gpu.compat.StaticKVCache`, torch/numpy; JAX uses `lax.dynamic_update_slice_in_dim` directly) |
| Variable-length sequence batching | each batch shape triggers a new compile | bucket into fixed-length bins before batching | done (`tpu_as_gpu.compat.bucket_length`) — pure logic, no device needed, verified on real TPU run anyway |
| `if tensor_value > x:` branching on data | can't trace a Python `if` on a traced value | `jax.lax.cond` / `torch.where`, no data-dependent Python control flow | done, **verified on real TPU** (`tpu_as_gpu.compat.where`, dispatches per framework) |
| `.item()` / `.cpu()` mid-graph sync points | forces a device sync every call — fine occasionally on GPU, very expensive on TPU if done per-step | move sync points outside hot loops, batch them | done, **verified on real TPU** (`tpu_as_gpu.compat.DeferredSync`) |
| Early-stopping / break-on-condition loops | can't break a traced loop on a data value | run fixed iteration count, mask out finished items | done, **verified on real TPU** (same `tpu_as_gpu.compat.where`, keyed by a done-mask instead of a condition) |

## C. Distributed / multi-device collectives

| Process | GPU convention | TPU convention | Status |
|---|---|---|---|
| Multi-device training backend | `torch.distributed` w/ NCCL | `torch_xla`'s own distributed backend (`xla` backend, not `nccl`) | not started |
| ZeRO optimizer state sharding | DeepSpeed (CUDA-memory-model assumptions) | DeepSpeed's TPU support is experimental/partial — likely needs a different sharding approach (FSDP via torch_xla, or JAX's native sharding) | not started |
| Device mesh / parallelism strategy | `torch.cuda.set_device`, NCCL process groups | JAX `Mesh`/`shard_map`, or torch_xla SPMD | not started |

## D. Precision & numerics conventions

| Process | GPU convention | TPU convention | Status |
|---|---|---|---|
| Mixed-precision training | fp16 + `torch.cuda.amp.GradScaler` (loss scaling needed, fp16 has narrow range) | bf16, no loss scaling needed (wider dynamic range) — `GradScaler` itself is CUDA-only and won't run | done (`tpu_as_gpu.precision.autocast_dtype`) — dispatch only, loop-level behavior still needs hardware to verify |
| `torch.compile` backend | Inductor (generates Triton/CUDA) | `backend="openxla"` — different compile path entirely | done (`tpu_as_gpu.precision.compile_backend`) — dispatch only, compile itself still needs hardware to verify |

## E. Memory management & profiling tooling

| Process | GPU tool | TPU equivalent | Status |
|---|---|---|---|
| Utilization/memory monitoring | `nvidia-smi`, `pynvml` | `torch_xla` metrics report / `xm.get_memory_info()`, or Cloud TPU profiler | done, **verified on real TPU** (`tpu_as_gpu.profiling.get_memory_info`) |
| Deep kernel profiling | Nsight Systems/Compute | XLA/TPU profiler (`torch_xla` or JAX profiler, different trace format) | done — documented pattern, not code, see `PERFORMANCE.md` |
| Clearing allocator cache | `torch.cuda.empty_cache()` | no direct equivalent — XLA manages its own allocator differently | done, **verified on real TPU** (`tpu_as_gpu.profiling.clear_cache`, honestly returns `False` on XLA instead of pretending to clear anything) |

## F. Data loading pipeline conventions

| Process | GPU convention | TPU convention | Status |
|---|---|---|---|
| Pinned-memory async transfer | `pin_memory=True`, `.to(device, non_blocking=True)` | `torch_xla.distributed.parallel_loader.MpDeviceLoader` instead | done, **verified on real TPU** — no `tpu_as_gpu` wrapper needed, `MpDeviceLoader` is already the drop-in fix, see `PERFORMANCE.md` |
| Preprocessing pipeline | arbitrary `DataLoader` + CPU workers, variable batch content | prefer pre-batched, fixed-shape input; `tf.data` is the most TPU-native option regardless of training framework | done — documented convention, not code, see `PERFORMANCE.md` |

## G. Debugging / dev workflow

| Process | GPU behavior | TPU behavior | Status |
|---|---|---|---|
| Inline `print()`/breakpoint on tensor values mid-forward-pass | works immediately, eager execution | shows an unresolved traced value until a sync point (`xm.mark_step()` / `block_until_ready()`) is hit | done — see `DEBUGGING.md`, documented pattern not a script |

## H. Model-code hygiene (not a feature gap, just bad habits that assume CUDA)

| Process | Bad pattern | Fix | Status |
|---|---|---|---|
| Device placement | hardcoded `.cuda()` calls scattered through model code instead of a `device` variable | sweep + replace with `xm.xla_device()` / passed-in device, one-time fix per model ported | done, **verified on real TPU** (`tpu_as_gpu.device.get_torch_device` resolved to `xla:0` on Colab v5e-1) |
| Seeding/determinism | `torch.cuda.manual_seed` | XLA has its own RNG handling — needs separate seeding path if determinism matters | done, **verified on real TPU** (`tpu_as_gpu.seeding.seed_everything`) |

## Priority order (suggested, not fixed)

1. **A** — A1/Whisper and A6/quantization done and verified. A2-A4 (Splash Attention, Ragged Paged Attention) blocked on a Mosaic version-skew bug (`research/prior_art.md` §8), needs a fresh environment with matched `jax`/`jaxlib`/`libtpu` to retry. Remaining 5 rows (8-bit optimizers, custom CUDA ops, fp8, Triton/Helion, GGUF local inference) are standalone adoption efforts, picked up as needed.
2. **B, D, E, F, G, H** — done and mostly real-TPU-verified (D is dispatch-only). No further action needed unless a row's claim turns out wrong in practice.
3. **C** — blocked on real multi-device hardware (this session only had a single-core v5e-1); revisit when multi-device access exists.

## Out of scope

- **MLX** — see `UNSUPPORTED_MLX.md`. No TPU backend exists, not fixable here.
- Anything requiring NVIDIA-specific hardware features with no TPU analog at
  all (e.g. Transformer Engine's Hopper-specific fp8 path) stays "wontfix"
  unless a TPU-native equivalent gets named here first.

## I. Inherent accelerator limits — GPU can't do these either, not a TPU gap

Not part of the 29-row count above. The 29 rows are things GPU *does* that
TPU needs a replacement for. These are things **neither** GPU nor TPU does
well — both are SIMT/SIMD throughput engines, not general sequential
processors — so there's no GPU-side version to port from and nothing to
build here. Listed once so "what can't this hardware class do" has one
answer instead of getting re-asked per model ported.

| Limit | Why neither GPU nor TPU is good at it |
|---|---|
| Branch-heavy / divergent control flow | SIMT: all lanes in a warp run the same instruction; a data-dependent `if` serializes the divergent lanes instead of skipping them |
| Pointer-chasing / irregular memory (linked lists, graph traversal, trees w/o locality) | Latency-bound random access; both chips are built around high-bandwidth *sequential* memory access, not cache-friendly pointer hopping |
| Low-latency single sequential dependency chain | Both hide latency by overlapping thousands of independent threads; a lone chain with no parallelism to overlap just pays kernel-launch + sync overhead for nothing |
| OS-level work: syscalls, filesystem, network stack, interrupts | Neither runs an OS or owns a device; always mediated by the host CPU, no interrupt controller on either |
| Hard real-time control loops (microsecond-deterministic latency) | Scheduling on both is throughput-optimized, not latency-guaranteed; jitter is a feature of the scheduler, not a bug to patch |
| Small problem sizes | Kernel-launch + host-device transfer overhead on both dwarfs the compute for small N; CPU just wins below some size threshold |
| Exact bit-for-bit numeric determinism across runs | Floating-point reduction order depends on how work got split across cores/threads on both; same code, different thread/block count, slightly different rounding |
| Deep/dynamic recursion, arbitrary heap allocation per thread | Limited per-thread stack on both, no general-purpose `malloc` pattern either is built for |
| OLTP-style transactional workloads (many small random writes + locking) | No transaction/locking model on either; this is a CPU-and-database problem, not an accelerator problem |

Nothing on this list gets a helper script — there's no "TPU-native fix" for
a limit the GPU shares. If a future need turns out to actually be
TPU-specific (not shared with GPU), move it up into Category A-H instead.
