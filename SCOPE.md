# Scope: GPU-only processes to port to TPU

**29 rows total (2026-10-08): 10 done / 1 in progress / 3 solution
identified, not adopted / 1 wontfix / 14 not started.**
Done = B (5/5), D (2/2), G (1/1), H (2/2) — all code-level, CPU-tested,
pending real-TPU verification. Not started = all of C/E/F plus 6 of A's 11
rows; those need either real hardware to measure or a larger adoption
effort than a dispatch function.

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
reimplement a kernel that already ships) and one open architectural
decision it surfaced: `torch_xla` vs `torchax` as the PyTorch-on-TPU bridge
for Category A rows — not yet decided.

The installable library living under `src/tpu_as_gpu/` is what the
"done" rows below actually point at; `examples/<framework>/` holds
runnable scripts for whatever still needs real TPU hardware to verify.

## A. CUDA-only kernels — no TPU equivalent exists, needs a different library entirely

These aren't "install the right package" gaps — the kernel is written in
CUDA C++ and only runs on NVIDIA hardware. Fix is always "swap to a
different implementation of the same idea," never "port the kernel."

| Process | GPU library | TPU replacement | Status |
|---|---|---|---|
| Whisper/speech-to-text inference | CTranslate2 (faster-whisper) | `transformers` Whisper on torch_xla | in progress (`torch_xla/whisper_tpu.py`) |
| Fused attention | FlashAttention / FlashAttention-2 | **Splash Attention** — JAX's FlashAttention equivalent, built on Pallas, already lands in JAX itself. See `research/prior_art.md` §3 | not started (solution identified, needs adopting) |
| Memory-efficient attention | xformers | same as Splash Attention row | not started (solution identified, needs adopting) |
| LLM inference serving w/ paged KV cache | vLLM (CUDA PagedAttention) | **Ragged Paged Attention v3** inside vLLM's TPU backend (`tpu-inference`), shipped and open source. See `research/prior_art.md` §4 | not started (solution identified, needs adopting) |
| Quantized local inference | llama.cpp / GGUF | no TPU path at all — needs re-export to a TPU-native format or a different runtime | not started |
| 4-bit/8-bit quantization | bitsandbytes | no TPU path — needs `optax`/XLA-native quantization or skip quantization on TPU | not started |
| 8-bit optimizers (Adam8bit, etc.) | bitsandbytes | standard `optax`/torch_xla optimizer, no 8-bit equivalent on TPU | not started |
| Custom CUDA ops (NMS, ROI align, deformable conv) | torchvision CUDA kernels | CPU fallback (slow) or XLA-native reimplementation | not started |
| Arbitrary custom CUDA extensions | `torch.utils.cpp_extension` | full rewrite as XLA op / Pallas kernel, case by case | not started |
| fp8 training | NVIDIA Transformer Engine (Hopper-only) | no equivalent library — TPU has its own fp8 path, different API | not started |
| Custom GPU kernels via Triton | OpenAI Triton | no mainstream TPU backend — rewrite in Pallas (JAX) | not started |
| Physics sim for RL | Isaac Gym (CUDA/PhysX) | no TPU path — would need a JAX-native sim (e.g. Brax) instead | wontfix (different sim entirely, not a port) |

## B. Static-shape / compilation constraints — architectural, not a missing library

TPU (XLA) compiles a graph for a fixed shape; GPU (eager execution) doesn't
care. Anything that changes tensor shape at runtime either recompiles
(slow) or breaks. Fix pattern: pad/bucket to fixed shapes, or restructure
control flow to avoid data-dependent branching.

| Process | Why it breaks on TPU | Fix pattern | Status |
|---|---|---|---|
| Autoregressive generation w/ growing KV cache | every new token = new shape = recompile | static-size KV cache buffer + masking, not append-and-grow | done (`tpu_as_gpu.compat.StaticKVCache`, torch/numpy; JAX uses `lax.dynamic_update_slice_in_dim` directly) |
| Variable-length sequence batching | each batch shape triggers a new compile | bucket into fixed-length bins before batching | done (`tpu_as_gpu.compat.bucket_length`) |
| `if tensor_value > x:` branching on data | can't trace a Python `if` on a traced value | `jax.lax.cond` / `torch.where`, no data-dependent Python control flow | done (`tpu_as_gpu.compat.where`, dispatches per framework) |
| `.item()` / `.cpu()` mid-graph sync points | forces a device sync every call — fine occasionally on GPU, very expensive on TPU if done per-step | move sync points outside hot loops, batch them | done (`tpu_as_gpu.compat.DeferredSync`) |
| Early-stopping / break-on-condition loops | can't break a traced loop on a data value | run fixed iteration count, mask out finished items | done (same `tpu_as_gpu.compat.where`, keyed by a done-mask instead of a condition) |

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
| Utilization/memory monitoring | `nvidia-smi`, `pynvml` | `torch_xla` metrics report / `xm.get_memory_info()`, or Cloud TPU profiler | not started |
| Deep kernel profiling | Nsight Systems/Compute | XLA/TPU profiler (`torch_xla` or JAX profiler, different trace format) | not started |
| Clearing allocator cache | `torch.cuda.empty_cache()` | no direct equivalent — XLA manages its own allocator differently | not started |

## F. Data loading pipeline conventions

| Process | GPU convention | TPU convention | Status |
|---|---|---|---|
| Pinned-memory async transfer | `pin_memory=True`, `.to(device, non_blocking=True)` | `torch_xla.distributed.parallel_loader.MpDeviceLoader` instead | not started |
| Preprocessing pipeline | arbitrary `DataLoader` + CPU workers, variable batch content | prefer pre-batched, fixed-shape input; `tf.data` is the most TPU-native option regardless of training framework | not started |

## G. Debugging / dev workflow

| Process | GPU behavior | TPU behavior | Status |
|---|---|---|---|
| Inline `print()`/breakpoint on tensor values mid-forward-pass | works immediately, eager execution | shows an unresolved traced value until a sync point (`xm.mark_step()` / `block_until_ready()`) is hit | done — see `DEBUGGING.md`, documented pattern not a script |

## H. Model-code hygiene (not a feature gap, just bad habits that assume CUDA)

| Process | Bad pattern | Fix | Status |
|---|---|---|---|
| Device placement | hardcoded `.cuda()` calls scattered through model code instead of a `device` variable | sweep + replace with `xm.xla_device()` / passed-in device, one-time fix per model ported | done (`tpu_as_gpu.device.get_torch_device`) |
| Seeding/determinism | `torch.cuda.manual_seed` | XLA has its own RNG handling — needs separate seeding path if determinism matters | done (`tpu_as_gpu.seeding.seed_everything`) |

## Priority order (suggested, not fixed)

1. **A** (Whisper row is already in progress) — biggest payoff, each one unlocks a whole workload.
2. **B, H, G** — done at the code level (`tpu_as_gpu` + `DEBUGGING.md`), still need real-TPU verification once hardware access is back.
3. **C, D, E, F** — as they come up per-workload, needs real TPU hardware to measure (profiling, pipeline throughput) or adopt (distributed, precision).

## Out of scope

- **MLX** — see `UNSUPPORTED_MLX.md`. No TPU backend exists, not fixable here.
- Anything requiring NVIDIA-specific hardware features with no TPU analog at
  all (e.g. Transformer Engine's Hopper-specific fp8 path) stays "wontfix"
  unless a TPU-native equivalent gets named here first.
