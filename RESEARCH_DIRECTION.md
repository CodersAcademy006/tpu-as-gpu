# Research direction: what's adopted now vs. deferred

An external review of `SCOPE.md` (2026-10-08) made two kinds of claims:
factual corrections (checked against primary sources, see
`research/prior_art.md` §11) and architectural suggestions (not
fact-checkable, a design opinion). This file is the second kind — recorded
so it isn't re-litigated from scratch, and so what got adopted vs.
deferred is explicit rather than silently dropped.

## The framing correction (adopted)

Don't claim "everything that runs on GPU runs on TPU" — it's false on its
face (CUDA-specific hardware features have no TPU analog at all, e.g.
Transformer Engine's Hopper fp8 path, Isaac Gym's PhysX). `SCOPE.md`
already encodes this via its `wontfix` rows; the more honest framing for
anything written about this project going forward:

> A hardware-compatibility layer that lets GPU-oriented ML workloads
> execute on TPU through TPU-native, compiler-native, or semantically
> equivalent implementations — not a claim of total equivalence.

## Status taxonomy (adopted, lightweight)

`SCOPE.md`'s `done` conflated two different things: "code exists and
passes a CPU-logic test" vs. "verified against real TPU hardware." This
session already fixed that in practice — every row now explicitly says
**verified on real TPU hardware** or **dispatch-only, not yet verified**
in its status cell, rather than a bare `done`. That's the taxonomy; it
doesn't need a formal enum (`IMPLEMENTED`/`TPU_VERIFIED`/`NUMERICALLY_
VALIDATED`/`PERFORMANCE_BENCHMARKED`/...) until there are rows where that
distinction stops fitting in one status-cell sentence. Add the formal
version when that happens, not before.

## No-silent-CPU-fallback (adopted as a going-forward rule, not retrofitted)

Real risk: a dispatch function quietly falls back to CPU and the caller
has no idea their "TPU" run was partly CPU. `tpu_as_gpu.quantization.
quantized_matmul` already does this right — it raises `NotImplementedError`
for non-JAX input instead of silently falling back to an unquantized
matmul. **Rule going forward: every new dispatch function raises on an
unsupported path by default, never silently degrades.** Not retrofitting
this as a library-wide policy object (`set_execution_policy`, `assert_
tpu_safe`) until there are enough dispatch functions that a per-function
raise stops being sufficient — right now it is.

## Backend abstraction (already true, no action needed)

The suggestion to not hard-code one TPU backend is already satisfied:
`tpu_as_gpu.device.get_torch_device()` is the one chokepoint that resolves
XLA vs CUDA vs CPU. When `TorchTPU` goes public (`research/prior_art.md`
§2), that's a third branch in one function, not a rewrite. This is the
payoff of having a device dispatch function at all — confirmed working as
designed, not a gap.

## What's explicitly NOT being built right now, and why

The review's remaining ~30 suggestions (capability registry, operator-
coverage database, portability scoring, TPU-generation abstraction layer,
cost/performance benchmarking framework, fault-tolerance/checkpoint
recovery, graph partitioning, kernel autotuning, 15+ new SCOPE categories)
are legitimate things a *mature* TPU-compatibility project eventually
needs. They are not being built now because:

- This library has ~10 public functions. A capability registry, dispatch
  resolver, and auto-generated docs/tests from YAML metadata is the right
  shape for hundreds of dispatch points, not ten — building it now is
  pure premature abstraction, the exact failure mode this project's own
  engineering discipline (YAGNI ladder) exists to catch.
- Several suggestions assume infra this session doesn't have: multi-device
  hardware (Category C can't even be tested on a single-core v5e-1, let
  alone benchmarked for cost/scaling-efficiency), multiple TPU generations
  (this session only touched v5e-1), a model zoo benchmark suite (no
  models beyond Whisper + a toy MLP have been ported yet).
- "Millions of helper scripts" was never the literal plan — restated by
  the user directly: the mission is "someone calls the right function and
  it just works," which is a dispatch-function API, not a literal
  one-script-per-combination filesystem explosion. The registry idea is
  the *right* answer to that mission once the function count justifies it.

**When to revisit this file:** once `SCOPE.md` has significantly more than
29 rows, or once a second TPU generation / multi-device setup is actually
available to test against, re-read this list and promote the items that
now have a concrete need behind them — not on a schedule, on a trigger.

## Specific follow-ups opened (not yet done)

- Verify **Qwix** (`google/qwix`) on real TPU hardware before considering
  it a replacement for AQT — AQT's quantized-training path (forward+
  backward) isn't something Qwix's fake-quantization QAT reproduces, so
  this isn't a drop-in swap even once verified. **Attempted 2026-10-08,
  blocked**: `pip install qwix` can't even be imported against the latest
  flax that satisfies its own declared `flax>=0.12.0` floor
  (`ImportError: cannot import name 'BaseConfigContext' from
  'flax.typing'`) — a real bug in qwix's own dependency range, not
  something to work around from this repo. Revisit when qwix cuts a new
  release; don't re-attempt against the same flax/qwix pair expecting a
  different result. `research/prior_art.md` §11.1.
- Helion as a PyTorch-native alternative to hand-written Pallas for future
  Triton-equivalent kernel work — not attempted yet, just identified as
  the better entry point than raw Pallas. `research/prior_art.md` §11.4.

## Follow-ups closed

- **FP8 on v5e-1, checked (2026-10-08 Colab pass).** All four `jnp` fp8
  dtypes (`float8_e4m3fn`, `float8_e5m2`, `float8_e4m3`, `float8_e3m4`)
  compile and run a matmul on this session's actual hardware
  (`results/colab_run_2026-10-08.md`). Narrower question still open: that
  check doesn't distinguish a genuine hardware fp8 op from an upcast-
  compute-cast-back emulation — would need an HLO dump or a bf16-vs-fp8
  perf-scaling comparison to tell, not attempted. SCOPE.md's fp8 row
  reflects this: "dtypes usable," not "native-fp8 confirmed."
