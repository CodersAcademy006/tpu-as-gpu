# Roadmap

Three stages. Each gates the next — don't start Stage N+1 before Stage N is
actually done, the paper needs both the taxonomy and the tool to be a real
contribution, not just the taxonomy.

## Stage 1 — helper scripts, all 29 `SCOPE.md` rows (current)

Port every GPU-only process to TPU, one row at a time, tracked as a GitHub
issue each. This is the raw material: what breaks, why, and the working fix
for each case. Status (2026-10-08): 17/29 done (A1/Whisper, A6/quantization,
B, D, E, F, G, H), with A1, A6, B1-B4, E1/E3, F1, and H1-H2 **verified on
real TPU hardware** (Colab v5e-1, `results/colab_run_2026-10-08.md`) —
D1-D2 are dispatch-only, not yet exercised in a real loop; E2/F2/G are
documented conventions, not code. 1/29 in progress: the custom-CUDA-ops row
shipped NMS (`tpu_as_gpu.ops.nms`, pure JAX, CPU-verified against an
independent reference, not yet run on real TPU hardware); ROI align and
deformable conv, same row, not started. Also verified, separately from any
SCOPE row: a real train-then-infer cycle on TPU (`examples/torch_xla/
train_infer_demo.py`), the first backward-pass+optimizer-step proof in
this repo. 3/29 (Splash Attention x2, Ragged Paged Attention) have the
right upstream solution identified but adoption is blocked by a real
Mosaic version-skew bug, not our code (`research/prior_art.md` §8). 7/29
not started (4 of A's remaining rows, all of C — needs multi-device
hardware). Beyond the 29 rows: the real mission (any model, not just the
ones already ported, gets TPU support automatically) is tracked as an
open backlog in issue #30. See `RESEARCH_DIRECTION.md` for what Stage 1
still needs before this is a research-grade compatibility layer rather
than a prototype backlog — capability registry, no-silent-CPU-fallback
policy, numerical/performance benchmarking — and why most of that is
deliberately deferred until the function count justifies it.

## Stage 2 — the auto-detect/auto-rewrite tool

Only once Stage 1 is actually finished (not "mostly," all 29 rows closed or
explicitly wontfix'd). A tool that scans PyTorch/JAX/TF code for the
patterns in `SCOPE.md` Category A/B/H (CUDA-only kernel calls, data-dependent
control flow, hardcoded `.cuda()`) and flags or rewrites them. This is the
actual novel artifact — Stage 1 alone is "we adopted some libraries,"
Stage 2 is "we built something nobody else has."

## Stage 3 — publish

Once Stage 1 + Stage 2 are both done. The paper's contribution is the pair:
a systematic empirical taxonomy of GPU→TPU portability gaps (no prior
survey like this exists, confirmed by direct search — see
`research/prior_art.md`) plus a tool that acts on it automatically. Not
decided yet: target venue (workshop vs full paper), and whether Stage 2's
tool needs its own benchmark section or rides on Stage 1's measured
overhead numbers.
