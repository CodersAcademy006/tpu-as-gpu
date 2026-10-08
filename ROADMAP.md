# Roadmap

Three stages. Each gates the next — don't start Stage N+1 before Stage N is
actually done, the paper needs both the taxonomy and the tool to be a real
contribution, not just the taxonomy.

## Stage 1 — helper scripts, all 29 `SCOPE.md` rows (current)

Port every GPU-only process to TPU, one row at a time, tracked as a GitHub
issue each. This is the raw material: what breaks, why, and the working fix
for each case. Status (2026-10-08): 11/29 done (A1/Whisper, B, D, G, H),
with A1 and B1-B4/H1-H2 **verified on real TPU hardware** (Colab v5e-1,
`results/colab_run_2026-10-08.md`) — D1-D2 are dispatch-only, not yet
exercised in a real loop. 3/29 have an identified upstream solution not yet
adopted (Splash Attention x2, Ragged Paged Attention — see
`research/prior_art.md`), 14/29 not started.

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
