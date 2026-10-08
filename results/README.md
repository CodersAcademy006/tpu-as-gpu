# Results

Empty on purpose. Real numbers land here only after an actual run on
Kaggle TPU hardware — `examples/torch_xla/smoke_test.py`,
`examples/jax/smoke_test.py`, `examples/tensorflow/smoke_test.py`, and
`examples/torch_xla/whisper_tpu.py` have not been executed on real TPU
hardware yet (Kaggle MCP session needed re-auth, done as of
2026-10-08).

One file per run, named `<framework>_<yyyy-mm-dd>.md`, containing:
raw stdout, wall-clock numbers, and the Kaggle notebook/kernel slug it
came from. No numbers get written into `SCOPE.md`/`README.md` as "done"
until the file backing them exists here.
