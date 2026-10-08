# MLX: not possible on TPU

MLX is Apple's array framework, built on Metal Performance Shaders. It only
targets Apple Silicon GPUs — there is no XLA backend, no TPU backend, and no
public roadmap from Apple to add one. This isn't a Kaggle limitation or a
missing pip package, it's the framework's hardware target. Nothing in this
repo can change that.

If you need MLX-equivalent workloads on TPU, the path is JAX or torch_xla
instead (both in this repo), not MLX itself.
