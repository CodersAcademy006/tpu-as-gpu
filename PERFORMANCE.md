# Profiling and data loading on TPU (Categories E, F)

Two conventions documented here instead of wrapped in `tpu_as_gpu` — the
native API is already the fix, a wrapper would only add indirection.

## Deep kernel profiling (SCOPE row E2)

GPU: Nsight Systems/Compute. TPU: the XLA/TPU profiler, triggered
differently per framework:

```python
# torch_xla
import torch_xla.debug.profiler as xp
xp.start_trace("/tmp/profile_logdir")
... # traced region
xp.stop_trace()

# JAX
import jax
jax.profiler.start_trace("/tmp/profile_logdir")
... # traced region
jax.profiler.stop_trace()
```

Both write a trace viewable in TensorBoard's profiler plugin
(`tensorboard --logdir /tmp/profile_logdir`), not Nsight's own viewer.

## Preprocessing pipeline (SCOPE row F2)

GPU convention: arbitrary `DataLoader` + CPU workers, variable batch
content, pinned memory + `non_blocking=True` transfers. TPU convention:
prefer pre-batched, fixed-shape input, and `tf.data` is the most
TPU-native pipeline regardless of which framework trains the model —
TensorFlow's TPU input pipeline tooling (`tf.data.Dataset`, especially
with `.batch(..., drop_remainder=True)` for fixed shapes) predates and is
more mature than the PyTorch/JAX equivalents, per `rishabhindoria/GPU-to-
TPU`'s own finding (`research/prior_art.md` §1) that an unoptimized input
pipeline leaves the TPU 95% idle.

For the PyTorch/torch_xla path specifically, the one real code change (not
just a convention) is `torch_xla.distributed.parallel_loader.MpDeviceLoader`
in place of `pin_memory=True` / `.to(device, non_blocking=True)` — verified
on real TPU hardware (Colab v5e-1, 2026-10-08, see
`results/colab_run_2026-10-08.md`):

```python
from torch.utils.data import DataLoader
from torch_xla.distributed.parallel_loader import MpDeviceLoader

loader = DataLoader(dataset, batch_size=...)
mp_loader = MpDeviceLoader(loader, device)  # drop-in: iterate this instead
for batch in mp_loader:
    ...  # batch is already on `device`
```
