# Debugging traced code (Category G)

GPU/eager execution runs each line immediately — `print(tensor)` or a
breakpoint shows the real value right away. TPU/XLA code is lazy: ops build
a graph, and nothing actually runs until a sync point. Hitting a sync point
is the fix, not a script — this is a pattern, not a helper function.

## The symptom

```python
x = a @ b
print(x)  # shows an unresolved/traced placeholder, not the real values
```

## The fix: force a sync before you print

- **torch_xla**: call `xm.mark_step()` (or `torch_xla.sync()`) before the
  print, or print `x.cpu()` directly — `.cpu()` itself is a sync point.
- **JAX**: call `x.block_until_ready()` before printing — `jax.jit` is
  async, the array exists before the computation finishes.
- **TensorFlow**: inside `tf.function`, use `tf.print(x)` instead of
  Python `print(x)` — regular `print` only runs at trace time, once, and
  never shows real values.

## Don't sync in a hot loop

Each sync point is a device round-trip. Syncing once per training step to
debug one value is fine; syncing once per inner-loop iteration silently
turns a TPU job into a GPU-speed job. Use
[`tpu_as_gpu.compat.DeferredSync`](src/tpu_as_gpu/compat.py) to collect
values across the loop and sync once, instead of syncing per-iteration.
