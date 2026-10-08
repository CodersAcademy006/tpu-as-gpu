"""tpu-as-gpu: real-hardware verification run.

Kaggle script kernels only upload the single `code_file` referenced in
kernel-metadata.json, so the tpu_as_gpu library (normally
src/tpu_as_gpu/{compat,device,precision,seeding}.py in the
CodersAcademy006/tpu-as-gpu repo) is inlined here rather than pip-installed
from the private repo. Source of truth stays the repo; this is a verify-run
snapshot of it.
"""
import json
import time

results = {}


def record(name, fn):
    try:
        t0 = time.time()
        out = fn()
        results[name] = {"ok": True, "elapsed_s": round(time.time() - t0, 3), "detail": str(out)}
    except Exception as e:
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(name, "->", results[name])


# ---- inlined tpu_as_gpu (mirrors src/tpu_as_gpu in the repo) ----

def where(condition, on_true, on_false):
    module_root = type(condition).__module__.split(".")[0]
    if module_root == "torch":
        import torch
        return torch.where(condition, on_true, on_false)
    if module_root in ("jax", "jaxlib"):
        import jax.numpy as jnp
        return jnp.where(condition, on_true, on_false)
    if module_root == "tensorflow":
        import tensorflow as tf
        return tf.where(condition, on_true, on_false)
    import numpy as np
    return np.where(condition, on_true, on_false)


def bucket_length(length, buckets):
    fitting = [b for b in sorted(buckets) if length <= b]
    if not fitting:
        raise ValueError(f"length {length} exceeds largest bucket {max(buckets)}")
    return fitting[0]


class DeferredSync:
    def __init__(self):
        self._pending = []

    def collect(self, value):
        self._pending.append(value)

    def resolve(self, sync_fn=None):
        if sync_fn is None:
            sync_fn = lambda values: [v.item() if hasattr(v, "item") else v for v in values]
        values = sync_fn(self._pending)
        self._pending = []
        return values


class StaticKVCache:
    def __init__(self, max_len, shape_per_token, zeros_fn):
        self.max_len = max_len
        self.buffer = zeros_fn((max_len, *shape_per_token))
        self.pos = 0

    def write(self, token_kv):
        if self.pos >= self.max_len:
            raise IndexError("KV cache full")
        self.buffer[self.pos] = token_kv
        self.pos += 1
        return self.buffer, self.pos


def get_torch_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_kind(device):
    s = str(device)
    if "xla" in s:
        return "xla"
    if "cuda" in s:
        return "cuda"
    return "cpu"


def autocast_dtype(kind):
    return "bfloat16" if kind == "xla" else "float16"


def compile_backend(kind):
    return "openxla" if kind == "xla" else "inductor"


def seed_everything(seed):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import jax
        return jax.random.PRNGKey(seed)
    except ImportError:
        return None

# ---- end inlined tpu_as_gpu ----


def torch_xla_smoke():
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = a @ b
    xm.mark_step()
    assert c.shape == (2048, 2048)
    return f"device={device} shape={tuple(c.shape)}"


def jax_smoke():
    import jax
    import jax.numpy as jnp

    assert jax.devices()[0].platform == "tpu", "no TPU backend picked up"
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (2048, 2048))
    b = jax.random.normal(key, (2048, 2048))
    c = jax.jit(jnp.matmul)(a, b)
    c.block_until_ready()
    return f"devices={jax.devices()} shape={c.shape}"


def tensorflow_smoke():
    import tensorflow as tf

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    with strategy.scope():
        a = tf.random.normal((2048, 2048))
        b = tf.random.normal((2048, 2048))

        @tf.function
        def matmul(a, b):
            return tf.matmul(a, b)

        c = strategy.run(matmul, args=(a, b))
    assert c.shape == (2048, 2048)
    return f"devices={tf.config.list_logical_devices('TPU')} shape={c.shape}"


def tpu_as_gpu_on_real_tpu():
    """The actual point of this run: exercise the library's logic against
    real XLA-backed tensors, not CPU/numpy stand-ins. Closes the "pending
    real-TPU verification" caveat on SCOPE.md rows B1-B4, D1-D2, H1-H2."""
    import torch
    import torch_xla.core.xla_model as xm

    device = get_torch_device()
    kind = device_kind(device)
    assert kind == "xla", f"expected xla device, got {kind}"

    cond = torch.tensor([True, False], device=device)
    out = where(cond, torch.tensor([1, 1], device=device), torch.tensor([0, 0], device=device))
    xm.mark_step()
    assert out.cpu().tolist() == [1, 0]

    assert bucket_length(5, [8, 16, 32]) == 8

    sync = DeferredSync()
    sync.collect(torch.tensor(3.0, device=device))
    sync.collect(torch.tensor(4.0, device=device))
    resolved = sync.resolve()
    xm.mark_step()
    assert resolved == [3.0, 4.0]

    cache = StaticKVCache(max_len=2, shape_per_token=(2,), zeros_fn=lambda shape: torch.zeros(shape, device=device))
    cache.write(torch.tensor([1.0, 2.0], device=device))
    xm.mark_step()

    assert autocast_dtype(kind) == "bfloat16"
    assert compile_backend(kind) == "openxla"
    seed_everything(0)

    return f"device={device} kind={kind} all tpu_as_gpu checks passed on real TPU"


record("torch_xla_smoke", torch_xla_smoke)
record("jax_smoke", jax_smoke)
record("tensorflow_smoke", tensorflow_smoke)
record("tpu_as_gpu_on_real_tpu", tpu_as_gpu_on_real_tpu)

with open("tpu_verify_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
