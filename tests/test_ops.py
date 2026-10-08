import pytest

from tpu_as_gpu.ops import nms


def _reference_nms(boxes, scores, iou_threshold):
    """Pure-Python greedy NMS, no deps, to check `nms` independently of torchvision."""

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-9)

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= iou_threshold for j in keep):
            keep.append(i)
    return set(keep)


def test_nms_matches_pure_python_reference():
    jnp = pytest.importorskip("jax.numpy")
    import random

    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(1, 12)
        boxes = []
        for _ in range(n):
            x1, y1 = rng.uniform(0, 50), rng.uniform(0, 50)
            boxes.append([x1, y1, x1 + rng.uniform(1, 20), y1 + rng.uniform(1, 20)])
        scores = [rng.uniform(0, 1) for _ in range(n)]

        keep_mask = nms(jnp.array(boxes, dtype=jnp.float32), jnp.array(scores), 0.5)
        ours = set(int(i) for i in jnp.nonzero(keep_mask)[0])
        assert ours == _reference_nms(boxes, scores, 0.5)


def test_nms_rejects_non_jax():
    import numpy as np

    with pytest.raises(NotImplementedError):
        nms(np.zeros((2, 4)), np.zeros(2))


def test_nms_matches_torchvision():
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    tv_ops = pytest.importorskip("torchvision.ops")

    boxes = [
        [0, 0, 10, 10],
        [1, 1, 11, 11],  # heavy overlap with box 0
        [50, 50, 60, 60],  # disjoint
        [51, 51, 59, 59],  # inside box 2, heavy overlap
    ]
    scores = [0.9, 0.8, 0.7, 0.6]

    keep_mask = nms(jnp.array(boxes, dtype=jnp.float32), jnp.array(scores), 0.3)
    ours = set(int(i) for i in jnp.nonzero(keep_mask)[0])

    ref_idx = tv_ops.nms(
        torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), 0.3
    )
    ref = set(int(i) for i in ref_idx)

    assert ours == ref == {0, 2}
