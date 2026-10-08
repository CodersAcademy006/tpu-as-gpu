"""Proves a TPU device is visible and a basic op/compile cycle works.
Run this before anything else on a fresh Kaggle TPU VM notebook."""
import time

import torch
import torch_xla.core.xla_model as xm


def main():
    device = xm.xla_device()
    print(f"xla device: {device}")

    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)

    start = time.time()
    c = a @ b
    xm.mark_step()  # force the lazy graph to actually execute
    elapsed = time.time() - start

    assert c.shape == (2048, 2048)
    print(f"matmul ok, shape={tuple(c.shape)}, first-compile+run={elapsed:.2f}s")


if __name__ == "__main__":
    main()
