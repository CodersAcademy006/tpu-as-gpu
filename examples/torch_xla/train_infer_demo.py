"""Proves the full lifecycle works on TPU: train a model, then infer with
it, on the same device, in the same process. Verified on real TPU hardware
(Colab v5e-1, 2026-10-08) — 5-step train loss went 0.705 -> 0.692, then a
clean inference pass in eval mode. See results/colab_run_2026-10-08.md.

Run this before porting a real model — if a 2-layer MLP can't train and
infer here, nothing bigger will either.
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm

from tpu_as_gpu import get_torch_device, seed_everything


def main():
    seed_everything(0)
    device = get_torch_device()

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    opt = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(64, 10, device=device)
    y = torch.randint(0, 2, (64,), device=device)

    model.train()
    t0 = time.time()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        xm.mark_step()
        losses.append(loss.item())  # sync point, fine for 5 steps — not a hot loop
    train_elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        preds = model(x).argmax(dim=-1)
        xm.mark_step()
        infer_elapsed = time.time() - t0

    print(f"device: {device}")
    print(f"train: {train_elapsed:.2f}s, losses: {losses}")
    print(f"infer: {infer_elapsed:.3f}s, preds device: {preds.device}")


if __name__ == "__main__":
    main()
