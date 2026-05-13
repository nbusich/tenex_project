"""Generic torch training loop. Used by both the MLP and AutoEncoder.

Single tensor in / single tensor out — if you want validation, schedulers,
gradient clipping, etc., extend `_step` or pass them in via subclassing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device | str = "cpu",
        log_every: int = 25,
    ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.log_every = log_every

    def _step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        pred = self.model(batch_x)
        loss = self.loss_fn(pred, batch_y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    def fit(self, loader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            for i, (x_batch, y_batch) in enumerate(loader):
                loss = self._step(x_batch, y_batch)
                if i % self.log_every == 0:
                    print(f"\tstep {i}/{len(loader)}  loss={loss:.4f}")
