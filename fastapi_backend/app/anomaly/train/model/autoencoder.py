"""Unsupervised AutoEncoder for anomaly detection.

Train only on rows labelled "normal". At inference time, the reconstruction
error is the anomaly score — squashed to [0, 1] using a percentile of the
training-set errors as the calibration anchor.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import DataLoader, TensorDataset

from ..data.pipeline import build_preprocessor
from .base import BaseAnomalyModel


class AutoEncoder(nn.Module):
    """Symmetric MLP autoencoder."""

    def __init__(self, in_features: int, hidden_dim: int = 64, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),)
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class AutoEncoderAnomalyModel(BaseAnomalyModel):
    name = "autoencoder"
    _PREPROC = "preprocessor.joblib"
    _WEIGHTS = "model.pt"
    _CONFIG = "config.json"

    def __init__(
        self,
        hidden_dim: int = 64,
        bottleneck: int = 16,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        normal_label: int = 0,
        calibration_percentile: float = 99.0,
        device: str | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.normal_label = normal_label
        self.calibration_percentile = calibration_percentile
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.preprocessor = None
        self.model: AutoEncoder | None = None
        self.in_features: int | None = None
        # `calibration_threshold` is the reconstruction error at the configured
        # percentile of the training set — used to squash arbitrary errors into
        # a [0, 1] anomaly score at inference time.
        self.calibration_threshold: float | None = None

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "AutoEncoderAnomalyModel":
        normals = df[df[label_col] == self.normal_label]
        if normals.empty:
            raise ValueError(
                f"AutoEncoder needs rows with label == {self.normal_label} to train on."
            )

        X = normals.drop(columns=[label_col])
        self.preprocessor = build_preprocessor()
        X_mat = self.preprocessor.fit_transform(X).astype(np.float32)

        self.in_features = X_mat.shape[1]
        self.model = AutoEncoder(
            self.in_features, hidden_dim=self.hidden_dim, bottleneck=self.bottleneck
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_mat)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            running = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running += loss.item() * batch.size(0)
            print(f"[AE] epoch {epoch + 1}/{self.epochs}  loss={running / len(loader.dataset):.4f}")

        # Calibrate threshold on the training set.
        errors = self._reconstruction_errors(X_mat)
        self.calibration_threshold = float(
            np.percentile(errors, self.calibration_percentile)
        )
        return self

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._assert_ready()
        X_mat = self.preprocessor.transform(df).astype(np.float32)
        errors = self._reconstruction_errors(X_mat)
        t = max(self.calibration_threshold or 1e-9, 1e-9)
        scores = 1.0 - np.exp(-errors / (2.0 * t))
        return np.clip(scores, 0.0, 1.0)

    @torch.no_grad()
    def _reconstruction_errors(self, X_mat: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_mat)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        for (batch,) in loader:
            batch = batch.to(self.device)
            recon = self.model(batch)
            err = ((recon - batch) ** 2).mean(dim=1)
            out.append(err.cpu().numpy())
        return np.concatenate(out)

    def save(self, artifact_dir: str | Path) -> None:
        self._assert_ready()
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path / self._PREPROC)
        torch.save(self.model.state_dict(), path / self._WEIGHTS)
        (path / self._CONFIG).write_text(
            json.dumps(
                {
                    "hidden_dim": self.hidden_dim,
                    "bottleneck": self.bottleneck,
                    "in_features": self.in_features,
                    "calibration_threshold": self.calibration_threshold,
                    "calibration_percentile": self.calibration_percentile,
                }
            )
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "AutoEncoderAnomalyModel":
        path = Path(artifact_dir)
        config = json.loads((path / cls._CONFIG).read_text())
        instance = cls(
            hidden_dim=config["hidden_dim"],
            bottleneck=config["bottleneck"],
            calibration_percentile=config.get("calibration_percentile", 99.0),
        )
        instance.in_features = config["in_features"]
        instance.calibration_threshold = config.get("calibration_threshold")
        instance.preprocessor = joblib.load(path / cls._PREPROC)
        instance.model = AutoEncoder(
            instance.in_features,
            hidden_dim=instance.hidden_dim,
            bottleneck=instance.bottleneck,
        ).to(instance.device)
        instance.model.load_state_dict(
            torch.load(path / cls._WEIGHTS, map_location=instance.device)
        )
        instance.model.eval()
        return instance

    def _assert_ready(self) -> None:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model has not been fit or loaded.")
        if self.calibration_threshold is None:
            raise RuntimeError("Model is missing its calibration threshold.")
