"""Transformer encoder for anomaly detection.

Treats logs as a sequence; features come from the shared sklearn pipeline.
Trains as a sequence autoencoder on NORMAL windows only, then uses
per-window reconstruction error as the anomaly signal.

The dataframe is expected to contain a `timestamp` column; sequences are
built in time-sorted order by `SequenceLogDataset`, and scores are mapped
back to the caller's original row order before returning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import SequenceLogDataset
from ..data.pipeline import build_preprocessor
from .base import BaseAnomalyModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LogSequenceTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 96,
        nhead: int = 8,
        num_layers: int = 4,
        seq_len: int = 10,
    ):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_projection = nn.Linear(d_model, feature_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, seq_len, feature_dim)
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        return self.output_projection(memory)


class TransformerAnomalyModel(BaseAnomalyModel):
    name = "transformer"
    _PREPROC = "preprocessor.joblib"
    _WEIGHTS = "model.pt"
    _CONFIG = "config.json"

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 8,
        num_layers: int = 4,
        seq_len: int = 10,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-4,
        calibration_percentile: float = 99.0,
        device: str | None = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.calibration_percentile = calibration_percentile
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.preprocessor = None
        self.model: LogSequenceTransformer | None = None
        # in_features is discovered after fitting the preprocessor, so the
        # caller does not have to know the feature schema up front.
        self.in_features: int | None = None
        self.calibration_threshold: float | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "TransformerAnomalyModel":
        self.preprocessor = build_preprocessor()

        # SequenceLogDataset handles `label` -> `X_raw` + `y_raw` and runs
        # fit_transform on the preprocessor when is_train=True.
        dataset = SequenceLogDataset(
            df.copy(), self.preprocessor, seq_len=self.seq_len, is_train=True
        )
        # Probe the actual feature width post-pipeline (includes StandardScaler).
        self.in_features = int(dataset.features.shape[1])

        self.model = LogSequenceTransformer(
            feature_dim=self.in_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            seq_len=self.seq_len,
        ).to(self.device)

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            print(f"Starting epoch: {epoch + 1}/{self.epochs}")
            for i, (x_batch, y_batch) in enumerate(loader):
                # Train only on windows that contain no anomalies — gives the
                # autoencoder a clean prior on "normal".
                normal_mask = (y_batch.squeeze(-1) == 0)
                if not normal_mask.any():
                    continue
                x_normal = x_batch[normal_mask].to(self.device)
                reconstructed = self.model(x_normal)
                loss = loss_fn(reconstructed, x_normal)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if i % 25 == 0:
                    print(
                        f"\tstep {i}/{len(loader)}  loss={loss.item():.6f}"
                    )

        # Calibrate threshold from training-set reconstruction errors.
        per_window_errors = self._reconstruction_errors_from_dataset(dataset)
        self.calibration_threshold = float(
            np.percentile(per_window_errors, self.calibration_percentile)
        )
        return self

    # ----------------------------------------------------------- inference

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._assert_ready()

        # Capture the time-sort order before SequenceLogDataset performs its
        # internal sort, so we can map per-row results back afterwards.
        df_in = df.copy()
        if "timestamp" in df_in.columns:
            df_in["_original_idx"] = np.arange(len(df_in))
            df_in["_ts_parsed"] = pd.to_datetime(df_in["timestamp"], errors="coerce")
            df_in = df_in.sort_values("_ts_parsed", kind="mergesort").reset_index(drop=True)
            original_order = df_in["_original_idx"].to_numpy()
            df_in = df_in.drop(columns=["_original_idx", "_ts_parsed"])
        else:
            original_order = np.arange(len(df_in))

        dataset = SequenceLogDataset(
            df_in, self.preprocessor, seq_len=self.seq_len, is_train=False
        )

        window_errors = self._reconstruction_errors_from_dataset(dataset)
    
        # Laplacian Kernel measures similarity score
        # This is 1-laplacian kernel which is anomaly score
        t = max(self.calibration_threshold or 1e-9, 1e-9)
        l = 2*t
        window_scores = 1.0 - np.exp(-window_errors / l)
        window_scores = np.clip(window_scores, 0.0, 1.0)

        # Map per-window scores back to per-row scores aligned with the
        # ORIGINAL df order. Each window covers `seq_len` rows; we apply
        # the window's score to every row it contains and take the max
        # across overlapping windows. That way the first (seq_len - 1)
        # rows still get scored

        n = len(df_in)
        per_row_sorted = np.zeros(n, dtype=np.float32)
        for i, score in enumerate(window_scores):
            end = min(i + self.seq_len, n)
            per_row_sorted[i:end] = np.maximum(per_row_sorted[i:end], score)

        out = np.zeros(len(df), dtype=np.float32)
        out[original_order] = per_row_sorted
        return out

    @torch.no_grad()
    def _reconstruction_errors_from_dataset(
        self, dataset: SequenceLogDataset
    ) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        out: list[np.ndarray] = []
        for x_batch, _y_batch in loader:
            x_batch = x_batch.to(self.device)
            recon = self.model(x_batch)
            # (B, seq_len, F) -> scalar per window
            err = ((recon - x_batch) ** 2).mean(dim=(1, 2))
            out.append(err.cpu().numpy())
        if not out:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(out)

    # -----------------------------------------------------------------

    def save(self, artifact_dir: str | Path) -> None:
        self._assert_ready()
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path / self._PREPROC)
        torch.save(self.model.state_dict(), path / self._WEIGHTS)
        (path / self._CONFIG).write_text(
            json.dumps(
                {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "seq_len": self.seq_len,
                    "in_features": self.in_features,
                    "calibration_threshold": self.calibration_threshold,
                    "calibration_percentile": self.calibration_percentile,
                }
            )
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "TransformerAnomalyModel":
        path = Path(artifact_dir)
        config = json.loads((path / cls._CONFIG).read_text())
        instance = cls(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            seq_len=config["seq_len"],
            calibration_percentile=config.get("calibration_percentile", 99.0),
        )
        instance.in_features = int(config["in_features"])
        instance.calibration_threshold = config.get("calibration_threshold")
        instance.preprocessor = joblib.load(path / cls._PREPROC)
        instance.model = LogSequenceTransformer(
            feature_dim=instance.in_features,
            d_model=instance.d_model,
            nhead=instance.nhead,
            num_layers=instance.num_layers,
            seq_len=instance.seq_len,
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
