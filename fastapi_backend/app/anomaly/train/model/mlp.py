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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.l1 = nn.Linear(in_dim, in_dim*4)
        self.l2 = nn.Linear(in_dim*4, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # X = B, D
        i = self.l1(x)
        i2 = self.relu(i)
        y = self.l2(i2)
        return y


class MLPAnomalyModel(BaseAnomalyModel):
    name = "mlp"
    _PREPROC = "preprocessor.joblib"
    _WEIGHTS = "model.pt"
    _CONFIG = "config.json"
    def __init__(self, in_dim, lr=1e-4, epochs=10, batch_size=256, anom_col=0, device=None):
        self.in_dim = in_dim
        self.model = None
        self.preprocessor = None
        self.anom_col = anom_col
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "BaseAnomalyModel":
        y_train=df[label_col].to_numpy().astype(np.float32)
        print(y_train.shape)
        X = df.drop(columns=[label_col])
        self.preprocessor = build_preprocessor()
        X_mat = self.preprocessor.fit_transform(X).astype(np.float32)

        self.in_dim = X_mat.shape[1]
        self.model = MLP(
            in_dim=self.in_dim
        ).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_mat), torch.from_numpy(y_train).unsqueeze(dim=1)),
            batch_size=256,
            shuffle=True,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(50):
            running = 0.0
            for (batch, y) in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = loss_fn(logits, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running += loss.item() * batch.size(0)
            print(f"[MLP] epoch {epoch + 1}/{self.epochs}  loss={running / len(loader.dataset):.4f}")
        return self


    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] — higher = more anomalous."""
        self._assert_ready()
        x = self.preprocessor.transform(df).astype(np.float32)
        logits = self.model(torch.from_numpy(x))
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs.reshape(-1)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(bool)

    def save(self, artifact_dir: str | Path) -> None:
        self._assert_ready()
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path / self._PREPROC)
        torch.save(self.model.state_dict(), path / self._WEIGHTS)
        (path / self._CONFIG).write_text(
            json.dumps(
                {
                    "in_dim": self.in_dim
                }
            )
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "BaseAnomalyModel":
        path = Path(artifact_dir)
        config = json.loads((path / cls._CONFIG).read_text())
        in_dim = int(config["in_dim"])
        instance = cls(in_dim=in_dim)
        instance.preprocessor = joblib.load(path / cls._PREPROC)
        instance.model = MLP(in_dim=in_dim).to(instance.device)
        instance.model.load_state_dict(
            torch.load(path / cls._WEIGHTS, map_location=instance.device)
        )
        instance.model.eval()
        return instance
    
    def _assert_ready(self) -> None:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model has not been fit or loaded.")