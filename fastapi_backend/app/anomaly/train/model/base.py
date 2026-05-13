"""Abstract base class shared by every anomaly model.

The website's `detector.py` only knows about this interface. Concrete models
(RandomForest, MLP, AutoEncoder, ...) are interchangeable as long as they:

  * `name` — short string ID (also the artifact subdirectory).
  * `fit(df, label_col)`            — train on a pandas DataFrame.
  * `predict_proba(df) -> np.ndarray`  — float scores in [0, 1].
  * `predict(df, threshold) -> np.ndarray`  — bool labels.
  * `save(artifact_dir)` / `cls.load(artifact_dir)`  — round-trip to disk.

`fit` takes the raw DataFrame (not pre-featurized) so each model owns its
own preprocessor instance and there is no "did you remember to use the
right preprocessor?" footgun at inference time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class BaseAnomalyModel(ABC):
    name: ClassVar[str]

    @abstractmethod
    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "BaseAnomalyModel":
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] — higher = more anomalous."""

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(bool)

    @abstractmethod
    def save(self, artifact_dir: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, artifact_dir: str | Path) -> "BaseAnomalyModel":
        ...
