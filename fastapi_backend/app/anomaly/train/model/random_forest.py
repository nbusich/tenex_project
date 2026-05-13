"""RandomForest classifier wrapper. Mirrors the notebook's full_pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..data.pipeline import build_preprocessor
from .base import BaseAnomalyModel


class RandomForestAnomalyModel(BaseAnomalyModel):
    name = "random_forest"
    _ARTIFACT = "pipeline.joblib"

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        class_weight: str | None = "balanced",
        n_jobs: int = -1,
    ):
        self.pipeline: Pipeline | None = None
        self._init_kwargs = dict(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "RandomForestAnomalyModel":
        X = df.drop(columns=[label_col])
        y = df[label_col].astype(int)

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("detector", RandomForestClassifier(**self._init_kwargs)),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fit or loaded.")
        return self.pipeline.predict_proba(df)[:, 1]

    def save(self, artifact_dir: str | Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save before fit.")
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / self._ARTIFACT)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "RandomForestAnomalyModel":
        path = Path(artifact_dir) / cls._ARTIFACT
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
