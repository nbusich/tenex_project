"""Inference entry point used by `app.anomaly.detector`.

Responsibilities:
  1. Lazily load a trained model from `train/artifacts/{name}/`.
  2. Convert the website's ZScaler list-of-dicts shape into the column
     schema the preprocessor expects.
  3. Return scores + per-entry reasons, ready to be merged back into the
     log-entry dicts.

The model is cached on the engine instance, so repeated `/logs/upload`
requests don't re-read weights from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from ..data.pipeline import EXPECTED_INPUT_COLUMNS
from ..model import BaseAnomalyModel, get_model_class

LOG = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class InferenceEngine:
    def __init__(self, model_name: str, artifacts_dir: Path | None = None):
        self.model_name = model_name
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        )
        self._model: BaseAnomalyModel | None = None
        self._lock = Lock()

    @property
    def artifact_path(self) -> Path:
        return self.artifacts_dir / self.model_name

    def is_available(self) -> bool:
        return self.artifact_path.exists() and any(self.artifact_path.iterdir())

    def model(self) -> BaseAnomalyModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    model_cls = get_model_class(self.model_name)
                    LOG.info("Loading %s model from %s", self.model_name, self.artifact_path)
                    self._model = model_cls.load(self.artifact_path)
        return self._model

    def score_entries(
        self,
        entries: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        if not entries:
            return entries

        df = _entries_to_dataframe(entries)
        model = self.model()
        proba = model.predict_proba(df)

        for entry, score in zip(entries, proba):
            score_f = float(score)
            entry["anomaly_score"] = round(score_f, 3)
            entry["is_anomaly"] = score_f >= threshold
            entry["anomaly_reason"] = (
                f"{self.model_name} score {score_f:.2f} >= {threshold:.2f}"
                if score_f >= threshold
                else None
            )
        return entries


# ---------------------------------------------------------------------------
# Engine singleton — `get_engine(name)` is called from the request hot path.
# ---------------------------------------------------------------------------

_ENGINES: dict[str, InferenceEngine] = {}
_ENGINES_LOCK = Lock()


def get_engine(model_name: str) -> InferenceEngine:
    with _ENGINES_LOCK:
        if model_name not in _ENGINES:
            _ENGINES[model_name] = InferenceEngine(model_name)
        return _ENGINES[model_name]


def score_entries(
    entries: list[dict[str, Any]],
    model_name: str,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    return get_engine(model_name).score_entries(entries, threshold=threshold)


# ---------------------------------------------------------------------------
# Mapping ZScaler log entries -> the column schema seen during training.
# ---------------------------------------------------------------------------


def _entries_to_dataframe(entries: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with EXACTLY the columns the preprocessor expects."""
    rows = []
    for e in entries:
        # Prefer the raw "Content-Length: N" string when the parser captured
        # one directly (CSIC-style logs); otherwise reconstruct it from the
        # numeric bytes_sent so non-CSIC ZScaler exports still parse.
        raw_cl = e.get("content_length")
        if isinstance(raw_cl, str) and raw_cl.strip():
            content_length = raw_cl
        elif e.get("bytes_sent") is not None:
            content_length = f"Content-Length: {e['bytes_sent']}"
        else:
            content_length = "Content-Length: 0"

        rows.append(
            {
                "timestamp": _coerce_timestamp(e.get("timestamp")),
                "user": e.get("user_login") or "unknown",
                "client_ip": e.get("source_ip") or "0.0.0.0",
                "method": (e.get("method") or "GET").upper(),
                "url": e.get("url") or "",
                "content_length": content_length,
            }
        )
    df = pd.DataFrame(rows, columns=EXPECTED_INPUT_COLUMNS)
    return df


def _coerce_timestamp(value: Any) -> str:
    """The preprocessor uses `pd.to_datetime`, which is happy with most shapes.

    We just need a non-None, parseable string. Fallback is the unix epoch
    so missing timestamps don't crash the pipeline.
    """
    if value is None:
        return "1970-01-01 00:00:00"
    if isinstance(value, str):
        return value
    return str(value)
