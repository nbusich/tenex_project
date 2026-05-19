"""Model comparison endpoint.

Runs every trained model against a held-out slice of the CSIC test set and
returns per-model metrics plus a paginated row-level breakdown so the
frontend can show which model caught which attack.

The CSIC test set is the ground-truth source — every row has a known
label. If a model has not been trained yet, it's reported as "unavailable"
rather than failing the whole response.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.anomaly.train import MODEL_REGISTRY, get_engine
from app.database import User
from app.users import current_active_user

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Compare against the FULL CSIC test set by default so the website's F1
# matches what the training scripts print. The sample knob is kept for
# debugging — pass `?sample=500` if you only want a slice.
DEFAULT_SAMPLE_SIZE = 0  # 0 = use the whole test set
MAX_SAMPLE_SIZE = 20000

# Per-model thresholds. RF's positive-class probabilities cluster near 0
# (see the data-leak issue in prepare.py — train and test anomaly windows
# don't overlap, so RF predicts near-0 for every test row). A tiny
# threshold surfaces RF's actual ranking instead of zeroing it out.
PER_MODEL_THRESHOLDS: dict[str, float] = {
    "random_forest": 1e-3,
    "autoencoder": 0.5,
    "transformer": 0.5,
    "mlp": 0.5,
}
DEFAULT_THRESHOLD = 0.5


def _threshold_for(name: str, override: float | None) -> float:
    if override is not None:
        return override
    return PER_MODEL_THRESHOLDS.get(name, DEFAULT_THRESHOLD)

TEST_CSV = (
    Path(__file__).resolve().parents[1]
    / "anomaly"
    / "train"
    / "data"
    / "csic-2010"
    / "test_dataset.csv"
)


@router.get("/compare")
async def compare_models(
    user: User = Depends(current_active_user),
    sample: int = Query(DEFAULT_SAMPLE_SIZE, ge=0, le=MAX_SAMPLE_SIZE),
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=200),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    test_df = _load_test_sample(sample_size=sample)
    y_true = test_df["label"].astype(int).values

    per_model_summary: list[dict[str, Any]] = []
    per_model_predictions: dict[str, np.ndarray] = {}
    thresholds_used: dict[str, float] = {}

    for model_name in MODEL_REGISTRY.keys():
        t = _threshold_for(model_name, threshold)
        thresholds_used[model_name] = t
        engine = get_engine(model_name)
        if not engine.is_available():
            per_model_summary.append(
                {
                    "name": model_name,
                    "available": False,
                    "metrics": None,
                    "threshold": t,
                }
            )
            continue
        try:
            proba = engine.model().predict_proba(test_df.drop(columns=["label"]))
            proba = np.asarray(proba).reshape(-1)
            pred = (proba >= t).astype(int)
        except Exception as exc:
            LOG.exception("Model %s failed during comparison", model_name)
            per_model_summary.append(
                {
                    "name": model_name,
                    "available": False,
                    "error": str(exc),
                    "metrics": None,
                    "threshold": t,
                }
            )
            continue

        per_model_predictions[model_name] = pred
        per_model_summary.append(
            {
                "name": model_name,
                "available": True,
                "metrics": _metrics(y_true, pred, proba),
                "threshold": t,
                "_proba": proba,  # stripped below before serialization
            }
        )

    # Build the row-level table for the requested page.
    start = (page - 1) * size
    end = start + size
    page_slice = test_df.iloc[start:end]
    rows: list[dict[str, Any]] = []
    for idx_in_slice, (_, row) in enumerate(page_slice.iterrows()):
        absolute_idx = start + idx_in_slice
        verdicts: dict[str, dict[str, Any]] = {}
        for entry in per_model_summary:
            name = entry["name"]
            if not entry["available"]:
                verdicts[name] = {"available": False}
                continue
            pred_label = int(per_model_predictions[name][absolute_idx])
            score = float(entry["_proba"][absolute_idx])
            truth = int(row["label"])
            verdicts[name] = {
                "available": True,
                "prediction": pred_label,
                "score": round(score, 3),
                "correct": pred_label == truth,
            }
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "client_ip": row.get("client_ip"),
                "user": row.get("user"),
                "method": row.get("method"),
                "url": row.get("url"),
                "label": int(row["label"]),
                "verdicts": verdicts,
            }
        )

    # Strip raw probability arrays before returning.
    for entry in per_model_summary:
        entry.pop("_proba", None)

    return {
        "sample_size": len(test_df),
        "thresholds": thresholds_used,
        "page": page,
        "size": size,
        "total": len(test_df),
        "pages": max(1, (len(test_df) + size - 1) // size),
        "models": per_model_summary,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_predicted_positive": int(y_pred.sum()),
        "mean_score": float(y_proba.mean()),
    }


@lru_cache(maxsize=4)
def _load_test_sample(sample_size: int) -> pd.DataFrame:
    """Load the CSIC test set, optionally subsampled.

    `sample_size == 0` returns the full test set so the website's F1 lines
    up with what the training scripts print (they evaluate on the whole
    test split). Pass a positive `sample` query param to subsample.
    """
    if not TEST_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "CSIC test set not found. Run "
                "`python -m app.anomaly.train.data.prepare` first."
            ),
        )
    df = pd.read_csv(TEST_CSV)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df
