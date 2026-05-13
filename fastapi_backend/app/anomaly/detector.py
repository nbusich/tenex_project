"""Anomaly detector entry point used by `routes/logs.py`.

The upload endpoint picks ONE strategy per file — there's no ensemble.
Combining models tended to flag almost every row because the three models
disagree (RF predicts near-0, AE/Transformer hover around 0.5). The
single-model path is more honest about what each model thinks.

Strategy is chosen via the `model_name` argument:

    detect_anomalies(entries, model_name="transformer")  # default
    detect_anomalies(entries, model_name="autoencoder")
    detect_anomalies(entries, model_name="random_forest")
    detect_anomalies(entries, model_name="heuristic")

Threshold defaults:
    random_forest -> 1e-3 (probs cluster near 0; see /compare)
    autoencoder   -> 0.5
    transformer   -> 0.5
    heuristic     -> 0.5

If a trained model is requested but its artifacts are missing, we fall
back to the heuristic so the website still works pre-training.
"""

from __future__ import annotations

import logging
import os
from collections import Counter

LOG = logging.getLogger(__name__)


# Display labels surfaced in `anomaly_reason`.
_MODEL_LABELS = {
    "random_forest": "RandomForest",
    "autoencoder": "AutoEncoder",
    "transformer": "Transformer",
    "heuristic": "Heuristic",
}

_DEFAULT_THRESHOLDS = {
    "random_forest": 1e-3,
    "autoencoder": 0.5,
    "transformer": 0.5,
    "heuristic": 0.5,
}

DEFAULT_MODEL = "transformer"
ALLOWED_MODELS = ("transformer", "autoencoder", "random_forest", "heuristic")


def detect_anomalies(
    entries: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    if not entries:
        return entries

    name = (model_name or DEFAULT_MODEL).strip().lower()
    if name not in ALLOWED_MODELS:
        LOG.warning("Unknown model %r — falling back to %s", model_name, DEFAULT_MODEL)
        name = DEFAULT_MODEL

    threshold = _threshold_for(name)

    if name == "heuristic":
        return _heuristic_detect(entries)

    scored = _try_model(entries, name, threshold)
    if scored is not None:
        return scored

    # Trained model unavailable — fall back to the heuristic so the upload
    # still produces something useful.
    LOG.info("Model %s unavailable; using heuristic for this upload.", name)
    return _heuristic_detect(entries)


# ---------------------------------------------------------------------------
# Trained-model path
# ---------------------------------------------------------------------------


def _try_model(
    entries: list[dict],
    model_name: str,
    threshold: float,
) -> list[dict] | None:
    try:
        from .train.serve import get_engine
    except Exception as exc:
        LOG.warning("Could not import anomaly inference engine: %s", exc)
        return None

    engine = get_engine(model_name)
    if not engine.is_available():
        LOG.warning(
            "Model %s requested but no artifacts at %s.",
            model_name,
            engine.artifact_path,
        )
        return None

    try:
        scored = engine.score_entries(entries, threshold=threshold)
    except Exception as exc:
        LOG.exception("Model %s failed during scoring: %s", model_name, exc)
        return None

    label = _MODEL_LABELS.get(model_name, model_name)
    for entry in scored:
        score = float(entry.get("anomaly_score") or 0.0)
        if entry.get("is_anomaly"):
            entry["anomaly_reason"] = f"{label} score {score:.3f} ≥ {threshold:g}"
        else:
            entry["anomaly_reason"] = None
    return scored


def _threshold_for(name: str) -> float:
    raw = os.environ.get("ANOMALY_THRESHOLD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            LOG.warning("Invalid ANOMALY_THRESHOLD=%r — using model default", raw)
    return _DEFAULT_THRESHOLDS.get(name, 0.5)


# ---------------------------------------------------------------------------
# Heuristic — writes is_anomaly / anomaly_score / anomaly_reason in-place.
# ---------------------------------------------------------------------------


def _heuristic_detect(entries: list[dict]) -> list[dict]:
    ip_counts = Counter(e.get("source_ip") for e in entries if e.get("source_ip"))
    burst_threshold = 50

    for entry in entries:
        reasons: list[str] = []
        score = 0.0

        status = entry.get("status_code")
        if isinstance(status, int) and status >= 500:
            reasons.append(f"server error status {status}")
            score = max(score, 0.6)

        action = (entry.get("action") or "").lower()
        if action in {"blocked", "block", "deny"}:
            reasons.append(f"action={action}")
            score = max(score, 0.5)

        threat = entry.get("threat_name")
        if threat and threat not in {"None", "-"}:
            reasons.append(f"threat detected: {threat}")
            score = max(score, 0.9)

        ip = entry.get("source_ip")
        if ip and ip_counts[ip] > burst_threshold:
            reasons.append(
                f"unusual volume from {ip} ({ip_counts[ip]} requests in file)"
            )
            score = max(score, 0.7)

        entry["is_anomaly"] = bool(reasons)
        entry["anomaly_score"] = round(score, 3) if reasons else 0.0
        entry["anomaly_reason"] = "; ".join(reasons) if reasons else None

    return entries
