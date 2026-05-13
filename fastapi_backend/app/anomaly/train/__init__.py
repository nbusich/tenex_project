"""Anomaly model training + serving package.

The website only imports from `serve` and `model` — everything else is for
offline training. Top-level convenience re-exports:
"""

from .model import (
    AutoEncoderAnomalyModel,
    BaseAnomalyModel,
    MODEL_REGISTRY,
    RandomForestAnomalyModel,
    TransformerAnomalyModel,
    get_model_class,
)
from .serve import InferenceEngine, get_engine, score_entries

__all__ = [
    "AutoEncoderAnomalyModel",
    "BaseAnomalyModel",
    "InferenceEngine",
    "MODEL_REGISTRY",
    "RandomForestAnomalyModel",
    "TransformerAnomalyModel",
    "get_engine",
    "get_model_class",
    "score_entries",
]
