"""Drop-in anomaly models. Add a new one by:

  1. Subclassing `BaseAnomalyModel`.
  2. Importing it below.
  3. Registering it in `MODEL_REGISTRY`.

Everything else (training scripts, inference, the website hookup) finds
models via this registry.
"""

from __future__ import annotations

from .autoencoder import AutoEncoderAnomalyModel
from .base import BaseAnomalyModel
from .encodertransformer import TransformerAnomalyModel
from .random_forest import RandomForestAnomalyModel
from .mlp import MLPAnomalyModel

MODEL_REGISTRY: dict[str, type[BaseAnomalyModel]] = {
    RandomForestAnomalyModel.name: RandomForestAnomalyModel,
    AutoEncoderAnomalyModel.name: AutoEncoderAnomalyModel,
    TransformerAnomalyModel.name: TransformerAnomalyModel,
    MLPAnomalyModel.name: MLPAnomalyModel
}


def get_model_class(name: str) -> type[BaseAnomalyModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Choices: {choices}") from exc


__all__ = [
    "AutoEncoderAnomalyModel",
    "BaseAnomalyModel",
    "MODEL_REGISTRY",
    "RandomForestAnomalyModel",
    "TransformerAnomalyModel",
    "MLPAnomalyModel"
    "get_model_class",
]
