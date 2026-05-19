"""Shared sklearn preprocessor used by every model.

Models that consume a 2-D float matrix (RandomForest, MLP, AutoEncoder) all
fit the SAME preprocessor; only the head differs. Keep this file the single
source of truth for the feature schema.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from .transformers import (
    ContentLengthTransformer,
    DateTransformer,
    URLFeatureExtractor,

)

EXPECTED_INPUT_COLUMNS = [
    "timestamp",
    "user",
    "method",
    "url",
    "content_length",
]


def build_preprocessor() -> Pipeline:
    """Return the fitted-on-demand preprocessor with scaling."""

    # 1. The Feature Extractor
    preprocessor = ColumnTransformer(
        transformers=[
            ("time", DateTransformer(), ["timestamp"]),
            ("url_text", URLFeatureExtractor(), ["url"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["method", "user"]),
            ("keep", "passthrough", ["content_length"])
        ]
    )
    
    # 2. The Final Pipeline (Extraction + Scaling)
    full_pipeline = Pipeline(steps=[
        ('features', preprocessor),
        ('scaler', StandardScaler()) 
    ])

    return full_pipeline
