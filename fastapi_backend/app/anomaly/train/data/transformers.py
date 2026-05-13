"""Custom sklearn-compatible transformers shared by every model.

Extracted from the original notebook (`data/explore.ipynb`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class _FittedFlagMixin:

    def _mark_fitted(self) -> None:
        self.fitted_ = True

    def __sklearn_is_fitted__(self) -> bool:  # noqa: D401
        return getattr(self, "fitted_", False)


class IPTransformer(_FittedFlagMixin, BaseEstimator, TransformerMixin):
    """Splits IP strings into 4 numerical octets."""

    def fit(self, X, y=None):
        self._mark_fitted()
        return self

    def transform(self, X):
        ips = X.iloc[:, 0].str.split(".", expand=True).astype(int)
        ips.columns = [f"ip_{i}" for i in range(4)]
        return ips

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ip_{i}" for i in range(4)])


class DateTransformer(_FittedFlagMixin, BaseEstimator, TransformerMixin):
    """Extracts behavioral features from timestamps."""
    def fit(self, X, y=None):
        self._mark_fitted()
        return self

    def transform(self, X):
        dt = pd.to_datetime(X.iloc[:, 0])
        return pd.DataFrame({
            'hour': dt.dt.hour,
            'day_of_week': dt.dt.dayofweek,
            'is_weekend': dt.dt.dayofweek.isin([5, 6]).astype(int),
            'unix_time': dt.astype('int64') // 10**9
        })

    def get_feature_names_out(self, input_features=None):
        return np.array(['hour', 'day_of_week', 'is_weekend', 'unix_time'])


class ContentLengthTransformer(_FittedFlagMixin, BaseEstimator, TransformerMixin):
    """Pull the numeric portion out of a `Content-Length: N` style string."""

    def fit(self, X, y=None):
        self._mark_fitted()
        return self

    def transform(self, X):
        target_col = X["content_length"]
        if target_col.dtype == object:
            extracted = target_col.astype(str).str.extract(
                r"Content-Length:\s*(\d+)", expand=False
            )
            return pd.DataFrame(extracted.astype(float).fillna(0.0))
        # Already numeric — coerce to float and forward.
        return pd.DataFrame(target_col.astype(float).fillna(0.0))

    def get_feature_names_out(self, input_features=None):
        return np.array(["content_length_clean"])


class URLFeatureExtractor(_FittedFlagMixin, BaseEstimator, TransformerMixin):
    """Dense, hand-engineered URL features (no TF-IDF / sparse matrices)."""

    SQL_CHARS = ("'", '"', "=", "--", ";")

    def fit(self, X, y=None):
        self._mark_fitted()
        return self

    def transform(self, X):
        urls = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X

        df = pd.DataFrame()
        df["url_length"] = urls.apply(lambda s: len(str(s)))
        df["num_digits"] = urls.apply(lambda s: sum(c.isdigit() for c in str(s)))
        df["num_special_chars"] = urls.apply(
            lambda s: sum(not c.isalnum() for c in str(s))
        )
        df["digit_ratio"] = df["num_digits"] / (df["url_length"] + 1e-5)

        df["has_sql_chars"] = urls.apply(
            lambda s: sum(str(s).count(c) for c in self.SQL_CHARS)
        )
        df["has_script"] = urls.apply(
            lambda s: 1 if "script" in str(s).lower() else 0
        )
        df["has_union"] = urls.apply(
            lambda s: 1 if "union" in str(s).lower() else 0
        )
        df["has_select"] = urls.apply(
            lambda s: 1 if "select" in str(s).lower() else 0
        )

        df["entropy"] = urls.apply(_shannon_entropy)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "url_length",
                "num_digits",
                "num_special_chars",
                "digit_ratio",
                "has_sql_chars",
                "has_script",
                "has_union",
                "has_select",
                "entropy",
            ]
        )

def _shannon_entropy(s) -> float:
    chars = list(str(s))
    if not chars:
        return 0.0
    _, counts = np.unique(chars, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


class NaNLogger(_FittedFlagMixin, BaseEstimator, TransformerMixin):
    """Diagnostic pass-through that prints NaN counts. Useful during dev."""

    def __init__(self, step_name: str = ""):
        self.step_name = step_name

    def fit(self, X, y=None):
        self._mark_fitted()
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            temp_df = pd.DataFrame(X.toarray())
        else:
            temp_df = pd.DataFrame(X)
        nans = int(temp_df.isna().sum().sum())
        print(f"[{self.step_name}] Total NaNs: {nans}")
        if nans > 0:
            print(temp_df.isna().sum()[temp_df.isna().sum() > 0])
        return X
