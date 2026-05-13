"""Single evaluation entry point shared by every training script.

Re-implementation of the notebook's `eval_random_forest`, but works with
NumPy arrays, returns a metrics dict, and only plots if asked. Decoupling
the plotting keeps it usable in headless CI as well.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate(
    y_true,
    y_pred,
    y_proba,
    *,
    plot: bool = False,
    plot_path: str | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    y_proba = np.asarray(y_proba).astype(float).reshape(-1)

    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }

    if plot:
        _plot_pr_curve(y_true, y_proba, save_to=plot_path)

    return metrics


def _plot_pr_curve(y_true, y_proba, save_to: str | None = None) -> None:
    import matplotlib.pyplot as plt

    p, r, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=p[:-1], y=r[:-1], c=thresholds)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Curve Colored by Threshold")
    fig.colorbar(scatter, label="Threshold")
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
