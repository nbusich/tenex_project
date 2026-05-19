"""Train + evaluate + save the AutoEncoder model.

Run from the backend root:
    python -m app.anomaly.train.train.train_autoencoder
"""

from __future__ import annotations

from ..eval.metrics import evaluate
from ..model.autoencoder import AutoEncoderAnomalyModel
from ._common import artifact_dir_for, load_splits


def train_autoencoder(
        hidden_dim=64,
        epochs=20,
        batch_size=256,
        lr=1e-3,):
    train_df, test_df = load_splits()
    model = AutoEncoderAnomalyModel(
        hidden_dim=hidden_dim,
        bottleneck=16,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        calibration_percentile=50.0,
    )
    model.fit(train_df, label_col="label")

    y_proba = model.predict_proba(test_df)
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = test_df["label"].astype(int).values

    metrics = evaluate(y_true, y_pred, y_proba, plot=False)
    print(metrics)

    model.save(artifact_dir_for(model.name))
    print(f"Saved model to {artifact_dir_for(model.name)}")
    return metrics


if __name__ == "__main__":
    train_autoencoder()
