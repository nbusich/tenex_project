"""Train + evaluate + save the TransformerEncoder model.

Run from the backend root:
    python -m app.anomaly.train.train.train_transformer
"""

from __future__ import annotations

from ..eval.metrics import evaluate
from ..model.encodertransformer import TransformerAnomalyModel
from ._common import artifact_dir_for, load_splits


def main() -> None:
    train_df, test_df = load_splits()

    model = TransformerAnomalyModel(
        seq_len=10,
        d_model=96,
        nhead=6,
        num_layers=4,
        epochs=10,
        batch_size=128,
        lr=1e-4,
        calibration_percentile=50.0)
    
    model.fit(train_df, label_col="label")

    y_proba = model.predict_proba(test_df)
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = test_df["label"].astype(int).values

    metrics = evaluate(y_true, y_pred, y_proba, plot=False)
    print(metrics)

    model.save(artifact_dir_for(model.name))
    print(f"Saved model to {artifact_dir_for(model.name)}")


if __name__ == "__main__":
    main()
