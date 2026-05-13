"""Train + evaluate + save the RandomForest model.

Run from the backend root:
    python -m app.anomaly.train.train.train_rf
"""

from __future__ import annotations

from ..eval.metrics import evaluate
from ..model.random_forest import RandomForestAnomalyModel
from ._common import artifact_dir_for, load_splits


def main() -> None:
    train_df, test_df = load_splits()

    model = RandomForestAnomalyModel()
    model.fit(train_df, label_col="label")

    y_proba = model.predict_proba(test_df)
    y_pred = (y_proba >= 1e-5).astype(int)
    y_true = test_df["label"].astype(int).values

    metrics = evaluate(y_true, y_pred, y_proba, plot=False)
    print(metrics)

    model.save(artifact_dir_for(model.name))
    print(f"Saved model to {artifact_dir_for(model.name)}")


if __name__ == "__main__":
    main()
