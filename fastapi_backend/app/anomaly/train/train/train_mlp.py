from ..eval.metrics import evaluate
from ..model.mlp import MLPAnomalyModel
from ._common import artifact_dir_for, load_splits


def main() -> None:
    train_df, test_df = load_splits()
    model = MLPAnomalyModel(in_dim=67,
        epochs=20,
        batch_size=256,
        lr=1e-3)

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