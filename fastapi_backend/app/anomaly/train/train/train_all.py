"""Train every model in MODEL_REGISTRY in one go.

Run from the backend root:
    python -m app.anomaly.train.train.train_all

Use this when you've changed the shared preprocessor or one of the custom
transformers — every saved artifact has to be re-fit on the new feature
schema, otherwise loading them at inference time can fail or score garbage.
"""

from __future__ import annotations

import sys
import traceback


def main() -> None:
    from . import train_autoencoder, train_rf, train_transformer

    steps = [
        ("RandomForest", train_rf.main),
        ("AutoEncoder", train_autoencoder.main),
        ("Transformer", train_transformer.main),
    ]
    failures: list[tuple[str, str]] = []
    for label, fn in steps:
        print(f"\n=== Training {label} ===")
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failures.append((label, "see traceback above"))

    print("\n=== Summary ===")
    for label, _ in steps:
        ok = not any(f[0] == label for f in failures)
        print(f"  {label:14s} {'OK' if ok else 'FAILED'}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
