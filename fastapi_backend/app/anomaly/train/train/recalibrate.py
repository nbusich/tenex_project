"""Update the saved `calibration_threshold` for the trained AE / Transformer.

The original training run saved the 50th-percentile (median) reconstruction
error as the calibration anchor. That's far too aggressive — anything above
the median scores past the squash midpoint, and with the soft squash that
translates to almost every uploaded row clearing the 0.5 cutoff.

This script:
  1. Loads each trained model from artifacts/.
  2. Runs the (existing, already-fitted) preprocessor + model on the training
     normals to recompute per-row / per-window reconstruction errors.
  3. Writes a new `calibration_threshold` at a higher percentile (default 95)
     back into the artifact's `config.json`.

No model weights are touched.

Usage from the backend root:
    python -m app.anomaly.train.train.recalibrate                      # both, 95th
    python -m app.anomaly.train.train.recalibrate --pct 99             # both, 99th
    python -m app.anomaly.train.train.recalibrate --model autoencoder --pct 85
    python -m app.anomaly.train.train.recalibrate --model transformer --pct 99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.autoencoder import AutoEncoderAnomalyModel
from ..model.encodertransformer import TransformerAnomalyModel
from ._common import ARTIFACTS_DIR, TRAIN_CSV


def _load_train_normals() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing {TRAIN_CSV}. Run prepare first.")
    df = pd.read_csv(TRAIN_CSV)
    return df[df["label"] == 0].reset_index(drop=True)


def _recalibrate_autoencoder(pct: float) -> None:
    art = ARTIFACTS_DIR / "autoencoder"
    if not art.exists():
        print(f"[autoencoder] skipped — no artifacts at {art}")
        return

    model = AutoEncoderAnomalyModel.load(art)
    train_normals = _load_train_normals().drop(columns=["label"])
    X_mat = model.preprocessor.transform(train_normals).astype(np.float32)
    errors = model._reconstruction_errors(X_mat)
    new_t = float(np.percentile(errors, pct))

    cfg_path = art / AutoEncoderAnomalyModel._CONFIG
    cfg = json.loads(cfg_path.read_text())
    old_t = cfg.get("calibration_threshold")
    cfg["calibration_threshold"] = new_t
    cfg["calibration_percentile"] = pct
    cfg_path.write_text(json.dumps(cfg))
    print(
        f"[autoencoder] threshold {old_t!r} -> {new_t!r} "
        f"(pct={pct}, n_train_normals={len(errors)})"
    )


def _recalibrate_transformer(pct: float) -> None:
    art = ARTIFACTS_DIR / "transformer"
    if not art.exists():
        print(f"[transformer] skipped — no artifacts at {art}")
        return

    from ..data.dataset import SequenceLogDataset

    model = TransformerAnomalyModel.load(art)
    train_normals = _load_train_normals()
    dataset = SequenceLogDataset(
        train_normals, model.preprocessor, seq_len=model.seq_len, is_train=False
    )
    errors = model._reconstruction_errors_from_dataset(dataset)
    new_t = float(np.percentile(errors, pct))

    cfg_path = art / TransformerAnomalyModel._CONFIG
    cfg = json.loads(cfg_path.read_text())
    old_t = cfg.get("calibration_threshold")
    cfg["calibration_threshold"] = new_t
    cfg["calibration_percentile"] = pct
    cfg_path.write_text(json.dumps(cfg))
    print(
        f"[transformer] threshold {old_t!r} -> {new_t!r} "
        f"(pct={pct}, n_train_windows={len(errors)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pct",
        type=float,
        default=95.0,
        help="Percentile of training reconstruction error to use as the new "
        "calibration anchor (default: 95.0).",
    )
    parser.add_argument(
        "--model",
        choices=("autoencoder", "transformer", "both"),
        default="both",
        help="Which model to recalibrate (default: both).",
    )
    args = parser.parse_args()

    if args.model in ("autoencoder", "both"):
        _recalibrate_autoencoder(args.pct)
    if args.model in ("transformer", "both"):
        _recalibrate_transformer(args.pct)


if __name__ == "__main__":
    main()
