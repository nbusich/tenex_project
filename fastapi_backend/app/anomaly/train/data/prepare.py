"""End-to-end data preparation.

Runs the full pipeline:
  1. Download CSIC-2010 (if missing).
  2. Augment + split into train_dataset.csv / test_dataset.csv (raw).
  3. Fit the preprocessor on the train split and dump processed CSVs to
     csic-2010/processed/dense/.

Usage:
    python -m app.anomaly.train.data.prepare              # incremental
    python -m app.anomaly.train.data.prepare --rebuild    # force everything

The raw split is what every model in this repo trains on (they fit their
own preprocessor internally). The processed CSVs are kept around for
ad-hoc exploration / notebooks.
"""

from __future__ import annotations

import argparse
import datetime
import random
from pathlib import Path

import pandas as pd

from .dataset import write_processed_csv
from .download import download_csic_2010
from .pipeline import build_preprocessor

DATA_DIR = Path(__file__).resolve().parent / "csic-2010"
RAW_CSV = DATA_DIR / "csic_database.csv"
TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "test_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed" / "dense"
PROCESSED_TRAIN = PROCESSED_DIR / "processed_train.csv"
PROCESSED_TEST = PROCESSED_DIR / "processed_test.csv"

BASE_TIME = datetime.datetime(2026, 5, 8, 9, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-generation of every artifact even if cached.",
    )
    args = parser.parse_args()

    _ensure_raw_csv(force=args.rebuild)
    _ensure_split(force=args.rebuild)
    _ensure_processed(force=args.rebuild)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _ensure_raw_csv(force: bool) -> None:
    if RAW_CSV.exists() and not force:
        print(f"[1/3] Raw CSIC CSV already present at {RAW_CSV}.")
        return
    print(f"[1/3] Downloading CSIC-2010 to {DATA_DIR} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_csic_2010(DATA_DIR)


def _ensure_split(force: bool) -> None:
    if TRAIN_CSV.exists() and TEST_CSV.exists() and not force:
        print(f"[2/3] Raw split already present at {TRAIN_CSV} / {TEST_CSV}.")
        return
    print(f"[2/3] Generating train/test splits ...")
    df = pd.read_csv(RAW_CSV)
    normal_df = df[df["classification"] == 0]
    anomaly_df = df[df["classification"] == 1]

    train_set = pd.concat(
        [
            normal_df.sample(frac=0.8, random_state=42),
            anomaly_df.sample(frac=0.8, random_state=42),
        ]
    )
    test_set = pd.concat(
        [
            normal_df.drop(normal_df.sample(frac=0.8, random_state=42).index),
            anomaly_df.drop(anomaly_df.sample(frac=0.8, random_state=42).index),
        ]
    )

    _augment(train_set).sample(frac=1, random_state=0).to_csv(TRAIN_CSV, index=False)
    _augment(test_set).sample(frac=1, random_state=0).to_csv(TEST_CSV, index=False)
    print(f"     wrote {TRAIN_CSV} ({len(train_set)} rows) + {TEST_CSV} ({len(test_set)} rows)")


def _ensure_processed(force: bool) -> None:
    if PROCESSED_TRAIN.exists() and PROCESSED_TEST.exists() and not force:
        print(f"[3/3] Processed CSVs already present at {PROCESSED_DIR}.")
        return
    print(f"[3/3] Building processed CSVs at {PROCESSED_DIR} ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    preprocessor = build_preprocessor()
    n_train_cols = write_processed_csv(
        train_df, label_col="label", preprocessor=preprocessor,
        out_path=PROCESSED_TRAIN, fit=True,
    )
    n_test_cols = write_processed_csv(
        test_df, label_col="label", preprocessor=preprocessor,
        out_path=PROCESSED_TEST, fit=False,
    )
    print(f"     train: {n_train_cols} features  test: {n_test_cols} features")


# ---------------------------------------------------------------------------
# Augmentation (re-implemented locally so this file is self-contained)
# ---------------------------------------------------------------------------


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    users = [f"user_{i:02d}@enterprise.com" for i in range(1, 51)]
    normal_ips = [f"10.10.1.{i}" for i in range(10, 60)]
    attacker_ips = ["192.168.5.99", "192.168.5.100"]

    rng = random.Random(7)
    current_time = BASE_TIME
    rows = []
    for _, row in df.iterrows():
        is_anomaly = row["classification"] == 1
        if is_anomaly:
            current_time += datetime.timedelta(seconds=rng.randint(1, 3))
            client_ip = rng.choice(attacker_ips)
        else:
            current_time += datetime.timedelta(seconds=rng.randint(10, 300))
            client_ip = rng.choice(normal_ips)
        rows.append(
            {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": rng.choice(users),
                "client_ip": client_ip,
                "method": row["Method"],
                "url": str(row["URL"]).split(" ")[0],
                "content_length": row["lenght"],
                "label": row["classification"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
