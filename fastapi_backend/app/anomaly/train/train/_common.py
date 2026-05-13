"""Shared helpers for the per-model training scripts."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# Repo-relative path to the CSIC train/test CSVs produced by data/build.py.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "csic-2010"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "test_dataset.csv"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_CSV} or {TEST_CSV}. Run `python data/build.py` first."
        )
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def artifact_dir_for(name: str) -> Path:
    out = ARTIFACTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out
