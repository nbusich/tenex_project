"""Dataset helpers shared by torch-based models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LogDataset(Dataset):
    """Reads a processed CSV (features + trailing label column)."""

    def __init__(self, csv_file_path: str | Path):
        self.data = pd.read_csv(csv_file_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        features = torch.tensor(row.iloc[:-1].values, dtype=torch.float32)
        label = torch.tensor(row.iloc[-1], dtype=torch.float32).unsqueeze(0)
        return features, label

    @property
    def num_features(self) -> int:
        return self.data.shape[1] - 1

class SequenceLogDataset(Dataset):
    def __init__(self, df, pipeline, seq_len=10, is_train=True):
        """
        Loads df, applies the pipeline, and serves sliding windows for Transformers.
        """
        super().__init__()
        self.seq_len = seq_len
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['timestamp'] = df['timestamp'].astype(str)
        else:
            print("WARNING: No timestamp column found. Sequences may be meaningless!")
        
        # 3. Separate features and labels
        if 'label' in df.columns:
            y_raw = df['label'].values
            X_raw = df.drop('label', axis=1)
        else:
            y_raw = np.zeros(len(df))
            X_raw = df

        print(f"Processing {'training' if is_train else 'testing'} data...")
        if is_train:
            X_processed = pipeline.fit_transform(X_raw)
        else:
            X_processed = pipeline.transform(X_raw)
        
        # Massive continuous pytorch tensors
        self.features = torch.tensor(X_processed, dtype=torch.float32)
        self.labels = torch.tensor(y_raw, dtype=torch.float32)
        
        print(f"Dataset ready. Total rows: {len(self.features)}. Sequence length: {self.seq_len}.")

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x_window = self.features[idx : idx + self.seq_len]
        window_labels = self.labels[idx : idx + self.seq_len]
        
        # If ANY log in this sequence is an attack, the sequence is an anomaly
        is_anomaly = 1.0 if torch.any(window_labels > 0) else 0.0
        return x_window, torch.tensor([is_anomaly], dtype=torch.float32)
    

def write_processed_csv(
    df_raw: pd.DataFrame,
    label_col: str,
    preprocessor,
    out_path: str | Path,
    fit: bool = True,
) -> int:
    """Fit (or transform with) the preprocessor and dump features + label.

    Returns the number of feature columns written. The label column is
    appended last so `LogDataset` can split it off via `iloc[:-1]`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = df_raw.drop(columns=[label_col])
    y = df_raw[label_col]

    matrix = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    cols = preprocessor.get_feature_names_out()

    if hasattr(matrix, "toarray"):
        processed = pd.DataFrame(matrix.toarray(), columns=cols)
    else:
        processed = pd.DataFrame(matrix, columns=cols)

    processed["target_label"] = y.values
    processed.to_csv(out_path, index=False)
    return matrix.shape[1]
