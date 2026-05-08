import numpy as np
import torch
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    """
    Univariate sliding-window dataset for a single time series.

    Each sample:
        x : (seq_len, 1)  — normalized load values for the past seq_len hours
        y : scalar        — normalized load for the next hour (t + seq_len)

    Args:
        series  : pd.Series or np.ndarray of normalized load values
        seq_len : number of past time steps used as input (default: 24)
    """

    def __init__(self, series, seq_len: int = 24):
        if hasattr(series, "values"):
            data = series.values.astype(np.float32)
        else:
            data = np.asarray(series, dtype=np.float32)

        # Support both pd.Series (1-D) and single-column pd.DataFrame (2-D)
        if data.ndim > 1:
            data = data.squeeze(axis=1)

        self.data      = data
        self.seq_len   = seq_len
        self.n_samples = len(data) - seq_len  # each window predicts one step ahead

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]  # (seq_len,)
        y = self.data[idx + self.seq_len]         # scalar

        x = torch.tensor(x).unsqueeze(-1)         # (seq_len, 1)
        y = torch.tensor(y)                        # scalar

        return x, y
