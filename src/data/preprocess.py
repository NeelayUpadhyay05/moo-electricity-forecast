import os
import json
import pandas as pd


def load_pjm_dataset(filepath: str) -> pd.Series:
    """
    Load a single PJM hourly CSV file.

    Expected format: two columns — a Datetime column (index) and a load column
    named {ZONE}_MW (e.g. PJME_MW, AEP_MW).

    The four DST duplicate timestamps per file (autumn clock-rollback) are
    removed by keeping the last reading at each duplicated timestamp.
    """
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    series = df.iloc[:, 0].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def split_chronological(series: pd.Series,
                         train_ratio: float = 0.70,
                         val_ratio:   float = 0.15):
    """
    Split a time series into train / val / test chronologically.
    test_ratio = 1 - train_ratio - val_ratio (default 0.15).
    """
    n         = len(series)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = series.iloc[:train_end]
    val   = series.iloc[train_end:val_end]
    test  = series.iloc[val_end:]

    return train, val, test


def normalize(train: pd.Series, val: pd.Series, test: pd.Series):
    """
    Z-score normalization using training statistics only.
    Val and test are transformed with the same mean / std — values outside
    the training range are left as-is (no clipping); the model sees them as
    slightly out-of-distribution, which is realistic.

    Returns:
        train_norm, val_norm, test_norm  — normalized pd.Series
        scaling_params                   — dict with 'mean' and 'std'
    """
    mean = float(train.mean())
    std  = float(train.std())

    if std < 1e-8:
        raise ValueError(
            "Training series has near-zero standard deviation. "
            "Check the raw data for constant or near-constant values."
        )

    train_norm = (train - mean) / std
    val_norm   = (val   - mean) / std
    test_norm  = (test  - mean) / std

    scaling_params = {"mean": mean, "std": std}

    return train_norm, val_norm, test_norm, scaling_params


def save_processed_data(train: pd.Series, val: pd.Series, test: pd.Series,
                         scaling_params: dict, zone: str,
                         save_dir: str = "data/processed"):
    """
    Save the three normalized series and the scaling parameters.

    Output files:
        {zone}_train.csv
        {zone}_val.csv
        {zone}_test.csv
        {zone}_scaling.json
    """
    os.makedirs(save_dir, exist_ok=True)

    train.to_csv(os.path.join(save_dir, f"{zone}_train.csv"), header=True)
    val.to_csv(  os.path.join(save_dir, f"{zone}_val.csv"),   header=True)
    test.to_csv( os.path.join(save_dir, f"{zone}_test.csv"),  header=True)

    with open(os.path.join(save_dir, f"{zone}_scaling.json"), "w") as f:
        json.dump(scaling_params, f, indent=4)
