import os
import glob
import json
import re
from typing import List, Dict

import pandas as pd


def clean_time_series(series: pd.Series, interpolate: bool = True) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if interpolate and series.isna().any():
        series = series.interpolate(method="time")
        series = series.ffill().bfill()

    return series.dropna()


def load_pjm_dataset(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    series = df.iloc[:, 0]
    return clean_time_series(series)


def load_nyiso_dataset(data_dir: str) -> pd.DataFrame:
    file_pattern = os.path.join(data_dir, "nyiso_load_act_hr_*.csv")
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        raise FileNotFoundError(f"No NYISO raw files found in {data_dir}. Expected nyiso_load_act_hr_*.csv")

    frames = []
    for filepath in filepaths:
        frame = pd.read_csv(filepath, skiprows=3)
        frame.columns = [column.strip() for column in frame.columns]
        frame["UTC Timestamp (Interval Ending)"] = pd.to_datetime(frame["UTC Timestamp (Interval Ending)"])
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("UTC Timestamp (Interval Ending)")
    combined = combined.drop_duplicates(subset=["UTC Timestamp (Interval Ending)"], keep="last")
    return combined.reset_index(drop=True)


def nyiso_region_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("Actual Load (MW)")]


def _slugify_region_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug.upper()


def extract_region_series(df: pd.DataFrame, column: str) -> pd.Series:
    timestamp_col = "UTC Timestamp (Interval Ending)"
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")

    series = pd.Series(
        pd.to_numeric(df[column], errors="coerce").values,
        index=pd.to_datetime(df[timestamp_col]),
        name=column,
    )
    return clean_time_series(series)


def load_india_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=[0])
    # Ensure datetime column
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df


def split_chronological(series: pd.Series, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    return train, val, test


def normalize(train: pd.Series, val: pd.Series, test: pd.Series):
    mean = float(train.mean())
    std = float(train.std())

    if std < 1e-8:
        raise ValueError("Training series has near-zero standard deviation.")

    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std

    scaling_params = {"mean": mean, "std": std}
    return train_norm, val_norm, test_norm, scaling_params


def save_processed_data(train: pd.Series, val: pd.Series, test: pd.Series, scaling_params: Dict, zone: str, save_dir: str = "data/processed"):
    os.makedirs(save_dir, exist_ok=True)

    train.to_csv(os.path.join(save_dir, f"{zone}_train.csv"), header=True)
    val.to_csv(os.path.join(save_dir, f"{zone}_val.csv"), header=True)
    test.to_csv(os.path.join(save_dir, f"{zone}_test.csv"), header=True)

    with open(os.path.join(save_dir, f"{zone}_scaling.json"), "w") as f:
        json.dump(scaling_params, f, indent=4)


def save_series_csv(series: pd.Series, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    series.to_csv(filepath, header=True)
