import os
import glob
import json
import re
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
    series = df.iloc[:, 0]
    return clean_time_series(series)


def clean_time_series(series: pd.Series, interpolate: bool = True) -> pd.Series:
    """Standardize a datetime-indexed series for forecasting."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if interpolate and series.isna().any():
        series = series.interpolate(method="time")
        series = series.ffill().bfill()

    return series.dropna()


def load_nyiso_dataset(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all yearly NYISO raw CSV files."""
    file_pattern = os.path.join(data_dir, "nyiso_load_act_hr_*.csv")
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        raise FileNotFoundError(
            f"No NYISO raw files found in {data_dir}. Expected nyiso_load_act_hr_*.csv"
        )

    frames = []
    for filepath in filepaths:
        frame = pd.read_csv(filepath, skiprows=3)
        frame.columns = [column.strip() for column in frame.columns]
        frame["UTC Timestamp (Interval Ending)"] = pd.to_datetime(
            frame["UTC Timestamp (Interval Ending)"]
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("UTC Timestamp (Interval Ending)")
    combined = combined.drop_duplicates(
        subset=["UTC Timestamp (Interval Ending)"], keep="last"
    )
    return combined.reset_index(drop=True)


def nyiso_region_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.endswith("Actual Load (MW)")]


def _slugify_region_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug.upper()


def rank_nyiso_regions(df: pd.DataFrame, top_k: int = 3) -> list[dict]:
    """Rank NYISO regions by signal strength and completeness."""
    zone_cols = nyiso_region_columns(df)
    if len(zone_cols) < top_k:
        raise ValueError(
            f"Requested top_k={top_k}, but only found {len(zone_cols)} NYISO regions."
        )

    ranked = []
    for column in zone_cols:
        raw_name = column.replace(" Actual Load (MW)", "")
        if " - " in raw_name:
            region_code, region_name = raw_name.split(" - ", 1)
        else:
            region_code, region_name = raw_name, raw_name

        series = pd.to_numeric(df[column], errors="coerce")
        missing_fraction = float(series.isna().mean())
        completeness = 1.0 - missing_fraction
        std = float(series.std())
        mean = float(series.mean())
        score = completeness * std

        ranked.append(
            {
                "raw_column": column,
                "region_code": region_code,
                "region_name": region_name,
                "zone": _slugify_region_name(region_name),
                "missing_fraction": missing_fraction,
                "completeness": completeness,
                "mean": mean,
                "std": std,
                "score": score,
            }
        )

    ranked.sort(
        key=lambda item: (item["score"], item["std"], item["mean"]),
        reverse=True,
    )
    return ranked[:top_k]


def rank_nyiso_regions_train_only(
    df: pd.DataFrame, top_k: int = 3, train_ratio: float = 0.70
) -> list[dict]:
    """Rank NYISO regions using only each region's training-period segment.

    This avoids future/test leakage when choosing which NYISO regions to keep.
    """
    zone_cols = nyiso_region_columns(df)
    if len(zone_cols) < top_k:
        raise ValueError(
            f"Requested top_k={top_k}, but only found {len(zone_cols)} NYISO regions."
        )

    ranked = []
    for column in zone_cols:
        raw_name = column.replace(" Actual Load (MW)", "")
        if " - " in raw_name:
            region_code, region_name = raw_name.split(" - ", 1)
        else:
            region_code, region_name = raw_name, raw_name

        full_series = extract_region_series(df, column)
        train_end = int(len(full_series) * train_ratio)
        train_series = full_series.iloc[:train_end]
        if len(train_series) == 0:
            continue

        missing_fraction = float(train_series.isna().mean())
        completeness = 1.0 - missing_fraction
        std = float(train_series.std())
        mean = float(train_series.mean())
        score = completeness * std

        ranked.append(
            {
                "raw_column": column,
                "region_code": region_code,
                "region_name": region_name,
                "zone": _slugify_region_name(region_name),
                "missing_fraction": missing_fraction,
                "completeness": completeness,
                "mean": mean,
                "std": std,
                "score": score,
            }
        )

    ranked.sort(
        key=lambda item: (item["score"], item["std"], item["mean"]),
        reverse=True,
    )
    return ranked[:top_k]


def extract_region_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a NYISO region column into a cleaned datetime-indexed series."""
    timestamp_col = "UTC Timestamp (Interval Ending)"
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")

    series = pd.Series(
        pd.to_numeric(df[column], errors="coerce").values,
        index=pd.to_datetime(df[timestamp_col]),
        name=column,
    )
    return clean_time_series(series)


def save_series_csv(series: pd.Series, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    series.to_csv(filepath, header=True)


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
