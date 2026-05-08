import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess_selected_zones import (
    load_pjm_dataset,
    load_nyiso_dataset,
    extract_region_series,
    load_india_dataset,
    split_chronological,
    normalize,
    save_processed_data,
    save_series_csv,
)


ZONE_MAP = {
    "PJM": ["PJME", "AEP"],
    "NYISO": ["NEW_YORK_CITY", "LONG_ISLAND"],
    "INDIA": ["NATIONAL", "NORTHERN"],
}


def process_pjm(data_dir, save_dir, zones, train_ratio, val_ratio):
    for zone in zones:
        filename = f"{zone}_hourly.csv"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] PJM raw file not found: {path}. Skipping {zone}.")
            continue

        print(f"Processing PJM zone: {zone}")
        series = load_pjm_dataset(path)
        train, val, test = split_chronological(series, train_ratio, val_ratio)
        train_n, val_n, test_n, scaling = normalize(train, val, test)
        save_processed_data(train_n, val_n, test_n, scaling, zone, save_dir)
        print(f"  Saved processed {zone} to {save_dir}")


def process_nyiso(data_dir, save_dir, zones, selection_dir, train_ratio, val_ratio):
    print(f"Loading NYISO files from {data_dir}")
    combined = load_nyiso_dataset(data_dir)
    os.makedirs(selection_dir, exist_ok=True)
    combined.to_csv(os.path.join(selection_dir, "nyiso_combined.csv"), index=False)
    for zone in zones:
        # Attempt to find columns matching zone names
        candidates = [c for c in combined.columns if zone.replace('_', ' ').upper().replace('  ', ' ') in c.upper() or c.endswith('Actual Load (MW)') and zone in c.upper()]
        # Preferred explicit mapping for known zones
        mapping = {
            'NEW_YORK_CITY': 'J - New York City Actual Load (MW)',
            'LONG_ISLAND': 'K - Long Island Actual Load (MW)'
        }
        col = mapping.get(zone, None)
        if col is None:
            # fallback: find by slug
            matches = [c for c in combined.columns if zone in c.upper()]
            col = matches[0] if matches else None

        if col is None or col not in combined.columns:
            print(f"[WARN] Could not find NYISO column for {zone}. Skipping.")
            continue

        series = extract_region_series(combined, col)
        raw_path = os.path.join(selection_dir, f"{zone}.csv")
        save_series_csv(series, raw_path)
        train, val, test = split_chronological(series, train_ratio, val_ratio)
        train_n, val_n, test_n, scaling = normalize(train, val, test)
        save_processed_data(train_n, val_n, test_n, scaling, zone, save_dir)
        print(f"  Saved processed {zone} to {save_dir}")


def process_india(filepath, save_dir, zones, train_ratio, val_ratio):
    if not os.path.exists(filepath):
        print(f"[WARN] India raw file not found: {filepath}. Skipping India processing.")
        return

    df = load_india_dataset(filepath)
    dt_col = df.columns[0]
    for zone in zones:
        # Map expected column names
        mapping = {
            'NATIONAL': 'National Hourly Demand',
            'NORTHERN': 'Northen Region Hourly Demand',
        }
        col = mapping.get(zone)
        if col not in df.columns:
            print(f"[WARN] India column '{col}' not found. Available: {list(df.columns)[:10]}")
            continue

        series = pd.Series(pd.to_numeric(df[col], errors='coerce').values, index=pd.to_datetime(df[dt_col]))
        series = clean_wrapper(series)
        train, val, test = split_chronological(series, train_ratio, val_ratio)
        train_n, val_n, test_n, scaling = normalize(train, val, test)
        save_processed_data(train_n, val_n, test_n, scaling, zone, save_dir)
        print(f"  Saved processed {zone} to {save_dir}")


def clean_wrapper(series):
    # lightweight wrapper to avoid circular import
    series = pd.to_numeric(series, errors='coerce')
    series = series.sort_index()
    series = series[~series.index.duplicated(keep='last')]
    if series.isna().any():
        series = series.interpolate(method='time')
        series = series.ffill().bfill()
    return series.dropna()


def main():
    parser = argparse.ArgumentParser(description='Preprocess selected zones for PJM/NYISO/India')
    parser.add_argument('--data_dir_pjm', default='data/raw/PJM')
    parser.add_argument('--data_dir_nyiso', default='data/raw/NYISO')
    parser.add_argument('--data_file_india', default='data/raw/India/hourlyLoadDataIndia.csv')
    parser.add_argument('--save_dir', default='data/processed')
    parser.add_argument('--selection_dir', default='data/processed/nyiso_selected')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()

    # PJM
    process_pjm(args.data_dir_pjm, args.save_dir, ZONE_MAP['PJM'], args.train_ratio, args.val_ratio)

    # NYISO
    process_nyiso(args.data_dir_nyiso, args.save_dir, ZONE_MAP['NYISO'], args.selection_dir, args.train_ratio, args.val_ratio)

    # India
    process_india(args.data_file_india, args.save_dir, ZONE_MAP['INDIA'], args.train_ratio, args.val_ratio)

    print('Preprocessing completed.')


if __name__ == '__main__':
    main()
