import argparse
import os

from src.data.preprocess import (
    load_pjm_dataset,
    split_chronological,
    normalize,
    save_processed_data,
)

# Maps CLI zone names to their raw CSV filenames
ZONE_FILES = {
    "PJME":   "PJME_hourly.csv",
    "AEP":    "AEP_hourly.csv",
    "DAYTON": "DAYTON_hourly.csv",
    "DUQ":    "DUQ_hourly.csv",
    "COMED":  "COMED_hourly.csv",
    "DOM":    "DOM_hourly.csv",
    "DEOK":   "DEOK_hourly.csv",
    "EKPC":   "EKPC_hourly.csv",
    "FE":     "FE_hourly.csv",
    "NI":     "NI_hourly.csv",
    "PJMW":   "PJMW_hourly.csv",
}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a PJM hourly electricity load series."
    )
    parser.add_argument(
        "--zone", type=str, required=True, choices=list(ZONE_FILES.keys()),
        help="PJM zone to preprocess (e.g. PJME)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw/PJM",
        help="Directory containing raw PJM CSV files"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/processed",
        help="Directory to write processed files"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70,
        help="Fraction of data used for training (default: 0.70)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Fraction of data used for validation (default: 0.15)"
    )
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be < 1.0")

    filepath = os.path.join(args.data_dir, ZONE_FILES[args.zone])
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw file not found: {filepath}\n"
            f"Download PJM CSVs and place them inside {args.data_dir}/"
        )

    # ------------------------------------------------------------------ load
    print(f"--- Loading {args.zone} ({ZONE_FILES[args.zone]}) ---")
    series = load_pjm_dataset(filepath)
    print(f"  Samples    : {len(series):,}")
    print(f"  Date range : {series.index.min()}  ->  {series.index.max()}")
    print(
        f"  Load (MW)  : mean={series.mean():.1f}  std={series.std():.1f}"
        f"  min={series.min():.1f}  max={series.max():.1f}"
    )

    # ----------------------------------------------------------------- split
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(
        f"\n--- Splitting  "
        f"(train={args.train_ratio:.0%} / "
        f"val={args.val_ratio:.0%} / "
        f"test={test_ratio:.0%}) ---"
    )
    train, val, test = split_chronological(series, args.train_ratio, args.val_ratio)
    print(
        f"  Train : {len(train):,} hours  "
        f"({train.index.min().date()} -> {train.index.max().date()})"
    )
    print(
        f"  Val   : {len(val):,} hours  "
        f"({val.index.min().date()} -> {val.index.max().date()})"
    )
    print(
        f"  Test  : {len(test):,} hours  "
        f"({test.index.min().date()} -> {test.index.max().date()})"
    )

    # --------------------------------------------------------------- normalize
    print("\n--- Normalizing (z-score, train statistics only) ---")
    train_norm, val_norm, test_norm, scaling_params = normalize(train, val, test)
    print(
        f"  Train mean : {scaling_params['mean']:.4f} MW"
        f"   std : {scaling_params['std']:.4f} MW"
    )
    print(
        f"  Normalized train: mean={train_norm.mean():.6f}"
        f"  std={train_norm.std():.6f}"
    )

    # ------------------------------------------------------------------ save
    print(f"\n--- Saving to {args.save_dir}/ ---")
    save_processed_data(
        train_norm, val_norm, test_norm,
        scaling_params, args.zone, args.save_dir
    )
    print(f"  {args.zone}_train.csv    ({len(train_norm):,} rows)")
    print(f"  {args.zone}_val.csv      ({len(val_norm):,} rows)")
    print(f"  {args.zone}_test.csv     ({len(test_norm):,} rows)")
    print(f"  {args.zone}_scaling.json")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
