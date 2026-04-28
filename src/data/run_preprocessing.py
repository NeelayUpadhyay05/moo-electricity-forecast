import argparse
import os
import json

from src.data.preprocess import (
    load_pjm_dataset,
    load_nyiso_dataset,
    rank_nyiso_regions_train_only,
    extract_region_series,
    split_chronological,
    normalize,
    save_processed_data,
    save_series_csv,
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
        description="Preprocess PJM or NYISO hourly electricity load series."
    )
    parser.add_argument(
        "--dataset", type=str, default="pjm", choices=["pjm", "nyiso"],
        help="Dataset to preprocess (default: pjm)"
    )
    parser.add_argument(
        "--zone", type=str, default="PJME", choices=list(ZONE_FILES.keys()),
        help="PJM zone to preprocess (used only when --dataset pjm)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw/PJM",
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/processed",
        help="Directory to write processed files"
    )
    parser.add_argument(
        "--selection_dir", type=str, default="data/processed/nyiso_selected",
        help="Directory to write NYISO selected-region CSVs"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of NYISO regions to select (default: 3)"
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

    if args.dataset == "pjm":
        filepath = os.path.join(args.data_dir, ZONE_FILES[args.zone])
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Raw file not found: {filepath}\n"
                f"Download PJM CSVs and place them inside {args.data_dir}/"
            )

        # ---------------------------------------------------------------- load
        print(f"--- Loading PJM {args.zone} ({ZONE_FILES[args.zone]}) ---")
        series = load_pjm_dataset(filepath)
        print(f"  Samples    : {len(series):,}")
        print(f"  Date range : {series.index.min()}  ->  {series.index.max()}")
        print(
            f"  Load (MW)  : mean={series.mean():.1f}  std={series.std():.1f}"
            f"  min={series.min():.1f}  max={series.max():.1f}"
        )

        # ------------------------------------------------------------- split
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

        # ----------------------------------------------------------- normalize
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

        # -------------------------------------------------------------- save
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
        return

    # ---------------------------------------------------------------- NYISO
    print(f"--- Loading NYISO yearly files from {args.data_dir} ---")
    combined = load_nyiso_dataset(args.data_dir)
    os.makedirs(args.selection_dir, exist_ok=True)
    combined_path = os.path.join(args.selection_dir, "nyiso_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"  Combined rows : {len(combined):,}")
    print(
        f"  Date range    : {combined['UTC Timestamp (Interval Ending)'].min()}"
        f"  ->  {combined['UTC Timestamp (Interval Ending)'].max()}"
    )
    print(f"  Combined raw saved to: {combined_path}")

    ranked_regions = rank_nyiso_regions_train_only(
        combined, top_k=args.top_k, train_ratio=args.train_ratio
    )
    summary_path = os.path.join(args.selection_dir, "nyiso_top_regions.json")
    with open(summary_path, "w") as f:
        json.dump(ranked_regions, f, indent=4)

    print("\n--- Selected NYISO regions ---")
    for item in ranked_regions:
        print(
            f"  {item['zone']:<20} {item['region_name']:<24} "
            f"std={item['std']:.1f} mean={item['mean']:.1f} score={item['score']:.1f}"
        )

    print(f"\n--- Saving selected region CSVs to {args.selection_dir}/ ---")
    for item in ranked_regions:
        series = extract_region_series(combined, item["raw_column"])
        raw_path = os.path.join(args.selection_dir, f"{item['zone']}.csv")
        save_series_csv(series, raw_path)
        print(f"  {item['zone']}.csv ({len(series):,} rows)")

        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        print(
            f"\n--- Processing {item['zone']} "
            f"(train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={test_ratio:.0%}) ---"
        )
        train, val, test = split_chronological(series, args.train_ratio, args.val_ratio)
        train_norm, val_norm, test_norm, scaling_params = normalize(train, val, test)
        save_processed_data(
            train_norm, val_norm, test_norm,
            scaling_params, item["zone"], args.save_dir
        )
        print(
            f"  Saved processed split files for {item['zone']}"
            f" -> {args.save_dir}/{item['zone']}_*.csv/json"
        )

    print(f"\nNYISO preprocessing complete. Selection summary: {summary_path}")


if __name__ == "__main__":
    main()
