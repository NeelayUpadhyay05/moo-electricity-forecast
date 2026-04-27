import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import pandas as pd

from src.models.arima import fit_arima, forecast_arima
from src.metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_r2
from src.config import Config
from src.utils.seed import set_seed


def load_data(config, zone="PJME"):
    base = f"data/processed/{zone}"
    train_df = pd.read_csv(f"{base}_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{base}_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(f"{base}_test.csv", index_col=0, parse_dates=True)
    with open(f"{base}_scaling.json") as f:
        scaling_params = json.load(f)
    if config.mode == "dev":
        train_df = train_df.iloc[:config.dev_timesteps]
        val_df = val_df.iloc[:config.dev_timesteps]
        test_df = test_df.iloc[:config.dev_timesteps]
    return train_df, val_df, test_df, scaling_params


def run_arima(train_df, val_df, test_df, scaling_params, config, seed=42, zone="PJME"):
    set_seed(seed)
    print("\n================ ARIMA =================")
    start = time.time()

    mean = scaling_params["mean"]
    std = scaling_params["std"]
    
    combined = pd.concat([train_df, val_df]).iloc[:, 0].values
    test_vals = test_df.iloc[:, 0].values

    fitted = fit_arima(combined, order=(5, 1, 0))
    preds = forecast_arima(fitted, steps=len(test_vals))

    # Compute metrics
    rmse = calculate_rmse(preds, test_vals)
    mae = calculate_mae(preds, test_vals)
    mape = calculate_mape(preds, test_vals)
    
    preds_norm = (preds - mean) / std
    test_norm = (test_vals - mean) / std
    r2 = calculate_r2(preds_norm, test_norm)

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/arima"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "seed": seed,
        "mode": config.mode,
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_mape": float(mape),
        "test_r2": float(r2),
        "runtime": runtime,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {out_dir}/metrics.json")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["dev", "full"])
    parser.add_argument("--zone", type=str, default="PJME")
    args = parser.parse_args()

    config = Config(mode=args.mode)
    train_df, val_df, test_df, scaling_params = load_data(config, zone=args.zone)

    run_arima(train_df, val_df, test_df, scaling_params, config, seed=args.seed, zone=args.zone)


if __name__ == "__main__":
    main()
