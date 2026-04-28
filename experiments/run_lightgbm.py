import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import pandas as pd

from src.models.lightgbm_model import train_lightgbm, predict_lightgbm
from src.metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_r2
from src.config import Config
from src.utils.seed import set_seed


def build_lag_features(series, lags=24):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


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


def run_lightgbm(train_df, val_df, test_df, scaling_params, config, seed=42, zone="PJME"):
    set_seed(seed)
    print("\n================ LightGBM =================")
    start = time.time()

    lags = config.seq_len
    mean = scaling_params["mean"]
    std = scaling_params["std"]
    
    train_series = train_df.iloc[:, 0].values
    val_series = val_df.iloc[:, 0].values
    test_series = test_df.iloc[:, 0].values

    X_train, y_train = build_lag_features(np.concatenate([train_series, val_series]), lags=lags)
    model = train_lightgbm(X_train, y_train, params=None, num_boost_round=100)

    # Build test features using trailing lags from combined
    combined = np.concatenate([train_series, val_series, test_series])
    X_all, y_all = build_lag_features(combined, lags=lags)
    # test starts after len(train)+len(val): compute index
    start_idx = len(train_series) + len(val_series) - lags
    X_test = X_all[start_idx: start_idx + len(test_series)]
    preds = predict_lightgbm(model, X_test)[:len(test_series)]

    # Compute metrics
    # Predictions and test_series are stored normalized in processed files.
    # Convert back to original MW scale before RMSE/MAE/MAPE for comparability.
    preds_orig = preds * std + mean
    test_orig = test_series * std + mean

    rmse = calculate_rmse(preds_orig, test_orig)
    mae = calculate_mae(preds_orig, test_orig)
    mape = calculate_mape(preds_orig, test_orig)

    # R2 computed on normalized values for comparability with model training
    preds_norm = preds
    test_norm = test_series
    r2 = calculate_r2(preds_norm, test_norm)

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/lightgbm"
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

    run_lightgbm(train_df, val_df, test_df, scaling_params, config, seed=args.seed, zone=args.zone)


if __name__ == "__main__":
    main()
