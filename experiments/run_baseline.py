import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
import pandas as pd

from src.utils.seed import set_seed
from src.training.training_pipeline import (
    train_single_configuration,
    retrain_and_evaluate
)
from src.config import Config


# ==========================================================
# Result Saving
# ==========================================================
def save_results(out_dir, runtime, val_mse, test_metrics, best_hyperparams, convergence, seed, mode):
    result = {
        "seed": seed,
        "mode": mode,
        "runtime_s": round(runtime, 2),
        "best_val_mse": float(val_mse),
        "best_test_nrmse": float(test_metrics["nrmse"]),
        "best_test_rmse": float(test_metrics["rmse"]),
        "best_test_mae": float(test_metrics["mae"]),
        "best_test_mape": float(test_metrics["mape"]),
        "best_hyperparams": best_hyperparams,
        "convergence": [float(v) for v in convergence],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {out_dir}/metrics.json")


# ==========================================================
# Data Loading (Mode Aware)
# ==========================================================
def load_data(config, zone="PJME"):

    base = f"data/processed/{zone}"
    train_df = pd.read_csv(f"{base}_train.csv", index_col=0, parse_dates=True)
    val_df   = pd.read_csv(f"{base}_val.csv",   index_col=0, parse_dates=True)
    test_df  = pd.read_csv(f"{base}_test.csv",  index_col=0, parse_dates=True)

    with open(f"{base}_scaling.json") as f:
        scaling_params = json.load(f)

    if config.mode == "dev":
        print(f"\n[DEV MODE ACTIVE] zone={zone}, timesteps={config.dev_timesteps}")
        train_df = train_df.iloc[:config.dev_timesteps]
        val_df   = val_df.iloc[:config.dev_timesteps]
        test_df  = test_df.iloc[:config.dev_timesteps]

    return train_df, val_df, test_df, scaling_params


# ==========================================================
# Baseline
# ==========================================================
def run_baseline(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):

    set_seed(seed)
    print("\n================ BASELINE =================")
    start = time.time()

    base_config = Config(mode=config.mode)
    base_config.hidden_dim = 128
    base_config.num_layers = 1
    base_config.lr = 0.004
    base_config.dropout = 0.0
    base_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/baseline_best.pt"

    os.makedirs(os.path.dirname(base_config.checkpoint_path), exist_ok=True)

    val_mse = train_single_configuration(
        train_df, val_df, device, base_config
    )

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, base_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/baseline"
    os.makedirs(out_dir, exist_ok=True)
    save_results(
        out_dir=out_dir,
        runtime=runtime,
        val_mse=val_mse,
        test_metrics=test_metrics,
        best_hyperparams={
            "hidden_dim": 128, "num_layers": 1, "lr": 0.004, "dropout": 0.0
        },
        convergence=[],
        seed=seed,
        mode=config.mode,
    )

    return val_mse, test_metrics["nrmse"], runtime


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["dev", "full"])
    parser.add_argument("--zone", type=str, default="PJME")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(mode=args.mode)
    train_df, val_df, test_df, scaling_params = load_data(config, zone=args.zone)

    val_mse, test_rmse, runtime = run_baseline(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test NRMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Baseline':<15}{val_mse:<15.6f}{test_rmse:<15.6f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
