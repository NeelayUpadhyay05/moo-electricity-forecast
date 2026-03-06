import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import random
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
def save_results(out_dir, runtime, val_mse, test_rmse, best_hyperparams, convergence):
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "runtime_s": round(runtime, 2),
        "best_val_mse": float(val_mse),
        "best_test_rmse": float(test_rmse),
        "best_hyperparams": best_hyperparams,
        "convergence": [float(v) for v in convergence],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {out_dir}/metrics.json")


# ==========================================================
# Data Loading (Mode Aware)
# ==========================================================
def load_data(config):

    train_df = pd.read_csv(
        "data/processed/electricity_train.csv",
        index_col=0,
        parse_dates=True
    )
    val_df = pd.read_csv(
        "data/processed/electricity_val.csv",
        index_col=0,
        parse_dates=True
    )
    test_df = pd.read_csv(
        "data/processed/electricity_test.csv",
        index_col=0,
        parse_dates=True
    )

    with open("data/processed/electricity_scaling.json") as f:
        scaling_params = json.load(f)

    if config.mode == "dev":
        print("\n[DEV MODE ACTIVE]")
        print(f"Using first {config.dev_households} households")
        print(f"Using first {config.dev_timesteps} timesteps")

        train_df = train_df.iloc[:config.dev_timesteps, :config.dev_households]
        val_df = val_df.iloc[:config.dev_timesteps, :config.dev_households]
        test_df = test_df.iloc[:config.dev_timesteps, :config.dev_households]

    return train_df, val_df, test_df, scaling_params


# ==========================================================
# Random Search
# ==========================================================
def run_random_search(train_df, val_df, test_df, scaling_params, device, config, seed=42):

    print("\n================ RANDOM SEARCH =================")
    start = time.time()

    best_val = float("inf")
    best_config = None
    convergence = []
    search_history = []

    for trial in range(config.random_trials):

        print(f"\n########## Trial {trial+1}/{config.random_trials} ##########")

        trial_config = Config(mode=config.mode)
        trial_config.hidden_dim = random.choice([32, 64, 128])
        trial_config.lr = random.uniform(1e-4, 5e-3)
        trial_config.dropout = random.uniform(0.0, 0.3)
        trial_config.checkpoint_path = f"checkpoints/seed_{seed}/random_trial.pt"

        os.makedirs(os.path.dirname(trial_config.checkpoint_path), exist_ok=True)

        print(
            f"Sampled -> hidden_dim={trial_config.hidden_dim}, "
            f"lr={trial_config.lr:.6f}, "
            f"dropout={trial_config.dropout:.3f}"
        )

        val_mse = train_single_configuration(
            train_df, val_df, device, trial_config
        )

        print(f"Validation MSE: {val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_config = trial_config
            print(">> New Best Found!")

        convergence.append(best_val)
        search_history.append({
            "trial": trial + 1,
            "hidden_dim": trial_config.hidden_dim,
            "lr": trial_config.lr,
            "dropout": trial_config.dropout,
            "val_mse": val_mse,
        })

    print("\nRetraining Best Random Configuration...")

    best_config.checkpoint_path = f"checkpoints/seed_{seed}/random_best.pt"

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/random_search"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(search_history).to_csv(
        os.path.join(out_dir, "search_history.csv"), index=False
    )
    save_results(
        out_dir=out_dir,
        runtime=runtime,
        val_mse=best_val,
        test_rmse=test_rmse,
        best_hyperparams={
            "hidden_dim": best_config.hidden_dim,
            "lr": best_config.lr,
            "dropout": best_config.dropout,
        },
        convergence=convergence,
    )

    return best_val, test_rmse, runtime


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = Config(mode="full")
    train_df, val_df, test_df, scaling_params = load_data(config)

    val_mse, test_rmse, runtime = run_random_search(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Random Search':<15}{val_mse:<15.6f}{test_rmse:<15.4f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
