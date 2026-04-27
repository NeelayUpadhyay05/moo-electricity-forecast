import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import math
import random
import torch
import pandas as pd

from src.utils.seed import set_seed
from src.models.lstm import LSTMModel, count_parameters
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
        "best_val_mse": float(val_mse),
        "best_test_nrmse": float(test_metrics["nrmse"]),
        "best_test_rmse": float(test_metrics["rmse"]),
        "best_test_mae": float(test_metrics["mae"]),
        "best_test_mape": float(test_metrics["mape"]),
        "best_complexity": int(best_hyperparams["complexity"]),
        "objectives": ["val_mse", "complexity"],
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
# Random Search
# ==========================================================
def dominates(a, b):
    return ((a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]))


def run_random_search(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):

    set_seed(seed)
    print("\n================ RANDOM SEARCH =================")
    start = time.time()

    best_val = float("inf")
    convergence = []
    search_history = []
    pareto_archive = []

    for trial in range(config.n_trials):

        print(f"\n########## Trial {trial+1}/{config.n_trials} ##########")

        trial_config = Config(mode=config.mode)
        b = config.hp_bounds
        trial_config.hidden_dim = random.randint(b["hidden_dim"][0], b["hidden_dim"][1])
        trial_config.num_layers = random.randint(b["num_layers"][0], b["num_layers"][1])
        trial_config.lr = 10 ** random.uniform(math.log10(b["lr"][0]), math.log10(b["lr"][1]))
        trial_config.dropout = random.uniform(b["dropout"][0], b["dropout"][1])
        trial_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/random_trial.pt"

        os.makedirs(os.path.dirname(trial_config.checkpoint_path), exist_ok=True)

        print(
            f"Sampled -> hidden_dim={trial_config.hidden_dim}, "
            f"num_layers={trial_config.num_layers}, "
            f"lr={trial_config.lr:.6f}, "
            f"dropout={trial_config.dropout:.3f}"
        )

        val_mse = train_single_configuration(
            train_df, val_df, device, trial_config
        )

        model = LSTMModel(
            hidden_dim=trial_config.hidden_dim,
            num_layers=trial_config.num_layers,
            dropout=trial_config.dropout,
        )
        complexity = count_parameters(model)

        print(f"Validation MSE: {val_mse:.6f} | Complexity: {complexity}")

        if val_mse < best_val:
            best_val = val_mse
            print(">> New Best Found!")

        convergence.append(best_val)
        search_history.append({
            "trial": trial + 1,
            "hidden_dim": trial_config.hidden_dim,
            "num_layers": trial_config.num_layers,
            "lr": trial_config.lr,
            "dropout": trial_config.dropout,
            "val_mse": val_mse,
            "complexity": int(complexity),
        })

        current = {
            "hidden_dim": trial_config.hidden_dim,
            "num_layers": trial_config.num_layers,
            "lr": trial_config.lr,
            "dropout": trial_config.dropout,
            "val_mse": float(val_mse),
            "complexity": int(complexity),
        }

        candidate = (current["val_mse"], current["complexity"])
        dominated = False
        filtered = []
        for item in pareto_archive:
            point = (item["val_mse"], item["complexity"])
            if dominates(point, candidate):
                dominated = True
                break
            if dominates(candidate, point):
                continue
            filtered.append(item)

        if not dominated:
            filtered.append(current)
            pareto_archive = filtered

    best_solution = min(pareto_archive, key=lambda x: x["val_mse"])

    print("\nRetraining Best Random Configuration...")

    best_config = Config(mode=config.mode)
    best_config.hidden_dim = best_solution["hidden_dim"]
    best_config.num_layers = best_solution["num_layers"]
    best_config.lr = best_solution["lr"]
    best_config.dropout = best_solution["dropout"]
    best_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/random_best.pt"

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/random_search"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(search_history).to_csv(
        os.path.join(out_dir, "search_history.csv"), index=False
    )
    pd.DataFrame(pareto_archive).to_csv(
        os.path.join(out_dir, "pareto_front.csv"), index=False
    )
    save_results(
        out_dir=out_dir,
        runtime=runtime,
        val_mse=best_solution["val_mse"],
        test_metrics=test_metrics,
        best_hyperparams={
            "hidden_dim": best_config.hidden_dim,
            "num_layers": best_config.num_layers,
            "lr": best_config.lr,
            "dropout": best_config.dropout,
            "complexity": best_solution["complexity"],
        },
        convergence=convergence,
        seed=seed,
        mode=config.mode,
    )

    return best_solution["val_mse"], test_metrics["nrmse"], runtime


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

    val_mse, test_rmse, runtime = run_random_search(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test NRMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Random Search':<15}{val_mse:<15.6f}{test_rmse:<15.6f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
