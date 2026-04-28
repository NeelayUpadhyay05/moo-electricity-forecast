import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
import pandas as pd
import optuna

from src.utils.seed import set_seed
from src.models.lstm import LSTMModel, count_parameters
from src.training.training_pipeline import (
    train_single_configuration,
    retrain_and_evaluate
)
from src.config import Config


# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================================
# Result Saving
# ==========================================================
def save_results(out_dir, runtime, test_metrics, best_hyperparams, search_history, seed, mode):
    result = {
        "seed": seed,
        "mode": mode,
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_mape": float(test_metrics["mape"]),
        "test_r2": float(test_metrics["r2"]),
        "best_hyperparams": best_hyperparams,
        "search_history": search_history,
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
# Optuna (TPE)
# ==========================================================
def run_optuna(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):

    set_seed(seed)
    print("\n================ OPTUNA (TPE) =================")
    start = time.time()

    convergence = []
    best_val = float("inf")
    search_history = []

    def objective(trial):
        nonlocal best_val

        trial_config = Config(mode=config.mode)
        b = config.hp_bounds

        trial_config.hidden_dim = trial.suggest_int("hidden_dim", b["hidden_dim"][0], b["hidden_dim"][1])
        trial_config.num_layers = trial.suggest_int("num_layers", b["num_layers"][0], b["num_layers"][1])
        trial_config.lr = trial.suggest_float("lr", b["lr"][0], b["lr"][1], log=True)
        trial_config.dropout = trial.suggest_float("dropout", b["dropout"][0], b["dropout"][1])
        trial_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/optuna_trial_{trial.number}.pt"

        os.makedirs(os.path.dirname(trial_config.checkpoint_path), exist_ok=True)

        print(f"\n########## Optuna Trial {trial.number + 1}/{config.n_trials} ##########")
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

        best_val = min(best_val, val_mse)
        convergence.append(best_val)
        search_history.append({
            "trial": trial.number + 1,
            "hidden_dim": trial_config.hidden_dim,
            "num_layers": trial_config.num_layers,
            "lr": trial_config.lr,
            "dropout": trial_config.dropout,
            "val_mse": val_mse,
            "complexity": int(complexity),
        })

        return float(val_mse)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config.n_trials)

    # Single-objective: choose best trial by validation MSE
    best_trial = study.best_trial
    best_config = Config(mode=config.mode)
    best_config.hidden_dim = best_trial.params["hidden_dim"]
    best_config.num_layers = best_trial.params["num_layers"]
    best_config.lr = best_trial.params["lr"]
    best_config.dropout = best_trial.params["dropout"]
    best_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/optuna_best.pt"

    os.makedirs(os.path.dirname(best_config.checkpoint_path), exist_ok=True)

    print("\nRetraining Best Optuna Configuration...")

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/optuna"
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(search_history).to_csv(
        os.path.join(out_dir, "search_history.csv"), index=False
    )

    save_results(
        out_dir=out_dir,
        runtime=runtime,
        test_metrics=test_metrics,
        best_hyperparams={
            "hidden_dim": best_config.hidden_dim,
            "num_layers": best_config.num_layers,
            "lr": best_config.lr,
            "dropout": best_config.dropout,
        },
        search_history=search_history,
        seed=seed,
        mode=config.mode,
    )

    return float(best_trial.value), test_metrics["rmse"], runtime


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
    os.environ["EXPERIMENT_SEED"] = str(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(mode=args.mode)
    train_df, val_df, test_df, scaling_params = load_data(config, zone=args.zone)

    val_mse, test_rmse, runtime = run_optuna(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test NRMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Optuna (TPE)':<15}{val_mse:<15.6f}{test_rmse:<15.6f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
