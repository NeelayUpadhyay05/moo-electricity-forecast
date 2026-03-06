import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import torch
import pandas as pd

from src.utils.seed import set_seed
from src.training.training_pipeline import (
    train_single_configuration,
    retrain_and_evaluate
)
from src.optimizers.moo import MOOOptimizer
from src.training.fitness import moo_fitness
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
# MOO
# ==========================================================
def run_moo(train_df, val_df, test_df, scaling_params, device, config, seed=42):

    print("\n================ MOO =================")
    start = time.time()

    b = config.hp_bounds
    bounds = [tuple(b["hidden_dim"]), tuple(b["lr"]), tuple(b["dropout"])]

    moo = MOOOptimizer(
        fitness_fn=lambda particle: moo_fitness(
            particle, train_df, val_df, device, mode=config.mode
        ),
        bounds=bounds,
        pop_size=config.moo_pop_size,
        generations=config.moo_generations,
        seed=seed,
    )

    pareto_solutions, history = moo.optimize()
    evaluated = []

    for solution in pareto_solutions:

        hidden_dim, lr, dropout = solution["params"]

        sol_config = Config(mode=config.mode)
        sol_config.hidden_dim = int(np.round(hidden_dim))
        sol_config.lr = float(lr)
        sol_config.dropout = float(dropout)
        sol_config.checkpoint_path = f"checkpoints/seed_{seed}/moo_hidden{int(hidden_dim)}.pt"

        os.makedirs(os.path.dirname(sol_config.checkpoint_path), exist_ok=True)

        test_rmse = retrain_and_evaluate(
            train_df, val_df, test_df,
            device, sol_config, scaling_params
        )

        evaluated.append({
            "val_mse": solution["val_mse"],
            "complexity": solution["complexity"],
            "test_rmse": test_rmse,
            "hyperparams": {
                "hidden_dim": sol_config.hidden_dim,
                "lr": sol_config.lr,
                "dropout": sol_config.dropout,
            },
        })

    runtime = time.time() - start

    best_val_solution  = min(evaluated, key=lambda x: x["val_mse"])
    best_test_solution = min(evaluated, key=lambda x: x["test_rmse"])

    out_dir = f"results/seed_{seed}/moo"
    os.makedirs(out_dir, exist_ok=True)
    pareto_rows = [
        {
            "hidden_dim": e["hyperparams"]["hidden_dim"],
            "lr":         e["hyperparams"]["lr"],
            "dropout":    e["hyperparams"]["dropout"],
            "val_mse":    float(e["val_mse"]),
            "complexity": float(e["complexity"]),
            "test_rmse":  float(e["test_rmse"]),
        }
        for e in evaluated
    ]
    pd.DataFrame(pareto_rows).to_csv(
        os.path.join(out_dir, "pareto_front.csv"), index=False
    )
    save_results(
        out_dir=out_dir,
        runtime=runtime,
        val_mse=best_val_solution["val_mse"],
        test_rmse=best_test_solution["test_rmse"],
        best_hyperparams=best_val_solution["hyperparams"],
        convergence=history,
    )

    return (
        best_val_solution["val_mse"],
        best_test_solution["test_rmse"],
        runtime
    )


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

    val_mse, test_rmse, runtime = run_moo(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'MOO':<15}{val_mse:<15.6f}{test_rmse:<15.4f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
