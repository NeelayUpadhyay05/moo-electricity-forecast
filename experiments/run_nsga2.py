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
from src.optimizers.nsga2 import NSGA2
from src.training.fitness import moo_fitness
from src.metrics import calculate_hypervolume, calculate_igd
from src.config import Config


# ==========================================================
# Result Saving
# ==========================================================
def save_results(out_dir, runtime, test_metrics, best_hyperparams, pareto_objectives, seed, mode):
    hv = calculate_hypervolume(pareto_objectives)
    
    result = {
        "seed": seed,
        "mode": mode,
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_mape": float(test_metrics["mape"]),
        "test_r2": float(test_metrics["r2"]),
        "hypervolume": float(hv),
        "objectives": ["val_mse", "complexity"],
        "best_hyperparams": best_hyperparams,
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
# NSGA-II Runner
# ==========================================================
def run_nsga2(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):

    set_seed(seed)
    print("\n================ NSGA-II =================")
    start = time.time()

    b = config.hp_bounds
    bounds = [
        tuple(b["hidden_dim"]),
        (b["num_layers"][0] - 0.5, b["num_layers"][1] + 0.5),
        (np.log10(b["lr"][0]), np.log10(b["lr"][1])),
        tuple(b["dropout"]),
    ]

    nsga = NSGA2(
        fitness_fn=lambda particle: moo_fitness(
            particle, train_df, val_df, device, mode=config.mode
        ),
        bounds=bounds,
        pop_size=config.moo_pop_size,
        generations=config.moo_generations,
        seed=seed,
    )

    pareto_solutions, history = nsga.optimize()

    evaluated = []
    for solution in pareto_solutions:
        hidden_dim, num_layers, lr, dropout = solution["params"]
        evaluated.append({
            "val_mse":    solution["val_mse"],
            "complexity": solution["complexity"],
            "hyperparams": {
                "hidden_dim": int(np.round(hidden_dim)),
                "num_layers": int(np.round(num_layers)),
                "lr":         float(10 ** lr),
                "dropout":    float(dropout),
            },
        })

    best_val_solution = min(evaluated, key=lambda x: x["val_mse"])
    best_hp = best_val_solution["hyperparams"]

    best_config = Config(mode=config.mode)
    best_config.hidden_dim = best_hp["hidden_dim"]
    best_config.num_layers = best_hp["num_layers"]
    best_config.lr         = best_hp["lr"]
    best_config.dropout    = best_hp["dropout"]
    best_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/nsga2_best.pt"

    os.makedirs(os.path.dirname(best_config.checkpoint_path), exist_ok=True)

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/nsga2"
    os.makedirs(out_dir, exist_ok=True)

    pareto_rows = [
        {
            "hidden_dim": e["hyperparams"]["hidden_dim"],
            "num_layers": e["hyperparams"]["num_layers"],
            "lr":         e["hyperparams"]["lr"],
            "dropout":    e["hyperparams"]["dropout"],
            "val_mse":    float(e["val_mse"]),
            "complexity": float(e["complexity"]),
        }
        for e in evaluated
    ]
    pd.DataFrame(pareto_rows).to_csv(
        os.path.join(out_dir, "pareto_front.csv"), index=False
    )

    pareto_objectives = [[e["val_mse"], e["complexity"]] for e in evaluated]
    
    save_results(
        out_dir=out_dir,
        runtime=runtime,
        test_metrics=test_metrics,
        best_hyperparams={
            "hidden_dim": best_hp["hidden_dim"],
            "num_layers": best_hp["num_layers"],
            "lr": best_hp["lr"],
            "dropout": best_hp["dropout"],
            "complexity": int(best_val_solution["complexity"]),
        },
        pareto_objectives=pareto_objectives,
        seed=seed,
        mode=config.mode,
    )

    return (
        best_val_solution["val_mse"],
        test_metrics["rmse"],
        runtime
    )


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

    val_mse, test_rmse, runtime = run_nsga2(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'NSGA-II':<15}{val_mse:<15.6f}{test_rmse:<15.6f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
