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
from src.optimizers.pso import PSO
from src.training.fitness import pso_fitness
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
# PSO
# ==========================================================
def run_pso(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):

    set_seed(seed)
    print("\n================ PSO =================")
    start = time.time()

    b = config.hp_bounds
    bounds = [
        b["hidden_dim"],
        [b["num_layers"][0] - 0.5, b["num_layers"][1] + 0.5],  # ±0.5 offset: equal probability for each integer after round+clip
        [np.log10(b["lr"][0]), np.log10(b["lr"][1])],
        b["dropout"],
    ]

    pso_search_history = []

    def tracked_fitness(particle):
        score = pso_fitness(particle, train_df, val_df, device, mode=config.mode)
        b = config.hp_bounds
        pso_search_history.append({
            "eval":       len(pso_search_history) + 1,
            "hidden_dim": int(np.clip(np.round(particle[0]), b["hidden_dim"][0], b["hidden_dim"][1])),
            "num_layers": int(np.clip(np.round(particle[1]), b["num_layers"][0], b["num_layers"][1])),
            "lr":         float(np.clip(10 ** particle[2], b["lr"][0], b["lr"][1])),
            "dropout":    float(np.clip(particle[3], b["dropout"][0], b["dropout"][1])),
            "val_mse":    float(score[0]),
            "complexity": int(score[1]),
        })
        return score

    pso = PSO(
        fitness_fn=tracked_fitness,
        bounds=bounds,
        swarm_size=config.pso_swarm_size,
        iterations=config.pso_iterations,
        seed=seed
    )

    pareto_solutions, history = pso.optimize()

    evaluated = []
    for solution in pareto_solutions:
        hidden_dim, num_layers, lr, dropout = solution["params"]
        evaluated.append({
            "val_mse": float(solution["val_mse"]),
            "complexity": int(solution["complexity"]),
            "hyperparams": {
                "hidden_dim": int(np.clip(np.round(hidden_dim), b["hidden_dim"][0], b["hidden_dim"][1])),
                "num_layers": int(np.clip(np.round(num_layers), b["num_layers"][0], b["num_layers"][1])),
                "lr": float(np.clip(10 ** lr, b["lr"][0], b["lr"][1])),
                "dropout": float(np.clip(dropout, b["dropout"][0], b["dropout"][1])),
            },
        })

    # Common fair selection policy: choose minimum validation MSE from Pareto set.
    best_solution = min(evaluated, key=lambda x: x["val_mse"])
    best_hp = best_solution["hyperparams"]

    best_config = Config(mode=config.mode)
    best_config.hidden_dim = best_hp["hidden_dim"]
    best_config.num_layers = best_hp["num_layers"]
    best_config.lr = best_hp["lr"]
    best_config.dropout = best_hp["dropout"]
    best_config.checkpoint_path = f"checkpoints/seed_{seed}/{zone}/pso_best.pt"

    os.makedirs(os.path.dirname(best_config.checkpoint_path), exist_ok=True)

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    out_dir = f"results/seed_{seed}/{zone}/pso"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(pso_search_history).to_csv(
        os.path.join(out_dir, "search_history.csv"), index=False
    )
    pareto_rows = [
        {
            "hidden_dim": e["hyperparams"]["hidden_dim"],
            "num_layers": e["hyperparams"]["num_layers"],
            "lr": e["hyperparams"]["lr"],
            "dropout": e["hyperparams"]["dropout"],
            "val_mse": e["val_mse"],
            "complexity": e["complexity"],
        }
        for e in evaluated
    ]
    pd.DataFrame(pareto_rows).to_csv(
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
        convergence=history,
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

    val_mse, test_rmse, runtime = run_pso(
        train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test NRMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'PSO':<15}{val_mse:<15.6f}{test_rmse:<15.6f}{runtime:<15.2f}")


if __name__ == "__main__":
    main()
