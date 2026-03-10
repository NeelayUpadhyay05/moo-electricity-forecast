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
from src.training.training_pipeline import train_single_configuration, retrain_and_evaluate
from src.optimizers.moo import MOOOptimizer
from src.training.fitness import moo_fitness
from src.config import Config


# ==========================================================
# High-Budget MOO Settings
# pop_size=15, generations=9 → 15 × (1 + 9) = 150 total evaluations
# Produces a Pareto front of up to 15 solutions vs. 6 in the main comparison
# ==========================================================
MOO_POP_SIZE    = 15
MOO_GENERATIONS = 9


# ==========================================================
# Data Loading
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
# High-Budget MOO Pareto Run
# ==========================================================
def run_moo_pareto(train_df, val_df, test_df, scaling_params, device, config,
                   seed=42, zone="PJME"):

    set_seed(seed)
    total_evals = MOO_POP_SIZE * (1 + MOO_GENERATIONS)
    print(f"\n======== MOO PARETO ANALYSIS ========")
    print(f"pop_size={MOO_POP_SIZE}, generations={MOO_GENERATIONS}, "
          f"total_evals={total_evals}")

    start = time.time()

    b = config.hp_bounds
    bounds = [
        tuple(b["hidden_dim"]),
        (b["num_layers"][0] - 0.5, b["num_layers"][1] + 0.5),
        (np.log10(b["lr"][0]), np.log10(b["lr"][1])),
        tuple(b["dropout"]),
    ]

    moo = MOOOptimizer(
        fitness_fn=lambda particle: moo_fitness(
            particle, train_df, val_df, device, mode=config.mode
        ),
        bounds=bounds,
        pop_size=MOO_POP_SIZE,
        generations=MOO_GENERATIONS,
        seed=seed,
    )

    pareto_solutions, history = moo.optimize()

    # Decode raw params into readable hyperparameters
    evaluated = []
    for solution in pareto_solutions:
        hidden_dim, num_layers, lr, dropout = solution["params"]
        evaluated.append({
            "val_mse":    float(solution["val_mse"]),
            "complexity": float(solution["complexity"]),
            "hyperparams": {
                "hidden_dim": int(np.clip(np.round(hidden_dim),
                                          b["hidden_dim"][0], b["hidden_dim"][1])),
                "num_layers": int(np.clip(np.round(num_layers),
                                          b["num_layers"][0], b["num_layers"][1])),
                "lr":         float(np.clip(10 ** lr, b["lr"][0], b["lr"][1])),
                "dropout":    float(np.clip(dropout, b["dropout"][0], b["dropout"][1])),
            },
        })

    # Sort by complexity (ascending) for readable Pareto front output
    evaluated.sort(key=lambda x: x["complexity"])

    # Retrain the best-accuracy solution on the Pareto front
    best_val_solution = min(evaluated, key=lambda x: x["val_mse"])
    best_hp = best_val_solution["hyperparams"]

    best_config = Config(mode=config.mode)
    best_config.hidden_dim = best_hp["hidden_dim"]
    best_config.num_layers = best_hp["num_layers"]
    best_config.lr         = best_hp["lr"]
    best_config.dropout    = best_hp["dropout"]
    best_config.checkpoint_path = (
        f"checkpoints/pareto_analysis/{zone}/seed_{seed}/moo_pareto_best.pt"
    )
    os.makedirs(os.path.dirname(best_config.checkpoint_path), exist_ok=True)

    test_metrics = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    out_dir = f"results/pareto_analysis/{zone}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Pareto front CSV — sorted by complexity
    pareto_rows = [
        {
            "hidden_dim": e["hyperparams"]["hidden_dim"],
            "num_layers": e["hyperparams"]["num_layers"],
            "lr":         e["hyperparams"]["lr"],
            "dropout":    e["hyperparams"]["dropout"],
            "val_mse":    e["val_mse"],
            "complexity": e["complexity"],
        }
        for e in evaluated
    ]
    pd.DataFrame(pareto_rows).to_csv(
        os.path.join(out_dir, "pareto_front.csv"), index=False
    )

    # Metrics JSON
    result = {
        "seed":               seed,
        "zone":               zone,
        "mode":               config.mode,
        "moo_pop_size":       MOO_POP_SIZE,
        "moo_generations":    MOO_GENERATIONS,
        "total_evaluations":  total_evals,
        "n_pareto_solutions": len(evaluated),
        "runtime_s":          round(runtime, 2),
        "best_val_mse":       float(best_val_solution["val_mse"]),
        "best_test_nrmse":    float(test_metrics["nrmse"]),
        "best_test_rmse":     float(test_metrics["rmse"]),
        "best_test_mae":      float(test_metrics["mae"]),
        "best_test_mape":     float(test_metrics["mape"]),
        "best_hyperparams":   best_hp,
        "convergence":        [float(v) for v in history],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nPareto front: {len(evaluated)} solutions (sorted by complexity)")
    print(f"{'Params':>10} {'Val MSE':>10} {'hidden_dim':>11} {'layers':>7}")
    print("-" * 44)
    for e in evaluated:
        print(f"{int(e['complexity']):>10,} {e['val_mse']:>10.6f} "
              f"{e['hyperparams']['hidden_dim']:>11} {e['hyperparams']['num_layers']:>7}")

    print(f"\nBest val_mse : {best_val_solution['val_mse']:.6f}")
    print(f"Test NRMSE   : {test_metrics['nrmse']:.6f}")
    print(f"Results saved to {out_dir}/")

    return best_val_solution["val_mse"], test_metrics["nrmse"], runtime


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

    val_mse, test_nrmse, runtime = run_moo_pareto(
        train_df, val_df, test_df, scaling_params, device, config,
        seed=args.seed, zone=args.zone
    )

    print(f"\n{'Method':<25}{'Val MSE':<15}{'Test NRMSE':<15}{'Time (s)'}")
    print("-" * 65)
    print(f"{'MOO Pareto (150 evals)':<25}{val_mse:<15.6f}{test_nrmse:<15.6f}{runtime:.2f}")


if __name__ == "__main__":
    main()
