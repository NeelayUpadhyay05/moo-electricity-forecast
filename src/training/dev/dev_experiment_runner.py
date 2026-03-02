import time
import json
import random
import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.training.dev.dev_trainer import (
    train_single_configuration,
    retrain_and_evaluate
    )
from src.optimizers.pso import PSO
from src.training.dev.dev_pso_fitness import pso_dev_fitness
from src.optimizers.moo import MOOOptimizer
from src.training.dev.dev_moo_fitness import dev_moo_fitness

def load_dev_data():
    train_df = pd.read_csv("data/processed/electricity_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/electricity_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/processed/electricity_test.csv", index_col=0, parse_dates=True)

    with open("data/processed/electricity_scaling.json") as f:
        scaling_params = json.load(f)

    # Dev mode: first 10 households
    train_df = train_df.iloc[:, :10]
    val_df = val_df.iloc[:, :10]
    test_df = test_df.iloc[:, :10]

    return train_df, val_df, test_df, scaling_params


def run_baseline(train_df, val_df, test_df, scaling_params, device):
    print("\n===== BASELINE LSTM =====")
    start = time.time()

    hidden_dim = 128
    lr = 0.001
    dropout = 0.0

    val_mse = train_single_configuration(
        train_df, val_df, device,
        hidden_dim, lr, dropout
    )

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df, device,
        hidden_dim, lr, dropout,
        scaling_params
    )

    runtime = time.time() - start

    return val_mse, test_rmse, runtime


def run_random_search(train_df, val_df, test_df, scaling_params, device):
    print("\n===== RANDOM SEARCH =====")
    start = time.time()

    n_trials = 3
    best_val = float("inf")
    best_config = None

    for _ in range(n_trials):
        hidden_dim = random.choice([32, 64, 128])
        lr = random.uniform(1e-4, 5e-3)
        dropout = random.uniform(0.0, 0.3)

        val_mse = train_single_configuration(
            train_df, val_df, device,
            hidden_dim, lr, dropout
        )

        if val_mse < best_val:
            best_val = val_mse
            best_config = (hidden_dim, lr, dropout)

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df, device,
        best_config[0], best_config[1], best_config[2],
        scaling_params
    )

    runtime = time.time() - start

    return best_val, test_rmse, runtime


def run_pso(train_df, val_df, test_df, scaling_params, device):
    print("\n===== PSO =====")
    start = time.time()

    bounds = [
        [32, 256],
        [1e-4, 5e-3],
        [0.0, 0.3]
    ]

    pso = PSO(
        fitness_fn=pso_dev_fitness,
        bounds=bounds,
        swarm_size=6,
        iterations=4,
        seed=42
    )

    best_position, best_val = pso.optimize()

    hidden_dim = int(np.round(best_position[0]))
    lr = float(best_position[1])
    dropout = float(best_position[2])

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df, device,
        hidden_dim, lr, dropout,
        scaling_params
    )

    runtime = time.time() - start

    return best_val, test_rmse, runtime


def run_moo(train_df, val_df, test_df, scaling_params, device):
    print("\n===== MOO (Pareto-based) =====")
    start = time.time()

    bounds = [
        (32, 256),
        (1e-4, 5e-3),
        (0.0, 0.3),
    ]

    moo = MOOOptimizer(
        fitness_fn=lambda particle: dev_moo_fitness(
            particle,
            train_df,
            val_df,
            device,
        ),
        bounds=bounds,
        pop_size=10,
        generations=3,
        seed=42,
    )

    pareto_solutions = moo.optimize()

    evaluated = []

    for solution in pareto_solutions:
        hidden_dim, lr, dropout = solution["params"]

        hidden_dim = int(np.round(hidden_dim))
        lr = float(lr)
        dropout = float(dropout)

        test_rmse = retrain_and_evaluate(
            train_df, val_df, test_df, device,
            hidden_dim, lr, dropout,
            scaling_params
        )

        evaluated.append({
            "hidden_dim": hidden_dim,
            "val_mse": solution["val_mse"],
            "test_rmse": test_rmse
        })

    runtime = time.time() - start

    # ---- Selection for final comparison ----
    best_val_solution = min(evaluated, key=lambda x: x["val_mse"])
    best_test_solution = min(evaluated, key=lambda x: x["test_rmse"])

    print("\nPareto Solutions:")
    for sol in evaluated:
        print(sol)

    print("\nBest Validation MSE (Pareto):", best_val_solution)
    print("Best Test RMSE (Pareto):", best_test_solution)

    return (
        best_val_solution["val_mse"],
        best_test_solution["test_rmse"],
        runtime
    )


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, scaling_params = load_dev_data()

    baseline = run_baseline(train_df, val_df, test_df, scaling_params, device)
    random_res = run_random_search(train_df, val_df, test_df, scaling_params, device)
    pso_res = run_pso(train_df, val_df, test_df, scaling_params, device)
    moo_res = run_moo(train_df, val_df, test_df, scaling_params, device)

    print("\n\n================ FINAL COMPARISON (DEV MODE) ================")
    print(f"{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)

    print(f"{'Baseline':<15}{baseline[0]:<15.6f}{baseline[1]:<15.4f}{baseline[2]:<15.2f}")
    print(f"{'Random':<15}{random_res[0]:<15.6f}{random_res[1]:<15.4f}{random_res[2]:<15.2f}")
    print(f"{'PSO':<15}{pso_res[0]:<15.6f}{pso_res[1]:<15.4f}{pso_res[2]:<15.2f}")
    print(f"{'MOO':<15}{moo_res[0]:<15.6f}{moo_res[1]:<15.4f}{moo_res[2]:<15.2f}")


if __name__ == "__main__":
    main()