import time
import json
import random
import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.training.training_pipeline import (
    train_single_configuration,
    retrain_and_evaluate
)
from src.optimizers.pso import PSO
from src.optimizers.moo import MOOOptimizer
from src.training.fitness import pso_fitness, moo_fitness
from src.config import Config


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
# Baseline
# ==========================================================
def run_baseline(train_df, val_df, test_df, scaling_params, device, config):

    print("\n================ BASELINE =================")
    start = time.time()

    base_config = Config(mode=config.mode)
    base_config.hidden_dim = 128
    base_config.lr = 0.001
    base_config.dropout = 0.0
    base_config.checkpoint_path = "checkpoints/baseline_best.pt"

    val_mse = train_single_configuration(
        train_df, val_df, device, base_config
    )

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, base_config, scaling_params
    )

    runtime = time.time() - start
    return val_mse, test_rmse, runtime


# ==========================================================
# Random Search
# ==========================================================
def run_random_search(train_df, val_df, test_df, scaling_params, device, config):

    print("\n================ RANDOM SEARCH =================")
    start = time.time()

    best_val = float("inf")
    best_config = None

    for trial in range(config.random_trials):

        print(f"\n########## Trial {trial+1}/{config.random_trials} ##########")

        trial_config = Config(mode=config.mode)
        trial_config.hidden_dim = random.choice([32, 64, 128])
        trial_config.lr = random.uniform(1e-4, 5e-3)
        trial_config.dropout = random.uniform(0.0, 0.3)
        trial_config.checkpoint_path = "checkpoints/random_trial.pt"

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

    print("\nRetraining Best Random Configuration...")

    best_config.checkpoint_path = "checkpoints/random_best.pt"

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start
    return best_val, test_rmse, runtime


# ==========================================================
# PSO
# ==========================================================
def run_pso(train_df, val_df, test_df, scaling_params, device, config):

    print("\n================ PSO =================")
    start = time.time()

    bounds = [
        [32, 256],
        [1e-4, 5e-3],
        [0.0, 0.3]
    ]

    pso = PSO(
        fitness_fn=lambda particle: pso_fitness(
            particle, train_df, val_df, device
        ),
        bounds=bounds,
        swarm_size=config.pso_swarm_size,
        iterations=config.pso_iterations,
        seed=42
    )

    best_position, best_val = pso.optimize()

    best_config = Config(mode=config.mode)
    best_config.hidden_dim = int(np.round(best_position[0]))
    best_config.lr = float(best_position[1])
    best_config.dropout = float(best_position[2])
    best_config.checkpoint_path = "checkpoints/pso_best.pt"

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df,
        device, best_config, scaling_params
    )

    runtime = time.time() - start
    return best_val, test_rmse, runtime


# ==========================================================
# MOO
# ==========================================================
def run_moo(train_df, val_df, test_df, scaling_params, device, config):

    print("\n================ MOO =================")
    start = time.time()

    bounds = [
        (32, 256),
        (1e-4, 5e-3),
        (0.0, 0.3),
    ]

    moo = MOOOptimizer(
        fitness_fn=lambda particle: moo_fitness(
            particle, train_df, val_df, device
        ),
        bounds=bounds,
        pop_size=config.moo_pop_size,
        generations=config.moo_generations,
        seed=42,
    )

    pareto_solutions = moo.optimize()
    evaluated = []

    for solution in pareto_solutions:

        hidden_dim, lr, dropout = solution["params"]

        sol_config = Config(mode=config.mode)
        sol_config.hidden_dim = int(np.round(hidden_dim))
        sol_config.lr = float(lr)
        sol_config.dropout = float(dropout)
        sol_config.checkpoint_path = f"checkpoints/moo_hidden{int(hidden_dim)}.pt"

        test_rmse = retrain_and_evaluate(
            train_df, val_df, test_df,
            device, sol_config, scaling_params
        )

        evaluated.append({
            "val_mse": solution["val_mse"],
            "test_rmse": test_rmse
        })

    runtime = time.time() - start

    best_val_solution = min(evaluated, key=lambda x: x["val_mse"])
    best_test_solution = min(evaluated, key=lambda x: x["test_rmse"])

    return (
        best_val_solution["val_mse"],
        best_test_solution["test_rmse"],
        runtime
    )


# ==========================================================
# Main
# ==========================================================
def main():

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(mode="dev")   # change to "full" later

    train_df, val_df, test_df, scaling_params = load_data(config)

    baseline = run_baseline(train_df, val_df, test_df, scaling_params, device, config)
    random_res = run_random_search(train_df, val_df, test_df, scaling_params, device, config)
    pso_res = run_pso(train_df, val_df, test_df, scaling_params, device, config)
    moo_res = run_moo(train_df, val_df, test_df, scaling_params, device, config)

    print("\n\n================ FINAL COMPARISON =================")
    print(f"{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)

    print(f"{'Baseline':<15}{baseline[0]:<15.6f}{baseline[1]:<15.4f}{baseline[2]:<15.2f}")
    print(f"{'Random':<15}{random_res[0]:<15.6f}{random_res[1]:<15.4f}{random_res[2]:<15.2f}")
    print(f"{'PSO':<15}{pso_res[0]:<15.6f}{pso_res[1]:<15.4f}{pso_res[2]:<15.2f}")
    print(f"{'MOO':<15}{moo_res[0]:<15.6f}{moo_res[1]:<15.4f}{moo_res[2]:<15.2f}")


if __name__ == "__main__":
    main()