import multiprocessing
import argparse
import json
import os
import torch

from src.utils.seed import set_seed
from src.config import Config
from experiments.run_baseline import load_data, run_baseline
from experiments.run_random_search import run_random_search
from experiments.run_moo import run_moo
from experiments.run_optuna import run_optuna

if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for num_workers > 0 on Windows

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["dev", "full"])
    parser.add_argument("--zone", type=str, default="PJME")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(mode=args.mode)
    train_df, val_df, test_df, scaling_params = load_data(config, zone=args.zone)

    baseline    = run_baseline(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone)
    random_res  = run_random_search(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone)
    optuna_res  = run_optuna(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone)
    moo_res     = run_moo(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone)

    print("\n\n================ FINAL COMPARISON =================")
    print(f"\n{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Baseline':<15}{baseline[0]:<15.6f}{baseline[1]:<15.6f}{baseline[2]:<15.2f}")
    print(f"{'Random':<15}{random_res[0]:<15.6f}{random_res[1]:<15.6f}{random_res[2]:<15.2f}")
    print(f"{'Optuna (TPE)':<15}{optuna_res[0]:<15.6f}{optuna_res[1]:<15.6f}{optuna_res[2]:<15.2f}")
    print(f"{'MOO':<15}{moo_res[0]:<15.6f}{moo_res[1]:<15.6f}{moo_res[2]:<15.2f}")

    comparison = {
        "seed": args.seed,
        "zone": args.zone,
        "mode": args.mode,
        "results": {
            "baseline":   {"val_mse": baseline[0],   "test_rmse": baseline[1]},
            "random":     {"val_mse": random_res[0],  "test_rmse": random_res[1]},
            "optuna_tpe": {"val_mse": optuna_res[0],  "test_rmse": optuna_res[1]},
            "moo":        {"val_mse": moo_res[0],     "test_rmse": moo_res[1]},
        },
    }
    out_dir = f"results/seed_{args.seed}/{args.zone}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"\nComparison saved to {out_dir}/comparison.json")
