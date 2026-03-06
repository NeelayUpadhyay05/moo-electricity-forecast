import multiprocessing
import argparse
import torch

from src.utils.seed import set_seed
from src.config import Config
from experiments.run_baseline import load_data, run_baseline
from experiments.run_random_search import run_random_search
from experiments.run_pso import run_pso
from experiments.run_moo import run_moo

if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for num_workers > 0 on Windows

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = Config(mode="full")
    train_df, val_df, test_df, scaling_params = load_data(config)

    baseline    = run_baseline(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed)
    random_res  = run_random_search(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed)
    pso_res     = run_pso(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed)
    moo_res     = run_moo(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed)

    print("\n\n================ FINAL COMPARISON =================")
    print(f"{'Method':<15}{'Val MSE':<15}{'Test RMSE':<15}{'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Baseline':<15}{baseline[0]:<15.6f}{baseline[1]:<15.4f}{baseline[2]:<15.2f}")
    print(f"{'Random':<15}{random_res[0]:<15.6f}{random_res[1]:<15.4f}{random_res[2]:<15.2f}")
    print(f"{'PSO':<15}{pso_res[0]:<15.6f}{pso_res[1]:<15.4f}{pso_res[2]:<15.2f}")
    print(f"{'MOO':<15}{moo_res[0]:<15.6f}{moo_res[1]:<15.4f}{moo_res[2]:<15.2f}")
