from src.config import Config
from src.utils.seed import set_seed
from src.training.training_pipeline import (
    train_single_configuration,
    retrain_and_evaluate
)

import pandas as pd
import torch
import json
import os
import time


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load processed data
    train_df = pd.read_csv("data/processed/electricity_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/electricity_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/processed/electricity_test.csv", index_col=0, parse_dates=True)

    with open("data/processed/electricity_scaling.json") as f:
        scaling_params = json.load(f)

    # Baseline configuration
    config = Config()
    config.hidden_dim = 128
    config.lr = 0.001
    config.dropout = 0.0
    config.epochs = 50
    config.patience = 10
    config.batch_size = 256
    config.checkpoint_path = "checkpoints/baseline_best.pt"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n===== Running Baseline LSTM =====")
    start = time.time()

    val_mse = train_single_configuration(train_df, val_df, device, config)

    test_rmse = retrain_and_evaluate(
        train_df,
        val_df,
        test_df,
        device,
        config,
        scaling_params
    )

    runtime = time.time() - start

    print(f"\nValidation MSE: {val_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Runtime: {runtime:.2f} sec")

    with open("results/baseline_metrics.json", "w") as f:
        json.dump({
            "val_mse": float(val_mse),
            "test_rmse": float(test_rmse),
            "runtime_sec": float(runtime)
        }, f, indent=4)


if __name__ == "__main__":
    main()