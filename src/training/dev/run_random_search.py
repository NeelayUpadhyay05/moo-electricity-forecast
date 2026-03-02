# DEVELOPMENT MODE - DO NOT COMMIT FULL DATASET YET

from src.utils.seed import set_seed
from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate, compute_rmse
from src.training.early_stopping import EarlyStopping
from src.training.dev.dev_trainer import (
    train_single_configuration,
    retrain_and_evaluate
)

import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import json
import numpy as np
import os

start_time = time.time()
os.makedirs("dev_checkpoints", exist_ok=True)

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df = pd.read_csv("data/processed/electricity_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/electricity_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/processed/electricity_test.csv", index_col=0, parse_dates=True)

    with open("data/processed/electricity_scaling.json") as f:
        scaling_params = json.load(f)

    # --- DEVELOPMENT MODE ---
    train_df = train_df.iloc[:, :10]
    val_df = val_df.iloc[:, :10]
    test_df = test_df.iloc[:, :10]

    n_trials = 3
    results = []

    for trial in range(n_trials):

        hidden_dim = random.choice([32, 64, 128])
        lr = random.uniform(1e-4, 5e-3)
        dropout = random.uniform(0.0, 0.3)

        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"hidden_dim={hidden_dim}, lr={lr:.6f}, dropout={dropout:.3f}")

        val_loss = train_single_configuration(
            train_df, val_df, device,
            hidden_dim, lr, dropout
        )

        print(f"Validation MSE: {val_loss:.6f}")

        results.append({
            "hidden_dim": hidden_dim,
            "lr": lr,
            "dropout": dropout,
            "val_loss": float(val_loss)
        })

    best_config = min(results, key=lambda x: x["val_loss"])

    print("\nBest Configuration Found:")
    print(best_config)

    test_rmse = retrain_and_evaluate(
        train_df, val_df, test_df, device,
        best_config["hidden_dim"],
        best_config["lr"],
        best_config["dropout"],
        scaling_params
    )

    print(f"\nFinal Test RMSE after retraining: {test_rmse:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal Random Search runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()