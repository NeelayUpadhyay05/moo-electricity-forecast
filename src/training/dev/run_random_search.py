# DEVELOPMENT MODE - DO NOT COMMIT FULL DATASET YET

from src.utils.seed import set_seed
from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate, compute_rmse
from src.training.early_stopping import EarlyStopping

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import json
import numpy as np
import os

os.makedirs("dev_checkpoints", exist_ok=True)


def train_single_configuration(train_df, val_df, device, hidden_dim, lr, dropout):
    train_dataset = GlobalLoadDataset(train_df)
    val_dataset = GlobalLoadDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = LSTMModel(
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=5, min_delta=1e-4, save_path="dev_checkpoints/temp_best.pt")

    max_epochs = 15
    best_val_loss = float("inf")

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        early_stopper.step(val_loss, model)

        if early_stopper.early_stop:
            break

    return best_val_loss


def retrain_and_evaluate(train_df, val_df, test_df, device, hidden_dim, lr, dropout, scaling_params):

    combined_df = pd.concat([train_df, val_df], axis=0)

    dataset = GlobalLoadDataset(combined_df)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = LSTMModel(
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_epochs = 15

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

    # No checkpoint loading — use final weights directly

    test_dataset = GlobalLoadDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_squared_errors = []

    household_columns = train_df.columns.tolist()

    with torch.no_grad():
        for x, y, household_idx in test_loader:
            x = x.to(device)
            outputs = model(x).cpu().numpy()
            targets = y.numpy()

            for i in range(outputs.shape[0]):
                col = household_columns[household_idx[i]]
                min_val = scaling_params[col]["min"]
                max_val = scaling_params[col]["max"]

                pred_inv = outputs[i] * (max_val - min_val) + min_val
                target_inv = targets[i] * (max_val - min_val) + min_val

                all_squared_errors.extend((pred_inv - target_inv) ** 2)

    rmse = np.sqrt(np.mean(all_squared_errors))
    return float(rmse)


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


if __name__ == "__main__":
    main()