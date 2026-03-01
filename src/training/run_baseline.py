from src.utils.seed import set_seed
from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate, compute_rmse
from src.training.early_stopping import EarlyStopping

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np


def evaluate_model(model, dataloader, device, scaling_params, household_columns):
    model.eval()

    all_squared_errors = []

    with torch.no_grad():
        for x, y, household_idx in dataloader:
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

    # Load processed data
    train_df = pd.read_csv("data/processed/electricity_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/electricity_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/processed/electricity_test.csv", index_col=0, parse_dates=True)

    with open("data/processed/electricity_scaling.json") as f:
        scaling_params = json.load(f)

    household_columns = train_df.columns.tolist()

    train_dataset = GlobalLoadDataset(train_df)
    val_dataset = GlobalLoadDataset(val_df)
    test_dataset = GlobalLoadDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = LSTMModel(hidden_dim=128).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stopper = EarlyStopping(patience=10, min_delta=1e-4, save_path="best_model.pt")

    max_epochs = 50

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        early_stopper.step(val_loss, model)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))

    val_rmse = evaluate_model(model, val_loader, device, scaling_params, household_columns)
    test_rmse = evaluate_model(model, test_loader, device, scaling_params, household_columns)

    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    os.makedirs("results", exist_ok=True)

    with open("results/baseline_metrics.json", "w") as f:
        json.dump({
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse)
        }, f, indent=4)


if __name__ == "__main__":
    main()