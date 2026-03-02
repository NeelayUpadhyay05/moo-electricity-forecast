from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def train_single_configuration(train_df, val_df, device,
                               hidden_dim, lr, dropout):

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

    early_stopper = EarlyStopping(
        patience=5,
        min_delta=1e-4,
        save_path="dev_checkpoints/temp_best.pt"
    )

    max_epochs = 15
    best_val_loss = float("inf")

    for epoch in range(max_epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        early_stopper.step(val_loss, model)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    return best_val_loss


def retrain_and_evaluate(train_df, val_df, test_df, device,
                         hidden_dim, lr, dropout,
                         scaling_params):

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

    for _ in range(max_epochs):
        train_one_epoch(model, dataloader, optimizer, criterion, device)

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