from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os


def train_single_configuration(train_df, val_df, device, config):

    train_dataset = GlobalLoadDataset(train_df)
    val_dataset = GlobalLoadDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    hidden_dim = config.hidden_dim
    lr = config.lr
    dropout = config.dropout

    model = LSTMModel(
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    early_stopper = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        save_path=config.checkpoint_path
    )

    max_epochs = config.epochs
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
                         config, scaling_params):

    # --------------------------------------------------
    # Combine Train + Validation
    # --------------------------------------------------
    combined_df = pd.concat([train_df, val_df], axis=0)

    dataset = GlobalLoadDataset(combined_df)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # --------------------------------------------------
    # Build Model
    # --------------------------------------------------
    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    print(f"\n[Final Retraining] epochs={config.epochs}")

    # --------------------------------------------------
    # Fixed-Length Training (NO Early Stopping)
    # --------------------------------------------------
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss:.6f}")

    # Save FINAL trained model
    torch.save(model.state_dict(), config.checkpoint_path)

    # --------------------------------------------------
    # Test Evaluation
    # --------------------------------------------------
    test_dataset = GlobalLoadDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    model.eval()
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

    print(f"[Final Test RMSE]: {rmse:.4f}")
    print(f"Model saved to: {config.checkpoint_path}")

    return float(rmse)