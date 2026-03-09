from src.data.dataset import LoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch import amp


# ==========================================================
# Hyperparameter Search Phase (with Early Stopping)
# ==========================================================
def train_single_configuration(train_data, val_data, device, config):

    train_dataset = LoadDataset(train_data, seq_len=config.seq_len)
    val_dataset   = LoadDataset(val_data,   seq_len=config.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.search_epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config.search_patience,
        min_delta=config.min_delta,
        save_path=config.checkpoint_path,
    )

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    epoch_bar = tqdm(
        range(config.search_epochs),
        desc="  Search",
        unit="ep",
        ncols=90,
        leave=True,
    )

    for _ in epoch_bar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
        )
        val_loss = validate(model, val_loader, criterion, device)

        epoch_bar.set_postfix(
            {"val": f"{val_loss:.5f}", "best": f"{early_stopper.best_loss:.5f}"}
        )

        early_stopper.step(val_loss, model)
        scheduler.step()

        if early_stopper.early_stop:
            epoch_bar.set_description("  Search [stopped]")
            break

    return early_stopper.best_loss


# ==========================================================
# Final Retraining Phase (NO Early Stopping)
# ==========================================================
def retrain_and_evaluate(train_data, val_data, test_data,
                         device, config, scaling_params):

    combined = pd.concat([train_data, val_data])

    dataset = LoadDataset(combined, seq_len=config.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # torch.compile speeds up the longer retrain run; skip on Windows where
    # the Triton backend is unavailable and compilation adds pure overhead.
    if sys.platform != "win32" and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.retrain_epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n[Final Retraining] epochs={config.retrain_epochs}")

    epoch_bar = tqdm(
        range(config.retrain_epochs),
        desc="  Retrain",
        unit="ep",
        ncols=90,
        leave=True,
    )

    best_train_loss = float("inf")

    for _ in epoch_bar:
        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, scaler,
        )
        scheduler.step()

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), config.checkpoint_path)

        epoch_bar.set_postfix(
            {"train": f"{train_loss:.5f}", "best": f"{best_train_loss:.5f}"}
        )

    # -------------------------
    # Test Evaluation
    # -------------------------
    test_dataset = LoadDataset(test_data, seq_len=config.seq_len)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()

    mean = scaling_params["mean"]
    std  = scaling_params["std"]

    all_norm_squared_errors = []
    all_squared_errors      = []
    all_abs_errors          = []
    all_target_values       = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            outputs = model(x).cpu().numpy()   # (batch,)
            targets = y.numpy()                # (batch,)

            # Normalized MSE — consistent with the val MSE search objective
            all_norm_squared_errors.append((outputs - targets) ** 2)

            # Inverse z-score to recover original MW scale
            pred_inv   = outputs * std + mean
            target_inv = targets * std + mean

            all_squared_errors.append((pred_inv - target_inv) ** 2)
            all_abs_errors.append(np.abs(pred_inv - target_inv))
            all_target_values.append(np.abs(target_inv))

    norm_sq = np.concatenate(all_norm_squared_errors)
    squared = np.concatenate(all_squared_errors)
    abs_err = np.concatenate(all_abs_errors)
    abs_tgt = np.concatenate(all_target_values)

    nrmse = float(np.sqrt(np.mean(norm_sq)))
    rmse  = float(np.sqrt(np.mean(squared)))
    mae   = float(np.mean(abs_err))
    # Floor denominator to avoid extreme percentages when targets are near zero
    mape  = float(np.mean(abs_err / np.clip(abs_tgt, 1.0, None)) * 100)

    print(
        f"[Final Test]  NRMSE: {nrmse:.6f}  "
        f"RMSE: {rmse:.4f} MW  MAE: {mae:.4f} MW  MAPE: {mape:.2f}%"
    )
    print(f"Model saved to: {config.checkpoint_path}")

    return {"nrmse": nrmse, "rmse": rmse, "mae": mae, "mape": mape}
