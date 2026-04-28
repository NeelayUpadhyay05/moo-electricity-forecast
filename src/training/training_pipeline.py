from src.data.dataset import LoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping
from src.metrics import calculate_r2
from src.utils.seed import worker_init_fn

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch import amp


def _get_search_batch_size(config):
    return getattr(config, "search_batch_size", getattr(config, "batch_size", 128))


def _get_retrain_batch_size(config):
    return getattr(config, "retrain_batch_size", _get_search_batch_size(config))


def _get_search_lr(config):
    return getattr(config, "search_lr", getattr(config, "lr", 1e-3))


def _get_retrain_lr(config):
    explicit = getattr(config, "retrain_lr", None)
    if explicit is not None:
        return explicit

    search_batch_size = _get_search_batch_size(config)
    retrain_batch_size = _get_retrain_batch_size(config)
    return _get_search_lr(config) * (retrain_batch_size / search_batch_size)


# ==========================================================
# Hyperparameter Search Phase (with Early Stopping)
# ==========================================================
def train_single_configuration(train_data, val_data, device, config):
    base_seed = getattr(config, "seed", 42)

    train_dataset = LoadDataset(train_data, seq_len=config.seq_len)
    val_dataset   = LoadDataset(val_data,   seq_len=config.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=_get_search_batch_size(config),
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed) if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_get_search_batch_size(config),
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed) if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_get_search_lr(config))
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
    base_seed = getattr(config, "seed", 42)

    combined = pd.concat([train_data, val_data])

    dataset = LoadDataset(combined, seq_len=config.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=_get_retrain_batch_size(config),
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed) if config.num_workers > 0 else None,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=_get_retrain_lr(config))
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
        batch_size=_get_retrain_batch_size(config),
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed) if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()

    mean = scaling_params["mean"]
    std  = scaling_params["std"]

    all_predictions = []
    all_targets_normalized = []
    all_targets_original = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x).cpu().numpy()   # (batch,) normalized predictions
            targets = y.numpy()                # (batch,) normalized targets

            all_predictions.append(outputs)
            all_targets_normalized.append(targets)
            
            # Inverse z-score to recover original MW scale
            target_inv = targets * std + mean
            all_targets_original.append(target_inv)

    pred_norm = np.concatenate(all_predictions)
    targ_norm = np.concatenate(all_targets_normalized)
    targ_orig = np.concatenate(all_targets_original)
    
    # Predictions in original scale
    pred_orig = pred_norm * std + mean

    # Calculate metrics
    rmse = float(np.sqrt(np.mean((pred_orig - targ_orig) ** 2)))
    mae = float(np.mean(np.abs(pred_orig - targ_orig)))
    mape = float(np.mean(np.abs(pred_orig - targ_orig) / np.clip(np.abs(targ_orig), 1.0, None)) * 100)
    r2 = calculate_r2(pred_norm, targ_norm)

    print(
        f"[Final Test]  RMSE: {rmse:.4f} MW  MAE: {mae:.4f} MW  MAPE: {mape:.2f}%  R2: {r2:.4f}"
    )
    print(f"Model saved to: {config.checkpoint_path}")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
