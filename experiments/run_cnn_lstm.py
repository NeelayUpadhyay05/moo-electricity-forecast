import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_lstm import CNNLSTM
from src.config import Config
from src.utils.seed import set_seed


def metrics(preds, targets, scaling_params):
    mean = scaling_params["mean"]
    std = scaling_params["std"]
    preds_n = (preds - mean) / std
    targets_n = (targets - mean) / std
    nrmse = float(np.sqrt(np.mean((preds_n - targets_n) ** 2)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    mape = float(np.mean(np.abs(preds - targets) / np.clip(np.abs(targets), 1.0, None)) * 100)
    return {"nrmse": nrmse, "rmse": rmse, "mae": mae, "mape": mape}


def make_dataset(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def load_data(config, zone="PJME"):
    base = f"data/processed/{zone}"
    train_df = pd.read_csv(f"{base}_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{base}_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(f"{base}_test.csv", index_col=0, parse_dates=True)
    with open(f"{base}_scaling.json") as f:
        scaling_params = json.load(f)
    if config.mode == "dev":
        train_df = train_df.iloc[:config.dev_timesteps]
        val_df = val_df.iloc[:config.dev_timesteps]
        test_df = test_df.iloc[:config.dev_timesteps]
    return train_df, val_df, test_df, scaling_params


def run_cnn_lstm(train_df, val_df, test_df, scaling_params, device, config, seed=42, zone="PJME"):
    set_seed(seed)
    print("\n================ CNN-LSTM =================")
    start = time.time()

    seq_len = config.seq_len
    train_series = train_df.iloc[:, 0].values
    val_series = val_df.iloc[:, 0].values
    test_series = test_df.iloc[:, 0].values

    X_train, y_train = make_dataset(np.concatenate([train_series, val_series]), seq_len)
    X_test, y_test = make_dataset(np.concatenate([train_series, val_series, test_series]), seq_len)
    # test portion at end
    start_idx = len(train_series) + len(val_series) - seq_len
    X_test = X_test[start_idx: start_idx + len(test_series)]
    y_test = y_test[start_idx: start_idx + len(test_series)]

    train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_tensor, batch_size=config.batch_size, shuffle=True)

    model = CNNLSTM(seq_len=seq_len, conv_channels=16, lstm_hidden=config.hidden_dim, lstm_layers=config.num_layers, dropout=config.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epochs = config.retrain_epochs if config.mode == "full" else 5
    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if ep % max(1, epochs//5) == 0:
            print(f"Epoch {ep+1}/{epochs} loss={np.mean(epoch_losses):.6f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds = model(X_test_t).cpu().numpy()

    test_metrics = metrics(preds, y_test, scaling_params)

    runtime = time.time() - start
    out_dir = f"results/seed_{seed}/{zone}/cnn_lstm"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"runtime": runtime, **test_metrics}, f, indent=4)
    print(f"Results saved to {out_dir}/metrics.json")

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["dev", "full"])
    parser.add_argument("--zone", type=str, default="PJME")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(mode=args.mode)
    train_df, val_df, test_df, scaling_params = load_data(config, zone=args.zone)

    run_cnn_lstm(train_df, val_df, test_df, scaling_params, device, config, seed=args.seed, zone=args.zone)


if __name__ == "__main__":
    main()
