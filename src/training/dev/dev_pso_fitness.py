import numpy as np
import torch
import pandas as pd
import json

from src.utils.seed import set_seed
from src.training.dev.dev_trainer import train_single_configuration


# ---- Load dev data ONCE at import time ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("data/processed/electricity_train.csv", index_col=0, parse_dates=True)
val_df = pd.read_csv("data/processed/electricity_val.csv", index_col=0, parse_dates=True)

# Development mode → first 10 households only
train_df = train_df.iloc[:, :10]
val_df = val_df.iloc[:, :10]


def pso_dev_fitness(particle):

    hidden_dim = int(np.round(particle[0]))
    lr = float(particle[1])
    dropout = float(particle[2])

    hidden_dim = max(32, min(256, hidden_dim))
    lr = max(1e-4, min(5e-3, lr))
    dropout = max(0.0, min(0.3, dropout))

    print("\n" + "="*50)
    print("Evaluating Particle")
    print(f"hidden_dim={hidden_dim} | lr={lr:.6f} | dropout={dropout:.3f}")
    print("-"*50)

    set_seed(42)

    best_val_mse = train_single_configuration(
        train_df,
        val_df,
        device,
        hidden_dim,
        lr,
        dropout
    )

    print(f"Best Validation MSE: {best_val_mse:.6f}")
    print("="*50)

    return best_val_mse