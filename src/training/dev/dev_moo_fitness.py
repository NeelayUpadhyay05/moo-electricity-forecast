import numpy as np

from src.training.dev.dev_trainer import train_single_configuration


def dev_moo_fitness(
    particle,
    train_df,
    val_df,
    device,
):
    """
    Multi-objective fitness function.

    Objectives:
        1. Minimize validation MSE
        2. Minimize model complexity (hidden_dim)

    Parameters
    ----------
    particle : array-like
        [hidden_dim, lr, dropout]

    Returns
    -------
    tuple
        (val_mse, hidden_dim)
    """

    hidden_dim, lr, dropout = particle

    # Ensure valid types
    hidden_dim = int(np.round(hidden_dim))

    # Safety clamp (important for mutation overflow)
    hidden_dim = max(32, min(256, hidden_dim))
    lr = max(1e-4, min(5e-3, lr))
    dropout = max(0.0, min(0.3, dropout))

    val_mse = train_single_configuration(
        train_df=train_df,
        val_df=val_df,
        device=device,
        hidden_dim=hidden_dim,
        lr=lr,
        dropout=dropout,
    )

    return (val_mse, hidden_dim)