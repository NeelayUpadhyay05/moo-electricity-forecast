import numpy as np
from src.config import Config
from src.utils.seed import set_seed
from src.training.training_pipeline import train_single_configuration


# --------------------------------------------------
# PSO Fitness (Single Objective)
# --------------------------------------------------

def pso_fitness(particle, train_df, val_df, device):

    config = Config()

    config.hidden_dim = int(np.round(particle[0]))
    config.lr = float(particle[1])
    config.dropout = float(particle[2])

    # Safety clamp
    config.hidden_dim = max(32, min(256, config.hidden_dim))
    config.lr = max(1e-4, min(5e-3, config.lr))
    config.dropout = max(0.0, min(0.3, config.dropout))

    print("\n" + "="*50)
    print("Evaluating Particle")
    print(
        f"hidden_dim={config.hidden_dim} | "
        f"lr={config.lr:.6f} | "
        f"dropout={config.dropout:.3f}"
    )
    print("-"*50)

    set_seed(42)

    best_val_mse = train_single_configuration(
        train_df,
        val_df,
        device,
        config
    )

    print(f"Best Validation MSE: {best_val_mse:.6f}")
    print("="*50)

    return best_val_mse


# --------------------------------------------------
# MOO Fitness (Multi Objective)
# --------------------------------------------------

def moo_fitness(particle, train_df, val_df, device):

    config = Config()

    config.hidden_dim = int(np.round(particle[0]))
    config.lr = float(particle[1])
    config.dropout = float(particle[2])

    # Safety clamp
    config.hidden_dim = max(32, min(256, config.hidden_dim))
    config.lr = max(1e-4, min(5e-3, config.lr))
    config.dropout = max(0.0, min(0.3, config.dropout))

    print("\n--- Evaluating New MOO Candidate ---")

    val_mse = train_single_configuration(
        train_df,
        val_df,
        device,
        config
    )

    # Objective 1: Validation MSE
    # Objective 2: Model complexity (hidden_dim)
    return (val_mse, config.hidden_dim)