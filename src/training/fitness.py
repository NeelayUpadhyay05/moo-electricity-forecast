import numpy as np
from src.config import Config
from src.models.lstm import LSTMModel, count_parameters
from src.training.training_pipeline import train_single_configuration


# --------------------------------------------------
# PSO Fitness (Single Objective)
# --------------------------------------------------

def pso_fitness(particle, train_df, val_df, device, mode):

    config = Config(mode=mode)
    b = config.hp_bounds

    config.hidden_dim  = int(np.clip(np.round(particle[0]), b["hidden_dim"][0], b["hidden_dim"][1]))
    config.num_layers  = int(np.clip(np.round(particle[1]), b["num_layers"][0], b["num_layers"][1]))
    config.lr          = float(np.clip(10 ** particle[2], b["lr"][0], b["lr"][1]))
    config.dropout     = float(np.clip(particle[3], b["dropout"][0], b["dropout"][1]))

    print("\n" + "="*50)
    print("Evaluating Particle")
    print(
        f"hidden_dim={config.hidden_dim} | "
        f"num_layers={config.num_layers} | "
        f"lr={config.lr:.6f} | "
        f"dropout={config.dropout:.3f}"
    )
    print("-"*50)

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

def moo_fitness(particle, train_df, val_df, device, mode):

    config = Config(mode=mode)
    b = config.hp_bounds

    config.hidden_dim  = int(np.clip(np.round(particle[0]), b["hidden_dim"][0], b["hidden_dim"][1]))
    config.num_layers  = int(np.clip(np.round(particle[1]), b["num_layers"][0], b["num_layers"][1]))
    config.lr          = float(np.clip(10 ** particle[2], b["lr"][0], b["lr"][1]))
    config.dropout     = float(np.clip(particle[3], b["dropout"][0], b["dropout"][1]))

    print("\n--- Evaluating New MOO Candidate ---")

    val_mse = train_single_configuration(
        train_df,
        val_df,
        device,
        config
    )

    # Objective 1: Validation MSE
    # Objective 2: Model complexity (parameter count)
    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    n_params = count_parameters(model)

    return (val_mse, n_params)
