import numpy as np
import uuid
import os
from src.config import Config
from src.models.lstm import LSTMModel, count_parameters
from src.training.training_pipeline import train_single_configuration


def decode_particle_to_config(particle, mode):

    config = Config(mode=mode)
    b = config.hp_bounds

    config.hidden_dim = int(np.clip(np.round(particle[0]), b["hidden_dim"][0], b["hidden_dim"][1]))
    config.num_layers = int(np.clip(np.round(particle[1]), b["num_layers"][0], b["num_layers"][1]))
    config.lr = float(np.clip(10 ** particle[2], b["lr"][0], b["lr"][1]))
    config.dropout = float(np.clip(particle[3], b["dropout"][0], b["dropout"][1]))

    # Ensure each HPO evaluation writes to a unique checkpoint to avoid clobbering
    fname = f"hpo_{uuid.uuid4().hex}.pt"
    config.checkpoint_path = os.path.join("checkpoints", "hpo", fname)
    # Propagate experiment-level seed (if set by runners) so DataLoader workers
    # use a consistent base seed during HPO evaluations.
    try:
        config.seed = int(os.environ.get("EXPERIMENT_SEED", 42))
    except Exception:
        config.seed = 42

    return config


def evaluate_multi_objective(particle, train_df, val_df, device, mode, log_prefix="Candidate"):

    config = decode_particle_to_config(particle, mode)

    print("\n" + "=" * 50)
    print(f"Evaluating {log_prefix}")
    print(
        f"hidden_dim={config.hidden_dim} | "
        f"num_layers={config.num_layers} | "
        f"lr={config.lr:.6f} | "
        f"dropout={config.dropout:.3f}"
    )
    print("-" * 50)

    val_mse = train_single_configuration(
        train_df,
        val_df,
        device,
        config
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    n_params = count_parameters(model)

    print(f"Validation MSE: {val_mse:.6f} | Complexity (params): {n_params}")
    print("=" * 50)

    return float(val_mse), int(n_params)


# --------------------------------------------------
# PSO Fitness (Multi Objective)
# --------------------------------------------------

def pso_fitness(particle, train_df, val_df, device, mode):
    return evaluate_multi_objective(
        particle,
        train_df,
        val_df,
        device,
        mode,
        log_prefix="PSO Particle",
    )


# --------------------------------------------------
# MOO Fitness (Multi Objective)
# --------------------------------------------------

def moo_fitness(particle, train_df, val_df, device, mode):
    return evaluate_multi_objective(
        particle,
        train_df,
        val_df,
        device,
        mode,
        log_prefix="MOO Candidate",
    )
