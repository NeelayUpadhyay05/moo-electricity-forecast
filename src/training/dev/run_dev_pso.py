import time
import numpy as np

from src.optimizers.pso import PSO
from src.training.dev.dev_pso_fitness import pso_dev_fitness
from src.utils.seed import set_seed

start_time = time.time()

def run_dev_pso():

    set_seed(42)

    # ---- Bounds: (dim, 2) ----
    bounds = [
        [32, 256],      # hidden_dim
        [1e-4, 5e-3],   # learning_rate
        [0.0, 0.3]      # dropout
    ]

    # ---- Initialize PSO ----
    pso = PSO(
        fitness_fn=pso_dev_fitness,
        bounds=bounds,
        swarm_size=6,      # small for dev
        iterations=4,      # small for dev
        seed=42
    )

    best_position, best_fitness = pso.optimize()

    print("\n===== DEV PSO RESULT =====")
    print("Best raw position:", best_position)
    print("Best validation MSE:", best_fitness)

    # ---- Decode best hyperparameters ----
    best_hidden = int(np.round(best_position[0]))
    best_lr = float(best_position[1])
    best_dropout = float(best_position[2])

    print("\nDecoded hyperparameters:")
    print("hidden_dim =", best_hidden)
    print("learning_rate =", best_lr)
    print("dropout =", best_dropout)

    total_time = time.time() - start_time
    print(f"\nTotal PSO runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    run_dev_pso()