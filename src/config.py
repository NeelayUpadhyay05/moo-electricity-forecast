import sys


class Config:
    def __init__(self, mode="dev"):

        # -----------------------
        # Mode
        # -----------------------
        self.mode = mode

        # -----------------------
        # Dev Controls
        # -----------------------
        self.dev_timesteps = 2000

        # -----------------------
        # Model
        # -----------------------
        self.seq_len    = 24
        self.hidden_dim = 64
        self.num_layers = 1
        self.dropout = 0.2

        # -----------------------
        # Training (mode-specific)
        # -----------------------
        self.base_batch_size = 128
        self.base_lr = 0.001

        if mode == "full":
            self.search_batch_size = 512
            self.retrain_batch_size = 128
            self.lr             = self.base_lr * (self.search_batch_size / self.base_batch_size)
            self.batch_size     = self.search_batch_size
            self.search_epochs  = 30      # epochs per fitness evaluation during HPO
            self.search_patience = 10      # early stopping patience during HPO
            self.retrain_epochs = 60      # epochs for final retrain after HPO
            self.num_workers    = 2  # with worker_init_fn for reproducibility
        else:
            self.search_batch_size = 128
            self.retrain_batch_size = 128
            self.lr             = self.base_lr
            self.batch_size     = self.search_batch_size
            self.search_epochs  = 10
            self.search_patience = 3
            self.retrain_epochs = 15
            self.num_workers    = 0

        self.search_lr = self.lr

        self.min_delta = 1e-4
        self.checkpoint_path = "checkpoints/temp_best.pt"

        # -----------------------
        # DataLoader
        # -----------------------
        self.pin_memory = True                            # no-op on CPU; speeds up CPU→GPU on CUDA
        self.drop_last = True                             # avoid noisy final batch during training
        self.persistent_workers = (self.num_workers > 0) # keep workers alive between epochs

        # -----------------------
        # Search Budgets
        # Full mode follows the requested research setting:
        # budget=200, pop/swarm=10, generations/iterations=20.
        # Random/Optuna are aligned to the same budget label (200 trials).
        # -----------------------
        if mode == "full":
            self.fair_budget_evals = 200
            self.n_trials        = self.fair_budget_evals
            self.pso_swarm_size  = 10
            self.pso_iterations  = 20
            self.moo_pop_size    = 10
            self.moo_generations = 19  # 10 initial + (10 * 19 offspring) = 200 total evals
        else:
            # Dev mode remains lightweight for iteration speed.
            self.fair_budget_evals = 12
            self.n_trials        = 12
            self.pso_swarm_size  = 4
            self.pso_iterations  = 2   # 4 × (1 + 2) = 12 total evals
            self.moo_pop_size    = 4
            self.moo_generations = 2   # 4 × (1 + 2) = 12 total evals

        # -----------------------
        # Hyperparameter Bounds (single source of truth)
        # -----------------------
        self.hp_bounds = {
            "hidden_dim": [32, 256],
            "num_layers": [1, 3],
            "lr":         [1e-4, 5e-3],
            "dropout":    [0.0, 0.3],
        }

        # -----------------------
        # Model registry
        # Order defines how experiments are enumerated. Only Musk Ox and
        # NSGA-II are multi-objective — all others are single-objective.
        # -----------------------
        self.model_list = [
            "baseline_lstm",
            "musk_ox_multi_lstm",
            "random_search_lstm",
            "optuna_lstm",
            "nsga2_direct",
            "arima",
            "lightgbm",
            "cnn_lstm",
        ]

        # Set of model keys treated as multi-objective optimizations.
        self.multi_objective_methods = {"musk_ox_multi_lstm", "nsga2_direct"}
