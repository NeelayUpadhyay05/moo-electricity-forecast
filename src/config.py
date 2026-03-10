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
        if mode == "full":
            self.lr             = 0.004   # linear LR scaling: 0.001 × (2048/512)
            self.batch_size     = 2048
            self.search_epochs  = 20      # epochs per fitness evaluation during HPO
            self.search_patience = 5      # early stopping patience during HPO
            self.retrain_epochs = 60      # epochs for final retrain after HPO
            self.num_workers    = 0 if sys.platform == "win32" else 4
        else:
            self.lr             = 0.001
            self.batch_size     = 512
            self.search_epochs  = 10
            self.search_patience = 3
            self.retrain_epochs = 15
            self.num_workers    = 0

        self.min_delta = 1e-4
        self.checkpoint_path = "checkpoints/temp_best.pt"

        # -----------------------
        # DataLoader
        # -----------------------
        self.pin_memory = True                            # no-op on CPU; speeds up CPU→GPU on CUDA
        self.drop_last = True                             # avoid noisy final batch during training
        self.persistent_workers = (self.num_workers > 0) # keep workers alive between epochs

        # -----------------------
        # Search Budgets (mode-specific, equal across all three methods)
        # Full: 30 evaluations each — random=30, PSO=6×(1+4)=30, MOO=6×(1+4)=30
        # Dev:  12 evaluations each — random=12, PSO=4×(1+2)=12, MOO=4×(1+2)=12
        # -----------------------
        if mode == "full":
            self.n_trials        = 30
            self.pso_swarm_size  = 6
            self.pso_iterations  = 4   # 6 × (1 + 4) = 30 total evals
            self.moo_pop_size    = 6
            self.moo_generations = 4   # 6 × (1 + 4) = 30 total evals
        else:
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
