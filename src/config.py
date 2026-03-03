class Config:
    def __init__(self, mode="full"):

        # -----------------------
        # Mode
        # -----------------------
        self.mode = mode

        # -----------------------
        # Dev Controls
        # -----------------------
        self.dev_households = 10
        self.dev_timesteps = 2000   # <-- new (adjust later if needed)

        # -----------------------
        # Model
        # -----------------------
        self.hidden_dim = 64
        self.dropout = 0.2

        # -----------------------
        # Optimization
        # -----------------------
        self.lr = 0.001

        # -----------------------
        # Training
        # -----------------------
        self.batch_size = 512
        self.epochs = 30
        self.patience = 5
        self.min_delta = 1e-4
        self.checkpoint_path = "checkpoints/temp_best.pt"

        # -----------------------
        # Search Budgets
        # -----------------------
        self.random_trials = 2
        self.pso_swarm_size = 4
        self.pso_iterations = 2
        self.moo_pop_size = 6
        self.moo_generations = 2