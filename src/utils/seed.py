import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Surface any remaining non-deterministic ops; warn_only avoids crashes
    # on ops that have no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # For extra determinism (optional but recommended)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"