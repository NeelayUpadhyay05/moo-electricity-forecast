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

    # For extra determinism (optional but recommended)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"