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


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """
    Initialize worker process with a deterministic seed.
    
    This function is used as the worker_init_fn for PyTorch DataLoaders
    to ensure reproducibility when num_workers > 0. Each worker gets
    a unique seed derived from the base seed.
    
    Args:
        worker_id: assigned by PyTorch DataLoader
        base_seed: the main random seed (typically from set_seed)
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)