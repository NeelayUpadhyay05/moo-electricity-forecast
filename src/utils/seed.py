import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Surface remaining non-deterministic ops; warn_only avoids hard failures
    # for ops that lack deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Extra determinism for cuBLAS (optional but recommended).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """
    Initialize a worker process with a deterministic seed.

    Use this as the DataLoader worker_init_fn when num_workers > 0.
    Each worker gets a unique seed derived from the base seed.

    Args:
        worker_id: assigned by PyTorch DataLoader
        base_seed: the main random seed (typically from set_seed)
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)