"""Deterministic seeding for Python, NumPy and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG used by the project.

    With ``deterministic=True`` cuDNN autotuning is disabled and deterministic
    algorithms are requested. Operations without a deterministic CUDA kernel
    only emit a warning (``warn_only``), so bitwise reproducibility across
    different GPU models or library versions is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Make DataLoader workers reproducible (torch seeds them from the loader generator)."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
