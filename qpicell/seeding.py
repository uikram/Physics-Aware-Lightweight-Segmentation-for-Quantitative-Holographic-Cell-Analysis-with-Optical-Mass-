"""Seeding of Python, NumPy and PyTorch (and of DataLoader workers)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch
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
    import torch
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
