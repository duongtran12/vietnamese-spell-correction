"""Reproducibility helpers for experiments."""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when they are available."""

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

