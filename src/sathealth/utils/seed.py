"""Reproducibility: one call seeds every random number generator we use."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch; return a fresh NumPy Generator.

    Prefer passing the returned ``Generator`` explicitly over relying on global state.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by some third-party code
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        pass
    else:  # pragma: no cover - torch arrives in a later phase
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
