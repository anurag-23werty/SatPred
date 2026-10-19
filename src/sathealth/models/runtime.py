"""Device selection and deterministic seeding for PyTorch.

Every deep model is trained under :func:`seed_torch`, so the same config, data and seed give
bit-identical weights and scores on CPU. GPU kernels (``mps``, ``cuda``) are faster but not
guaranteed to be reproducible, which is why ``cpu`` is the default device.
"""

import torch

from sathealth.logging import get_logger

log = get_logger(__name__)


def resolve_device(name: str) -> torch.device:
    """``auto`` -> cuda, then mps, then cpu. An explicit but unavailable device is an error."""
    if name == "auto":
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - Apple silicon only
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(name)


def seed_torch(seed: int, threads: int | None = None) -> torch.Generator:
    """Seed torch, enable deterministic kernels and return a CPU generator for shuffling."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if threads is not None:
        torch.set_num_threads(threads)
    return torch.Generator().manual_seed(seed)
