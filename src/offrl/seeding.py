"""Seed every source of randomness we control.

Environments are seeded separately at ``env.reset(seed=...)`` in the evaluator.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def derive_seeds(seed: int, n: int = 3) -> list[int]:
    """Independent child seeds of a run seed: network init, buffer indices, diagnostic sampling.

    Seeding the buffer generator with the run seed itself would replay torch's global stream.
    """
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    """Seed Python, NumPy and torch (CPU + CUDA); return a NumPy Generator for local use.

    ``deterministic=True`` also forces deterministic cuDNN/cuBLAS kernels. It is slower,
    so use it for reproducibility tests, not for the grid.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (legacy global state, for third-party code)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)
