"""Load Minari episodes into a plain, library-independent ``Episode`` type.

Everything downstream (validation, EDA, mixtures) depends on ``Episode`` only, so tests
can build synthetic episodes and CI never needs the real datasets.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Episode:
    """One episode. ``obs`` has one more row than ``act`` (initial + final observation)."""

    id: int
    obs: np.ndarray  # (N + 1, obs_dim)
    act: np.ndarray  # (N, act_dim)
    rew: np.ndarray  # (N,)
    term: np.ndarray  # (N,) bool, true environment termination (the hopper fell)
    trunc: np.ndarray  # (N,) bool, time-limit cut, NOT a terminal state

    @property
    def length(self) -> int:
        return len(self.act)

    @property
    def ret(self) -> float:
        return float(self.rew.sum())


def load_episodes(dataset_id: str, download: bool = False) -> list[Episode]:
    """Read every episode of a Minari dataset into memory (about 100 MB per Hopper dataset)."""
    import minari  # local import: tests and CI never touch Minari

    ds = minari.load_dataset(dataset_id, download=download)
    return [
        Episode(
            id=int(ep.id),
            obs=np.asarray(ep.observations, dtype=np.float64),
            act=np.asarray(ep.actions, dtype=np.float64),
            rew=np.asarray(ep.rewards, dtype=np.float64),
            term=np.asarray(ep.terminations, dtype=bool),
            trunc=np.asarray(ep.truncations, dtype=bool),
        )
        for ep in ds.iterate_episodes()
    ]


def dataset_metadata(dataset_id: str) -> dict[str, int]:
    """Counts the dataset reports about itself (reconciled against our own sums)."""
    import minari

    ds = minari.load_dataset(dataset_id)
    return {"total_steps": int(ds.total_steps), "total_episodes": int(ds.total_episodes)}


def dataset_dir(dataset_id: str) -> Path:
    root = os.environ.get("MINARI_DATASETS_PATH") or Path.home() / ".minari" / "datasets"
    return Path(root) / dataset_id


def file_hashes(dataset_id: str) -> dict[str, str]:
    """SHA-256 of every file in the dataset folder: proof of exactly which bytes we used."""
    base = dataset_dir(dataset_id)
    hashes = {}
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        hashes[path.relative_to(base).as_posix()] = digest.hexdigest()
    return hashes
