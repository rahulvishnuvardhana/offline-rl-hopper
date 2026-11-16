"""State normalizer and reward scale, fit on one mixture's own transitions (ADR-5, ADR-9)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from offrl.data.load import Episode
from offrl.data.mixture import SOURCES, Picks

REWARD_SCALE_NUMERATOR = 1000.0
BETA_REF_SCALE = 0.2512  # reward scale of the p = 1 mixtures (Phase 2, A12); beta_eff reference


@dataclass(frozen=True)
class StateNormalizer:
    """``(x - mean) / (std + eps)``. ``std`` is stored without ``eps``; both float64."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-3

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) / (self.std + self.eps)).astype(np.float32)


def normalizer_from_arrays(arrays: Mapping[str, np.ndarray]) -> StateNormalizer:
    """The normalizer a mixture file carries. The buffer and the evaluator both use this one
    object, so training and evaluation observations cannot be transformed differently."""
    return StateNormalizer(
        np.asarray(arrays["state_mean"], dtype=np.float64),
        np.asarray(arrays["state_std"], dtype=np.float64),
        float(arrays["state_std_eps"]),
    )


def fit_state_normalizer(
    episodes: Mapping[str, Sequence[Episode]], picks: Picks, eps: float = 1e-3
) -> StateNormalizer:
    """Mean and population std (ddof = 0) of the mixture's ``obs`` rows, accumulated in float64
    over the float64 source rows, never over the float32 file."""
    obs = np.concatenate([episodes[s][i].obs[:n] for s in SOURCES for i, n in picks[s]])
    return StateNormalizer(mean=obs.mean(0), std=obs.std(0), eps=eps)


def reward_scale(episode_returns: np.ndarray, whole: np.ndarray) -> float:
    """IQL: ``1000 / (max_ret - min_ret)`` over untrimmed episodes only (ADR-9). A trimmed tail
    can be a few steps long and would otherwise set the minimum."""
    r = np.asarray(episode_returns, dtype=np.float64)[np.asarray(whole, dtype=bool)]
    if len(r) < 2 or r.max() == r.min():
        raise ValueError("need at least two untrimmed episodes with different returns")
    return REWARD_SCALE_NUMERATOR / float(r.max() - r.min())


def beta_eff(beta: float, scale: float) -> float:
    """IQL's effective temperature: advantages grow with the reward scale (ADR-9 caveat)."""
    return beta * scale / BETA_REF_SCALE
