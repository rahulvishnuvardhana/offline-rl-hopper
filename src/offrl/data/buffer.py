"""Replay buffer: the whole mixture on the training device, sampled uniformly (ROADMAP 3.4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from offrl.data.mixture import load_mixture
from offrl.data.normalize import StateNormalizer


@dataclass(frozen=True)
class Batch:
    obs: torch.Tensor  # (B, 11) float32, normalized
    act: torch.Tensor  # (B, 3)  float32, in [-1, 1]
    rew: torch.Tensor  # (B, 1)  float32, scaled for IQL, raw for BC
    next_obs: torch.Tensor  # (B, 11) float32, normalized
    done: torch.Tensor  # (B, 1)  float32, terminations ONLY


class ReplayBuffer:
    """Normalizes and scales once, then keeps the five model arrays on ``device``.

    Build it with ``from_mixture``; the plain constructor exists for tests.

    Indices are drawn with a CPU generator and moved to the device, so a seed gives the same
    batch sequence on the laptop and on a cluster GPU.
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        normalizer: StateNormalizer,
        reward_scale: float | None,
        device: str | torch.device,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        rew = np.asarray(arrays["rew"], dtype=np.float64)
        if reward_scale is not None:
            rew = rew * reward_scale

        def to_device(x: np.ndarray, column: bool = False) -> torch.Tensor:
            t = torch.as_tensor(np.asarray(x, dtype=np.float32))
            return (t.unsqueeze(1) if column else t).to(self.device)

        self.obs = to_device(normalizer.transform(arrays["obs"]))
        self.act = to_device(arrays["act"])
        self.rew = to_device(rew, column=True)
        self.next_obs = to_device(normalizer.transform(arrays["next_obs"]))
        self.done = to_device(arrays["done"], column=True)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    @classmethod
    def from_mixture(
        cls,
        mixture: str | Path | Mapping[str, np.ndarray],
        scale_rewards: bool,
        device: str | torch.device,
        seed: int,
    ) -> ReplayBuffer:
        """The only constructor training code should use: the normalizer, eps and reward scale
        come from the mixture itself, so another mixture's statistics cannot leak in (ADR-5)."""
        arrays = load_mixture(mixture) if isinstance(mixture, str | Path) else mixture
        norm = StateNormalizer(arrays["state_mean"], arrays["state_std"], float(arrays["state_std_eps"]))
        scale = float(arrays["reward_scale"]) if scale_rewards else None
        return cls(arrays, norm, scale, device, seed)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(len(self), (batch_size,), generator=self._generator).to(self.device)
        return Batch(self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])
