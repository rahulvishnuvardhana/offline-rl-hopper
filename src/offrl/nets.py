"""Networks shared by BC and IQL (ROADMAP 3.2). PyTorch default init, as CORL (spec Section 3)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


class MLP(nn.Sequential):
    """Linear-ReLU blocks, then a final Linear (no output activation)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        super().__init__(*layers)


class GaussianPolicy(nn.Module):
    """CORL's IQL policy: ``mean = tanh(MLP(obs))``, state-independent ``log_std``.

    The clamp is applied in ``forward``; the raw parameter is not clamped. Samples are not
    squashed, so dataset actions at the bound |a| = 1 keep a finite log-prob.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def effective_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self.log_std_min, self.log_std_max)

    def forward(self, obs: torch.Tensor) -> Normal:
        return Normal(torch.tanh(self.net(obs)), self.effective_log_std().exp())

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Log-density summed over action dimensions, shape (B,)."""
        return self(obs).log_prob(act).sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: the mean, in [-1, 1]."""
        return torch.tanh(self.net(obs))


class TwinQ(nn.Module):
    """Two independent Q heads on ``(obs, act)``, each returning (B, 1). Used from Phase 5."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNet(nn.Module):
    """State value V(s), shape (B, 1). Used from Phase 5."""

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)
