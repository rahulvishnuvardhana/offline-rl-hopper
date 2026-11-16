"""BC = IQL's policy loss with every weight 1 (``beta = 0``), same policy, optimizer and LR schedule."""

from __future__ import annotations

from typing import Any

import torch

from offrl.algos.losses import policy_loss
from offrl.data.buffer import Batch
from offrl.nets import GaussianPolicy


def make_policy(cfg: dict[str, Any], obs_dim: int, act_dim: int, device: torch.device) -> GaussianPolicy:
    net = cfg["net"]
    return GaussianPolicy(obs_dim, act_dim, net["hidden"], net["log_std_min"], net["log_std_max"]).to(device)


def make_actor_optimizer(
    policy: GaussianPolicy, cfg: dict[str, Any]
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.LRScheduler | None]:
    """Adam, plus cosine decay to 0 over ``train.steps`` when ``train.actor_cosine_lr`` (both algos)."""
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["train"]["lr"])
    sched = None
    if cfg["train"]["actor_cosine_lr"]:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["steps"])
    return opt, sched


class BCTrainer:
    LOG_KEYS = ("policy_loss", "log_std_mean", "actor_lr")

    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int, device: torch.device) -> None:
        # The policy is built first after seeding, so BC and IQL with one seed start identically.
        self.policy = make_policy(cfg, obs_dim, act_dim, device)
        self.opt, self.sched = make_actor_optimizer(self.policy, cfg)

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        weights = torch.ones(batch.obs.shape[0], device=batch.obs.device)
        loss = policy_loss(self.policy, batch.obs, batch.act, weights)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        lr = torch.tensor(self.opt.param_groups[0]["lr"])
        if self.sched is not None:
            self.sched.step()
        return {
            "policy_loss": loss.detach(),
            "log_std_mean": self.policy.effective_log_std().detach().mean(),
            "actor_lr": lr,
        }

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.act(obs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": None if self.sched is None else self.sched.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.opt.load_state_dict(sd["opt"])
        if self.sched is not None:
            self.sched.load_state_dict(sd["sched"])
