"""IQL as CORL's ``ImplicitQLearning`` (Phase 5 spec Section 2), on the Phase 4 actor.

Per step: ``next_v = V(s')`` before V moves; V by expectile regression on the target twin's
min; Q towards ``r + gamma (1 - done) next_v``, then the soft target update; the actor by the
shared policy loss weighted with ``exp(beta * adv)``, ``adv`` from the V step (pre-update V).
Official IQL feeds the updated V instead; CORL is the reference this phase validates against.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from offrl.algos.bc import make_actor_optimizer, make_policy
from offrl.algos.losses import advantage_weights, expectile_loss, policy_loss, q_target
from offrl.data.buffer import Batch
from offrl.nets import TwinQ, ValueNet


class IQLTrainer:
    LOG_KEYS = (
        "v_loss",
        "q_loss",
        "policy_loss",
        "v_mean",
        "q_mean",
        "q_target_mean",
        "adv_mean",
        "adv_std",
        "weight_mean",
        "frac_clipped",
        "log_std_mean",
        "actor_lr",
    )

    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int, device: torch.device) -> None:
        # The policy is built first after seeding, so BC and IQL with one seed start identically.
        self.policy = make_policy(cfg, obs_dim, act_dim, device)
        self.opt, self.sched = make_actor_optimizer(self.policy, cfg)
        hidden = cfg["net"]["hidden"]
        self.q = TwinQ(obs_dim, act_dim, hidden).to(device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.v = ValueNet(obs_dim, hidden).to(device)
        lr = cfg["train"]["lr"]
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)
        a = cfg["algo"]
        self.expectile, self.beta, self.discount = a["expectile"], a["beta"], a["discount"]
        self.target_update_rate, self.exp_adv_max = a["target_update_rate"], a["exp_adv_max"]

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        obs, act = batch.obs, batch.act
        with torch.no_grad():
            next_v = self.v(batch.next_obs)
            target_q = torch.min(*self.q_target(obs, act))

        v = self.v(obs)
        adv = target_q - v
        v_loss = expectile_loss(adv, self.expectile)
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        y = q_target(batch.rew, batch.done, next_v, self.discount)
        q1, q2 = self.q(obs, act)
        q_loss = (F.mse_loss(q1, y) + F.mse_loss(q2, y)) / 2
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        self._soft_update()

        weights = advantage_weights(adv, self.beta, self.exp_adv_max)
        loss = policy_loss(self.policy, obs, act, weights)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        lr = torch.full((), self.opt.param_groups[0]["lr"], device=loss.device)
        if self.sched is not None:
            self.sched.step()

        adv = adv.detach()
        clipped = torch.exp(self.beta * adv) > self.exp_adv_max
        return {
            "v_loss": v_loss.detach(),
            "q_loss": q_loss.detach(),
            "policy_loss": loss.detach(),
            "v_mean": v.detach().mean(),
            "q_mean": torch.min(q1, q2).detach().mean(),
            "q_target_mean": target_q.mean(),
            "adv_mean": adv.mean(),
            "adv_std": adv.std(),
            "weight_mean": weights.mean(),
            "frac_clipped": clipped.to(adv.dtype).mean(),
            "log_std_mean": self.policy.effective_log_std().detach().mean(),
            "actor_lr": lr,
        }

    @torch.no_grad()
    def _soft_update(self) -> None:
        """``Q_targ <- (1 - rate) Q_targ + rate Q``, CORL's ``soft_update`` arithmetic."""
        rate = self.target_update_rate
        for t, s in zip(self.q_target.parameters(), self.q.parameters(), strict=True):
            t.copy_((1 - rate) * t + rate * s)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.act(obs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "v": self.v.state_dict(),
            "opt": self.opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "v_opt": self.v_opt.state_dict(),
            "sched": None if self.sched is None else self.sched.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        self.v.load_state_dict(sd["v"])
        self.opt.load_state_dict(sd["opt"])
        self.q_opt.load_state_dict(sd["q_opt"])
        self.v_opt.load_state_dict(sd["v_opt"])
        if self.sched is not None:
            self.sched.load_state_dict(sd["sched"])
