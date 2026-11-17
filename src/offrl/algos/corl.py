"""CORL's ``ImplicitQLearning`` behind the ``Trainer`` protocol (Phase 5 spec Section 5).

Runs through ``train.run``, so CORL gets our pre-flight, data, evaluator and final score. The
vendored code is not edited; anything it hard-codes is checked against the config here.
"""

from __future__ import annotations

from typing import Any

import torch

from offrl._vendor import corl_iql as corl
from offrl.data.buffer import Batch


class CorlTrainer:
    LOG_KEYS = ("v_loss", "q_loss", "policy_loss", "log_std_mean", "actor_lr")

    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int, device: torch.device) -> None:
        net, tr, a = cfg["net"], cfg["train"], cfg["algo"]
        hidden = net["hidden"]
        fixed = {
            "net.hidden all one width": len(set(hidden)) == 1,
            "net.log_std_min = LOG_STD_MIN": net["log_std_min"] == corl.LOG_STD_MIN,
            "net.log_std_max = LOG_STD_MAX": net["log_std_max"] == corl.LOG_STD_MAX,
            "algo.exp_adv_max = EXP_ADV_MAX": a["exp_adv_max"] == corl.EXP_ADV_MAX,
            "train.actor_cosine_lr (CORL always decays)": tr["actor_cosine_lr"] is True,
        }
        if not all(fixed.values()):
            raise ValueError(f"CORL hard-codes: {[k for k, ok in fixed.items() if not ok]}")
        width, depth, lr = hidden[0], len(hidden), tr["lr"]
        # CORL's construction order (its train()): Q, V, then the actor.
        q = corl.TwinQ(obs_dim, act_dim, width, depth).to(device)
        v = corl.ValueFunction(obs_dim, width, depth).to(device)
        actor = corl.GaussianPolicy(obs_dim, act_dim, 1.0, width, depth).to(device)
        self.iql = corl.ImplicitQLearning(
            max_action=1.0,
            actor=actor,
            actor_optimizer=torch.optim.Adam(actor.parameters(), lr=lr),
            q_network=q,
            q_optimizer=torch.optim.Adam(q.parameters(), lr=lr),
            v_network=v,
            v_optimizer=torch.optim.Adam(v.parameters(), lr=lr),
            iql_tau=a["expectile"],
            beta=a["beta"],
            max_steps=tr["steps"],
            discount=a["discount"],
            tau=a["target_update_rate"],
            device=device,
        )
        self.device = device

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        """CORL returns Python floats (a GPU sync per step), so CORL timings are not reported."""
        lr = self.iql.actor_optimizer.param_groups[0]["lr"]
        log = self.iql.train([batch.obs, batch.act, batch.rew, batch.next_obs, batch.done])
        values = {
            "v_loss": log["value_loss"],
            "q_loss": log["q_loss"],
            "policy_loss": log["actor_loss"],
            "log_std_mean": self.iql.actor.log_std.clamp(corl.LOG_STD_MIN, corl.LOG_STD_MAX).mean().item(),
            "actor_lr": lr,
        }
        return {k: torch.tensor(v, device=self.device) for k, v in values.items()}

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """The tanh mean (``max_action`` is 1, and CORL's ``act`` takes NumPy one state at a time)."""
        return self.iql.actor(obs).mean

    def state_dict(self) -> dict[str, Any]:
        # CORL's own state_dict omits the target network; its load resets it to a copy of Q.
        return {**self.iql.state_dict(), "q_target": self.iql.q_target.state_dict()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.iql.load_state_dict(sd)
        self.iql.q_target.load_state_dict(sd["q_target"])
        self.iql.q_target.requires_grad_(False)
