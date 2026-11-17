"""Shared by test_iql.py and test_corl.py (on the path via pytest's ``pythonpath``)."""

import torch

from offrl.algos.iql import IQLTrainer
from offrl.data.buffer import Batch
from offrl.seeding import seed_everything

OBS, ACT, B = 11, 3, 64
RTOL, ATOL = 1e-9, 1e-12


def config(beta=3.0, steps=1000, lr=3e-4, discount=0.99, hidden=(32, 32), target_update_rate=0.005):
    return {
        "net": {"hidden": list(hidden), "log_std_min": -20.0, "log_std_max": 2.0},
        "train": {"lr": lr, "steps": steps, "actor_cosine_lr": True},
        "algo": {
            "name": "iql",
            "expectile": 0.7,
            "beta": beta,
            "discount": discount,
            "target_update_rate": target_update_rate,
            "exp_adv_max": 100.0,
        },
    }


def make_batch(gen, n=B, rew_scale=50.0):
    """Done takes both values; large rewards give large Q targets after a few steps."""
    done = torch.zeros(n, 1)
    done[: n // 4] = 1.0
    return Batch(
        obs=torch.randn(n, OBS, generator=gen),
        act=torch.rand(n, ACT, generator=gen) * 2 - 1,
        rew=(torch.rand(n, 1, generator=gen) * 2 - 1) * rew_scale,
        next_obs=torch.randn(n, OBS, generator=gen),
        done=done,
    )


def make_trainer(seed=0, **kw):
    seed_everything(seed)
    return IQLTrainer(config(**kw), OBS, ACT, torch.device("cpu"))


def spread_targets(t: IQLTrainer, scale=200.0):
    """Scale the target twin's output layers so adv takes both signs and some weights clip."""
    with torch.no_grad():
        for head in (t.q_target.q1, t.q_target.q2):
            head[-1].weight.mul_(scale)


def branch_coverage(t: IQLTrainer, batch: Batch) -> None:
    with torch.no_grad():
        adv = torch.min(*t.q_target(batch.obs, batch.act)) - t.v(batch.obs)
    assert (adv > 0).any() and (adv < 0).any()
    assert (torch.exp(t.beta * adv) > t.exp_adv_max).any()
    assert set(batch.done.flatten().tolist()) == {0.0, 1.0}


def two_mode_batch(gen, n=256):
    """One-step episodes: action dim 0 is +0.8 (reward 1) or -0.8 (reward 0), equally often."""
    good = torch.rand(n, 1, generator=gen) < 0.5
    act = torch.zeros(n, ACT)
    act[:, :1] = torch.where(good, 0.8, -0.8)
    return Batch(
        obs=torch.randn(n, OBS, generator=gen),
        act=act,
        rew=good.float(),
        next_obs=torch.randn(n, OBS, generator=gen),
        done=torch.ones(n, 1),
    )
