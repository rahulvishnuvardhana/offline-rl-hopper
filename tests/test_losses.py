"""B1 to B4 (spec Section 7): the shared policy loss and the BC trainer. IQL checks join in Phase 5."""

import math

import numpy as np
import pytest
import torch

from offrl.algos.bc import BCTrainer
from offrl.algos.losses import policy_loss
from offrl.data.buffer import Batch
from offrl.nets import GaussianPolicy

CFG = {
    "net": {"hidden": [32, 32], "log_std_min": -20.0, "log_std_max": 2.0},
    "train": {"lr": 3e-4, "steps": 2000, "actor_cosine_lr": True},
}


def closed_form(policy: GaussianPolicy, obs, act, weights) -> float:
    """Per-sample float64 loop, written without torch.distributions (independent of the code)."""
    mean = torch.tanh(policy.net(obs)).detach().double().numpy()
    log_std = policy.effective_log_std().detach().double().numpy()
    a, w = act.double().numpy(), np.asarray(weights, dtype=np.float64).reshape(-1)
    total = 0.0
    for i in range(len(a)):
        nll = 0.0
        for d in range(a.shape[1]):
            z = (a[i, d] - mean[i, d]) / math.exp(log_std[d])
            nll += 0.5 * z * z + log_std[d] + 0.5 * math.log(2 * math.pi)
        total += w[i] * nll
    return total / len(a)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(16, 11), torch.rand(16, 3) * 2 - 1


def test_b1_matches_closed_form_with_random_weights(batch):
    torch.manual_seed(1)
    policy = GaussianPolicy(11, 3, (32, 32))
    with torch.no_grad():
        policy.log_std.copy_(torch.tensor([-0.5, 0.3, 0.1]))
    obs, act = batch
    weights = torch.rand(16, 1) * 5  # non-constant: a (B, B) broadcast would change the value
    got = policy_loss(policy, obs, act, weights).item()
    assert got == pytest.approx(closed_form(policy, obs, act, weights), abs=1e-5)
    assert policy_loss(policy, obs, act, weights[:, 0]).item() == got  # (B,) and (B, 1) agree


def test_b1_shape_mismatch_raises(batch):
    policy = GaussianPolicy(11, 3, (32, 32))
    obs, act = batch
    with pytest.raises(ValueError):
        policy_loss(policy, obs, act, torch.ones(16, 16))
    with pytest.raises(ValueError):
        policy_loss(policy, obs, act, torch.ones(8))


def test_b1_weights_are_constants(batch):
    policy = GaussianPolicy(11, 3, (32, 32))
    obs, act = batch
    weights = torch.rand(16, requires_grad=True)
    policy_loss(policy, obs, act, weights).backward()
    assert weights.grad is None
    loss1 = policy_loss(policy, obs, act, torch.ones(16)).item()
    assert policy_loss(policy, obs, act, torch.full((16,), 3.0)).item() == pytest.approx(3 * loss1, rel=1e-6)


def test_b2_bc_update_is_unit_weight_policy_loss(batch):
    torch.manual_seed(2)
    trainer = BCTrainer(CFG, 11, 3, torch.device("cpu"))
    obs, act = batch
    expected = policy_loss(trainer.policy, obs, act, torch.ones(16)).item()
    zeros = torch.zeros(16, 1)
    out = trainer.update(Batch(obs, act, zeros, obs, zeros))
    assert out["policy_loss"].item() == expected
    assert set(out) == set(BCTrainer.LOG_KEYS)


def test_b3_cosine_schedule_reaches_zero():
    cfg = {**CFG, "train": {**CFG["train"], "steps": 10}}
    trainer = BCTrainer(cfg, 11, 3, torch.device("cpu"))
    zeros = torch.zeros(4, 1)
    lrs = [
        trainer.update(Batch(torch.randn(4, 11), torch.zeros(4, 3), zeros, zeros, zeros))["actor_lr"].item()
        for _ in range(10)
    ]
    assert lrs[0] == pytest.approx(3e-4) and lrs == sorted(lrs, reverse=True)
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_b4_overfit_beats_best_constant_gaussian(make_mixture):
    arrays = make_mixture(n=256, seed=3)
    obs, act = torch.as_tensor(arrays["obs"]), torch.as_tensor(arrays["act"])
    # Best constant Gaussian: per-dimension mean and std of the actions (its NLL in closed form).
    a = arrays["act"].astype(np.float64)
    const_nll = float(np.sum(0.5 * np.log(2 * np.pi * a.var(0)) + 0.5))
    torch.manual_seed(4)
    cfg = {**CFG, "train": {"lr": 3e-3, "steps": 2000, "actor_cosine_lr": False}}
    trainer = BCTrainer(cfg, 11, 3, torch.device("cpu"))
    zeros = torch.zeros(256, 1)
    for _ in range(2000):
        loss = trainer.update(Batch(obs, act, zeros, obs, zeros))["policy_loss"].item()
    assert loss < const_nll - 1.0
