"""I1 to I7, I9 and I11 (Phase 5 spec Section 7): IQL's update pinned down on synthetic data.

Exactness tests run in float64 on the CPU with one thread, so a 1e-12 tolerance means the
arithmetic, not float32 rounding, is being compared.
"""

import copy
import math
import time

import pytest
import torch

from iql_helpers import (
    ACT,
    ATOL,
    OBS,
    RTOL,
    branch_coverage,
    config,
    make_batch,
    make_trainer,
    spread_targets,
    two_mode_batch,
)
from offrl.algos import iql as iql_mod
from offrl.algos.bc import BCTrainer
from offrl.algos.iql import IQLTrainer
from offrl.algos.losses import advantage_weights, expectile_loss, q_target
from offrl.data.buffer import Batch
from offrl.seeding import seed_everything

# ---------- I1, I3, I4: the helpers ----------


def test_i1_expectile_loss():
    one = torch.ones(1, 1)
    assert expectile_loss(one, 0.7).item() == pytest.approx(0.7, rel=1e-6)  # float32 mask, as CORL
    assert expectile_loss(-one, 0.7).item() == pytest.approx(0.3, rel=1e-6)
    assert expectile_loss(torch.zeros(1, 1), 0.7).item() == 0.0
    with pytest.raises(ValueError, match=r"expected \(B, 1\)"):
        expectile_loss(torch.ones(4), 0.7)


def test_i3_weights_are_capped_and_finite():
    adv = torch.tensor([[1e4], [0.0], [-1e4], [1.0]])
    w = advantage_weights(adv, 3.0, 100.0)
    assert w.shape == (4,) and torch.isfinite(w).all()
    assert w.tolist() == pytest.approx([100.0, 1.0, 0.0, math.exp(3.0)])
    with pytest.raises(ValueError):
        advantage_weights(torch.ones(4), 3.0, 100.0)


def test_i3_frac_clipped_counts_the_clipped_entries(f64):
    t = make_trainer()
    spread_targets(t)
    batch = make_batch(torch.Generator().manual_seed(1))
    with torch.no_grad():
        adv = torch.min(*t.q_target(batch.obs, batch.act)) - t.v(batch.obs)
    expected = (torch.exp(3.0 * adv) > 100.0).double().mean().item()
    assert 0 < expected < 1
    assert t.update(batch)["frac_clipped"].item() == expected


def test_i4_q_target_terminal_and_bootstrap():
    rew, next_v = torch.tensor([[2.0], [2.0]]), torch.tensor([[10.0], [10.0]])
    y = q_target(rew, torch.tensor([[1.0], [0.0]]), next_v, 0.9)
    assert y.flatten().tolist() == pytest.approx([2.0, 11.0])
    with pytest.raises(ValueError):
        q_target(rew.flatten(), torch.zeros(2, 1), next_v, 0.9)


def test_i4_next_v_is_read_before_the_v_step(f64, monkeypatch):
    t = make_trainer()
    batch = make_batch(torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = copy.deepcopy(t.v)(batch.next_obs)
    seen = {}
    real = iql_mod.q_target

    def spy(rew, done, next_v, gamma):
        seen["next_v"] = next_v.clone()
        return real(rew, done, next_v, gamma)

    monkeypatch.setattr(iql_mod, "q_target", spy)
    t.update(batch)
    assert torch.equal(seen["next_v"], expected)
    assert not torch.equal(t.v(batch.next_obs), expected)  # V did move in this step


# ---------- I5, I6: which network feeds what ----------


def test_i5_v_reads_the_target_twin_and_gradients_stay_separate(f64):
    t = make_trainer()
    with torch.no_grad():  # online and target Q differ, so reading the wrong one shows
        for p in t.q.parameters():
            p.add_(0.1)
    batch = make_batch(torch.Generator().manual_seed(3))
    v0, q0, qt0 = copy.deepcopy(t.v), copy.deepcopy(t.q), copy.deepcopy(t.q_target)

    t.update(batch)

    # Expected V gradient: the expectile loss on the target twin's min, nothing else.
    adv = torch.min(*qt0(batch.obs, batch.act)).detach() - v0(batch.obs)
    expectile_loss(adv, 0.7).backward()
    for p_t, p_ref in zip(t.v.parameters(), v0.parameters(), strict=True):
        torch.testing.assert_close(p_t.grad, p_ref.grad, rtol=RTOL, atol=ATOL)
    # Expected Q gradient: the TD loss alone (no path from V's loss into Q).
    with torch.no_grad():
        y = batch.rew + (1.0 - batch.done) * 0.99 * v0(batch.next_obs)
    q1, q2 = q0(batch.obs, batch.act)
    (((q1 - y) ** 2).mean() / 2 + ((q2 - y) ** 2).mean() / 2).backward()
    for p_t, p_ref in zip(t.q.parameters(), q0.parameters(), strict=True):
        torch.testing.assert_close(p_t.grad, p_ref.grad, rtol=RTOL, atol=ATOL)


def test_i6_soft_update_is_exact_and_target_gets_no_gradient(f64):
    t = make_trainer()
    qt0 = copy.deepcopy(t.q_target)
    t.update(make_batch(torch.Generator().manual_seed(4)))
    for new, old, q in zip(t.q_target.parameters(), qt0.parameters(), t.q.parameters(), strict=True):
        torch.testing.assert_close(new, 0.995 * old + 0.005 * q, rtol=0, atol=1e-12)
        assert not new.requires_grad and new.grad is None


# ---------- I7: one full step against a straight-line reference ----------


def reference_step(nets, opts, sched, batch, cfg):
    """Spec Section 2 written out by hand: own Gaussian log-density, own MSE, own Polyak."""
    a = cfg["algo"]
    policy, q, q_targ, v = nets
    with torch.no_grad():
        next_v = v(batch.next_obs)
        tq1, tq2 = q_targ(batch.obs, batch.act)
        target_q = torch.minimum(tq1, tq2)
    v_now = v(batch.obs)
    adv = target_q - v_now
    # float32 weights: CORL casts the mask with .float(), and offrl copies that (expectile_loss)
    weight = torch.where(adv < 0, 1 - a["expectile"], a["expectile"]).float()
    v_loss = (weight * adv**2).mean()
    opts["v"].zero_grad()
    v_loss.backward()
    opts["v"].step()

    y = batch.rew + a["discount"] * (1 - batch.done) * next_v
    q1, q2 = q(batch.obs, batch.act)
    q_loss = 0.5 * (((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean())
    opts["q"].zero_grad()
    q_loss.backward()
    opts["q"].step()
    with torch.no_grad():
        rate = a["target_update_rate"]
        for pt, p in zip(q_targ.parameters(), q.parameters(), strict=True):
            pt.mul_(1 - rate).add_(rate * p)

    w = torch.exp(a["beta"] * adv.detach()).clamp(max=a["exp_adv_max"]).flatten()
    mean = torch.tanh(policy.net(batch.obs))
    log_std = policy.log_std.clamp(-20.0, 2.0)
    z = (batch.act - mean) / log_std.exp()
    log_prob = (-0.5 * z**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    pi_loss = (w * -log_prob).mean()
    opts["pi"].zero_grad()
    pi_loss.backward()
    opts["pi"].step()
    sched.step()
    return {
        "v_loss": v_loss,
        "q_loss": q_loss,
        "policy_loss": pi_loss,
        "v_mean": v_now.mean(),
        "q_mean": torch.minimum(q1, q2).mean(),
        "q_target_mean": target_q.mean(),
        "adv_mean": adv.mean(),
        "adv_std": adv.std(),
        "weight_mean": w.mean(),
        "log_std_mean": policy.log_std.clamp(-20.0, 2.0).mean(),  # after the step, as BC logs it
    }


def test_i7_one_step_matches_the_straight_line_reference(f64):
    cfg = config()
    t = make_trainer()
    spread_targets(t)
    batch = make_batch(torch.Generator().manual_seed(5))
    branch_coverage(t, batch)
    nets = [copy.deepcopy(m) for m in (t.policy, t.q, t.q_target, t.v)]
    lr = cfg["train"]["lr"]
    opts = {
        "pi": torch.optim.Adam(nets[0].parameters(), lr=lr),
        "q": torch.optim.Adam(nets[1].parameters(), lr=lr),
        "v": torch.optim.Adam(nets[3].parameters(), lr=lr),
    }
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opts["pi"], T_max=cfg["train"]["steps"])

    ref = reference_step(nets, opts, sched, batch, cfg)
    out = t.update(batch)

    for ours, theirs in zip((t.policy, t.q, t.q_target, t.v), nets, strict=True):
        for (name, p), (_, r) in zip(ours.named_parameters(), theirs.named_parameters(), strict=True):
            torch.testing.assert_close(p, r, rtol=RTOL, atol=ATOL, msg=name)
    for key, value in ref.items():
        torch.testing.assert_close(out[key], value.detach(), rtol=RTOL, atol=ATOL, msg=key)
    assert out["frac_clipped"].item() > 0
    assert out["actor_lr"].item() == cfg["train"]["lr"]
    assert set(out) == set(IQLTrainer.LOG_KEYS)


# ---------- I2: beta = 0 is BC ----------


def test_i2_beta_zero_is_bc_bitwise():
    cfg = config(beta=0.0)
    gen = torch.Generator().manual_seed(6)
    batches = [make_batch(gen) for _ in range(50)]
    seed_everything(7)
    bc = BCTrainer(cfg, OBS, ACT, torch.device("cpu"))
    seed_everything(7)
    iql = IQLTrainer(cfg, OBS, ACT, torch.device("cpu"))
    first = (bc.update(batches[0])["policy_loss"], iql.update(batches[0])["policy_loss"])
    assert torch.equal(*first)
    for b in batches[1:]:
        bc.update(b)
        iql.update(b)
    for p, q in zip(bc.policy.parameters(), iql.policy.parameters(), strict=True):
        assert torch.equal(p, q)


# ---------- I9, I11: learning behaviour ----------


def test_i9_td_fixed_point_on_constant_reward(f64):
    """Constant r, no terminations, discount 0.5: Q = V = r / (1 - 0.5) = 2r.

    A faster target update (0.05) reaches the fixed point in 800 steps instead of 3,000, which
    keeps the test near 10 s on a slow CPU; the fixed point does not depend on the rate.
    """
    start = time.perf_counter()
    r = 1.0
    t = make_trainer(seed=8, discount=0.5, lr=1e-3, steps=800, target_update_rate=0.05)
    gen = torch.Generator().manual_seed(9)
    out = {}
    for _ in range(800):
        b = make_batch(gen, rew_scale=0.0)
        b = Batch(b.obs, b.act, torch.full_like(b.rew, r), b.next_obs, torch.zeros_like(b.done))
        out = t.update(b)
    q, v = out["q_mean"].item(), out["v_mean"].item()
    print(f"I9: Q {q:.4f}, V {v:.4f}, expected {2 * r}, {time.perf_counter() - start:.1f} s")
    assert q == pytest.approx(2 * r, rel=0.05) and v == pytest.approx(2 * r, rel=0.05)


def test_i11_value_learning_moves_the_policy_bc_does_not(f64):
    steps = 1000  # IQL reaches about 0.72 by 1,000 steps (0.72 at 2,000 too)
    means = {}
    for name, cls in (("iql", IQLTrainer), ("bc", BCTrainer)):
        seed_everything(10)
        t = cls(config(beta=3.0, steps=steps, lr=1e-3), OBS, ACT, torch.device("cpu"))
        gen = torch.Generator().manual_seed(11)
        for _ in range(steps):
            t.update(two_mode_batch(gen))
        test_obs = torch.randn(512, OBS, generator=torch.Generator().manual_seed(12))
        means[name] = t.act(test_obs)[:, 0].mean().item()
    print(f"I11: mean action dim 0, IQL {means['iql']:.3f}, BC {means['bc']:.3f}")
    assert means["iql"] > 0.5
    assert abs(means["bc"]) < 0.2
