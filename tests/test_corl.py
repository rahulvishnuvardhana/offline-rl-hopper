"""I8 and I12 (Phase 5 spec Section 7): our IQL against CORL's own update, and the vendored file."""

import hashlib
from pathlib import Path

import pytest
import torch

from offrl._vendor import corl_iql as corl
from offrl.algos.corl import CorlTrainer
from test_iql import ACT, ATOL, OBS, RTOL, branch_coverage, config, make_batch, make_trainer, spread_targets

VENDORED = Path(corl.__file__)
README = VENDORED.with_name("README.md")


def corl_name(ours: str) -> str:
    """Our parameter name -> CORL's: CORL's MLP wraps its layers in ``.net``."""
    head, _, rest = ours.partition(".")
    return ours if head == "log_std" else f"{head}.net.{rest}"


def copy_into(theirs: torch.nn.Module, ours: torch.nn.Module) -> None:
    """Every parameter on both sides mapped, none left over."""
    src = dict(ours.named_parameters())
    dst = dict(theirs.named_parameters())
    assert {corl_name(k) for k in src} == set(dst), (sorted(src), sorted(dst))
    with torch.no_grad():
        for k, p in src.items():
            dst[corl_name(k)].copy_(p)


def assert_same(ours: torch.nn.Module, theirs: torch.nn.Module) -> None:
    dst = dict(theirs.named_parameters())
    for k, p in ours.named_parameters():
        torch.testing.assert_close(p, dst[corl_name(k)], rtol=RTOL, atol=ATOL, msg=k)


def build_corl(t):
    """The adapter as train.run builds it, then our weights copied in, all in float64."""
    adapter = CorlTrainer(config(), OBS, ACT, torch.device("cpu"))
    iql = adapter.iql
    for net in (iql.qf, iql.q_target, iql.vf, iql.actor):
        net.double()  # in place, so the optimizers still hold these parameters; log_std was float32
    for theirs, ours in ((iql.qf, t.q), (iql.q_target, t.q_target), (iql.vf, t.v), (iql.actor, t.policy)):
        copy_into(theirs, ours)  # the target too: ours was spread for branch coverage
    return adapter


@pytest.mark.parametrize("n_steps", [1, 20])
def test_i8_matches_corl_implicit_q_learning(f64, n_steps):
    """Through ``CorlTrainer.update``, so the adapter's wiring (tau, discount, beta, max_steps,
    the log names) is under test too, not only CORL's ``train``."""
    t = make_trainer()
    spread_targets(t)
    gen = torch.Generator().manual_seed(5)
    batches = [make_batch(gen) for _ in range(n_steps)]
    branch_coverage(t, batches[0])
    adapter = build_corl(t)
    iql = adapter.iql
    for b in batches:
        ours = t.update(b)
        theirs = adapter.update(b)
    for net, other in ((t.policy, iql.actor), (t.q, iql.qf), (t.q_target, iql.q_target), (t.v, iql.vf)):
        assert_same(net, other)
    for key in CorlTrainer.LOG_KEYS:
        torch.testing.assert_close(theirs[key], ours[key], rtol=RTOL, atol=ATOL, msg=key)
    assert t.opt.param_groups[0]["lr"] == iql.actor_optimizer.param_groups[0]["lr"]


def test_i12_vendored_file_is_unmodified():
    body = VENDORED.read_bytes().split(b"\n", 1)[1]  # line 1 is our notice
    readme = README.read_text(encoding="utf-8")
    recorded = readme.split("checked by `tests/test_corl.py` (I12):")[1].split()[0]
    assert hashlib.sha256(body).hexdigest() == recorded.strip("`")
    assert corl.ImplicitQLearning and corl.EXP_ADV_MAX == 100.0


def test_adapter_refuses_settings_corl_hard_codes():
    cfg = config()
    cfg["net"]["hidden"] = [32, 64]
    with pytest.raises(ValueError, match="one width"):
        CorlTrainer(cfg, OBS, ACT, torch.device("cpu"))
    cfg = config()
    cfg["algo"]["exp_adv_max"] = 50.0
    with pytest.raises(ValueError, match="EXP_ADV_MAX"):
        CorlTrainer(cfg, OBS, ACT, torch.device("cpu"))


def test_adapter_act_and_state_round_trip():
    torch.manual_seed(0)
    a = CorlTrainer(config(), OBS, ACT, torch.device("cpu"))
    b = CorlTrainer(config(), OBS, ACT, torch.device("cpu"))
    gen = torch.Generator().manual_seed(1)
    out = a.update(make_batch(gen))
    assert set(out) == set(CorlTrainer.LOG_KEYS)
    b.load_state_dict(a.state_dict())
    obs = torch.randn(8, OBS, generator=gen)
    assert torch.equal(a.act(obs), b.act(obs)) and a.act(obs).abs().max() <= 1
    for p, q in zip(a.iql.q_target.parameters(), b.iql.q_target.parameters(), strict=True):
        assert torch.equal(p, q) and not q.requires_grad  # the target is restored, not reset to Q
