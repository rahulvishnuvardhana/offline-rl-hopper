"""E1, E2, E4 (CI, no dataset) and E3 (local, needs the datasets). Spec Section 7."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from offrl.algos.bc import BCTrainer
from offrl.analysis.eda import normalized_score
from offrl.data.buffer import ReplayBuffer
from offrl.data.leakage import COLLECTION_SEED_OFFSET
from offrl.evaluate import deterministic_policy, evaluate, sampled_policy
from offrl.rollout import run_policy

CFG = {
    "net": {"hidden": [32, 32], "log_std_min": -20.0, "log_std_max": 2.0},
    "train": {"lr": 3e-4, "steps": 10, "actor_cosine_lr": True},
    "eval": {"episodes": 2, "seed_offset": 20000},
    "score": {"r_random": 17.0, "r_expert": 3857.0},
}


@pytest.fixture
def setup(make_mixture):
    torch.manual_seed(0)
    buffer = ReplayBuffer.from_mixture(make_mixture(n=100), False, "cpu", seed=0)
    trainer = BCTrainer(CFG, 11, 3, torch.device("cpu"))
    return buffer, trainer


def test_e1_same_policy_same_seeds_same_returns(setup):
    buffer, trainer = setup
    env = gym.make("Hopper-v5")
    fn = deterministic_policy(trainer, buffer.normalizer, torch.device("cpu"))
    a, rows_a = evaluate(env, fn, CFG, 3)
    b, rows_b = evaluate(env, fn, CFG, 3)
    assert a == b and rows_a == rows_b
    assert [r["seed"] for r in rows_a] == [20000, 20001, 20002]
    assert a["n_episodes"] == 3 and a["sub_n_episodes"] == 2  # the 10-seed subset (here 2)
    assert a["sub_return_mean"] == pytest.approx(np.mean([r["return"] for r in rows_a[:2]]))
    env.close()


def test_sampled_diagnostic_is_reproducible_and_leaves_global_rng_alone(setup):
    buffer, trainer = setup
    env = gym.make("Hopper-v5")
    state = torch.get_rng_state()
    runs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(5)
        fn = sampled_policy(trainer.policy, buffer.normalizer, torch.device("cpu"), gen)
        runs.append(evaluate(env, fn, CFG, 2)[1])
    assert runs[0] == runs[1]
    assert torch.equal(state, torch.get_rng_state())
    env.close()


def test_e2_evaluator_uses_the_buffers_normalizer(setup, make_mixture):
    buffer, trainer = setup
    arrays = make_mixture(n=100)
    np.testing.assert_array_equal(buffer.normalizer.transform(arrays["obs"]), buffer.obs.numpy())
    seen = []
    trainer.act = lambda x: (seen.append(x.clone()), torch.zeros(1, 3))[1]
    deterministic_policy(trainer, buffer.normalizer, torch.device("cpu"))(arrays["obs"][7].astype(np.float64))
    assert torch.equal(seen[0][0], buffer.obs[7])  # bitwise: same object, same float32 transform


def test_e4_score_anchors():
    assert normalized_score(17.0288, 17.0288, 3857.8036) == 0.0
    assert normalized_score(3857.8036, 17.0288, 3857.8036) == 100.0


@pytest.mark.requires_data
def test_e3_replaying_dataset_actions_reproduces_the_data():
    """Collection seeds are used on purpose, via run_policy directly; the evaluator never sees them.
    Hard checks: reset obs, first 100 steps, length and fall flag. Full-episode return: relative
    error <= 1e-5, pre-registered from the first measurement (max 4.0e-7 over these 6 episodes,
    2026-11-16, MuJoCo 3.2.3), so a 25x margin."""
    import minari

    from offrl.data.load import load_episodes

    for dataset_id in ("mujoco/hopper/expert-v0", "mujoco/hopper/simple-v0"):
        env = minari.load_dataset(dataset_id).recover_environment()
        for i, ep in enumerate(load_episodes(dataset_id)[:3]):
            obs, _ = env.reset(seed=COLLECTION_SEED_OFFSET + i)
            assert np.abs(obs - ep.obs[0]).max() <= 1e-8
            for t in range(100):
                obs, *_ = env.step(ep.act[t])
                assert np.abs(obs - ep.obs[t + 1]).max() <= 1e-9
            actions = iter(ep.act)
            out = run_policy(env, lambda _obs, it=actions: next(it), [COLLECTION_SEED_OFFSET + i])
            assert out["lengths"][0] == ep.length and bool(out["fell"][0]) == bool(ep.term[-1])
            rel = abs(out["returns"][0] - ep.ret) / abs(ep.ret)
            assert rel <= 1e-5, f"{dataset_id} ep{i}: relative return error {rel:.2e}"
        env.close()
