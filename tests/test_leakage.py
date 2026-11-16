"""Leakage checks: each passes on clean synthetic data and fails on the leak it guards.

The config test needs no datasets, so CI blocks any eval-seed change that would start
evaluation episodes from training states.
"""

from pathlib import Path

import numpy as np

from offrl.config import load_config
from offrl.data import leakage
from offrl.data.load import Episode
from offrl.rollout import FINAL_EVAL_SEED_OFFSET, RANDOM_BASELINE_SEEDS, eval_seeds

ROOT = Path(__file__).resolve().parents[1]


def make_episode(ep_id: int, n: int = 5, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed + ep_id)
    term = np.zeros(n, dtype=bool)
    term[-1] = True
    return Episode(ep_id, rng.normal(size=(n + 1, 11)), rng.uniform(-1, 1, (n, 3)), np.ones(n), term, ~term)


def fake_reset(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=11)


def episodes_from_seeds(seeds, data_seed: int) -> list[Episode]:
    eps = []
    for i, s in enumerate(seeds):
        ep = make_episode(i, seed=data_seed)
        obs = ep.obs.copy()
        obs[0] = fake_reset(s)
        eps.append(Episode(ep.id, obs, ep.act, ep.rew, ep.term, ep.trunc))
    return eps


def test_collection_seeds_verified_and_mismatch_detected():
    eps = episodes_from_seeds(range(123, 133), data_seed=0)
    assert leakage.check_collection_seeds(eps, fake_reset, range(123, 133)).passed
    assert not leakage.check_collection_seeds(eps, fake_reset, range(124, 134)).passed
    assert not leakage.check_collection_seeds(eps, fake_reset, range(123, 132)).passed  # count mismatch


def test_seeds_disjoint():
    collection = [range(123, 1209), range(123, 2075)]
    assert leakage.check_seeds_disjoint("eval", range(10_000, 10_010), collection).passed
    clash = leakage.check_seeds_disjoint("eval", range(2_070, 2_080), collection)
    assert not clash.passed and "5 overlapping" in clash.detail


def test_cross_duplicates_detected():
    a = [make_episode(i, seed=0) for i in range(3)]
    b = [make_episode(i, seed=100) for i in range(3)]
    assert leakage.check_no_cross_duplicates(a, b).passed
    assert not leakage.check_no_cross_duplicates(a, [*b, a[1]]).passed


def test_shared_states_only_initial():
    a = episodes_from_seeds(range(5), data_seed=0)
    b = episodes_from_seeds(range(5), data_seed=100)  # same reset seeds: initial states shared
    ok = leakage.check_shared_states_are_initial(a, b)
    assert ok.passed and ok.detail.startswith("5 exact shared")
    obs = b[0].obs.copy()
    obs[2] = a[3].obs[2]  # a mid-trajectory state copied across datasets
    b[0] = Episode(b[0].id, obs, b[0].act, b[0].rew, b[0].term, b[0].trunc)
    assert not leakage.check_shared_states_are_initial(a, b).passed


def test_collection_ranges_match_documented_counts():
    assert len(leakage.COLLECTION_SEEDS["mujoco/hopper/expert-v0"]) == 1_086
    assert len(leakage.COLLECTION_SEEDS["mujoco/hopper/simple-v0"]) == 1_952


def test_configured_rollout_seeds_never_reuse_collection_seeds():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    collection = leakage.COLLECTION_SEEDS.values()
    assert leakage.check_seeds_disjoint("eval", eval_seeds(cfg["eval"]), collection).passed
    final_cfg = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "final.yaml")
    assert final_cfg["eval"]["seed_offset"] == FINAL_EVAL_SEED_OFFSET  # final runs get reserved seeds
    final = {**final_cfg["eval"], "episodes": 50}
    assert leakage.check_seeds_disjoint("final", eval_seeds(final), collection).passed
    # development and final evaluation seeds never overlap either
    assert not set(eval_seeds(cfg["eval"])) & set(eval_seeds(final))
    assert leakage.check_seeds_disjoint("random", RANDOM_BASELINE_SEEDS, collection).passed
