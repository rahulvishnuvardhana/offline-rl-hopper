"""Each check must pass on well-formed synthetic data and fail on the specific corruption it guards."""

from dataclasses import replace

import numpy as np
import pytest

from offrl.data import validate as v
from offrl.data.load import Episode


def make_episode(ep_id: int, n: int, terminated: bool, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed + ep_id)
    obs = rng.uniform(-1, 1, size=(n + 1, 11))
    obs[:, 0] = 1.25  # healthy torso height
    obs[:, 1] = 0.0  # healthy torso angle
    if terminated:
        obs[-1, 0] = 0.5  # fell
    term = np.zeros(n, dtype=bool)
    trunc = np.zeros(n, dtype=bool)
    (term if terminated else trunc)[-1] = True
    return Episode(
        id=ep_id,
        obs=obs,
        act=rng.uniform(-1, 1, size=(n, 3)),
        rew=rng.uniform(0, 3, size=n),
        term=term,
        trunc=trunc,
    )


@pytest.fixture
def episodes():
    return [make_episode(0, 120, terminated=True), make_episode(1, v.MAX_EPISODE_STEPS, terminated=False)]


def documented(eps):
    return {"total_steps": sum(e.length for e in eps), "total_episodes": len(eps)}


def by_name(checks):
    return {c.name: c for c in checks}


def test_clean_data_passes_every_check(episodes):
    checks = v.run_all(episodes, documented(episodes), documented(episodes))
    failed = [c for c in checks if not c.passed]
    assert not failed, failed
    assert len(checks) == 9


def test_counts_mismatch_with_metadata_fails(episodes):
    meta = {"total_steps": 999, "total_episodes": 2}
    assert not v.check_counts(episodes, documented(episodes), meta).passed


def test_obs_off_by_one_fails(episodes):
    bad = replace(episodes[0], obs=episodes[0].obs[:-1])
    assert not v.check_shapes([bad]).passed


def test_nan_fails(episodes):
    obs = episodes[0].obs.copy()
    obs[3, 4] = np.nan
    assert not v.check_finite([replace(episodes[0], obs=obs)]).passed


def test_action_out_of_bounds_fails(episodes):
    act = episodes[0].act.copy()
    act[0, 0] = 1.5
    assert not v.check_action_bounds([replace(episodes[0], act=act)]).passed


def test_mid_episode_termination_fails(episodes):
    term = episodes[0].term.copy()
    term[10] = True
    assert not v.check_episode_ends([replace(episodes[0], term=term)]).passed


def test_early_truncation_fails():
    short_trunc = make_episode(5, 300, terminated=False)
    assert not v.check_episode_ends([short_trunc]).passed


def test_both_flags_on_last_step_fails(episodes):
    trunc = episodes[0].trunc.copy()
    trunc[-1] = True
    assert not v.check_episode_ends([replace(episodes[0], trunc=trunc)]).passed


def test_obs_beyond_health_bound_fails(episodes):
    obs = episodes[0].obs.copy()
    obs[5, 7] = 150.0
    assert not v.check_obs_bounds([replace(episodes[0], obs=obs)]).passed


def test_termination_label_on_healthy_state_fails(episodes):
    obs = episodes[0].obs.copy()
    obs[-1, 0] = 1.25  # labelled terminal but still standing
    assert not v.check_termination_physics([replace(episodes[0], obs=obs)]).passed


def test_fall_without_termination_label_fails(episodes):
    obs = episodes[1].obs.copy()
    obs[50, 1] = 0.5  # torso tipped over mid-episode, no terminal flag
    assert not v.check_termination_physics([replace(episodes[1], obs=obs)]).passed


def test_duplicate_episode_and_id_fail(episodes):
    clone = replace(episodes[0])
    assert not v.check_no_duplicate_episodes([*episodes, clone]).passed
    assert not v.check_unique_ids([*episodes, clone]).passed
