"""EDA functions on synthetic episodes whose answers are known exactly."""

import numpy as np
import pytest

from offrl.analysis import eda
from offrl.data.load import Episode


def make_episode(ep_id: int, n: int, fell: bool, reward: float = 1.0, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed + ep_id)
    term = np.zeros(n, dtype=bool)
    trunc = np.zeros(n, dtype=bool)
    (term if fell else trunc)[-1] = True
    return Episode(
        id=ep_id,
        obs=rng.normal(size=(n + 1, 11)),
        act=rng.uniform(-1, 1, size=(n, 3)),
        rew=np.full(n, reward),
        term=term,
        trunc=trunc,
    )


@pytest.fixture
def episodes():
    # returns: 10*1 = 10 (fell), 20*2 = 40 (fell), 30*3 = 90 (no fall)
    return [make_episode(0, 10, True, 1.0), make_episode(1, 20, True, 2.0), make_episode(2, 30, False, 3.0)]


def test_episode_stats_and_fall_split(episodes):
    s = eda.episode_stats(episodes)
    assert s["n_episodes"] == 3 and s["n_transitions"] == 60
    assert s["fall_rate"] == pytest.approx(2 / 3)
    assert s["return"]["mean"] == pytest.approx(140 / 3)
    assert s["return"]["median"] == 40.0
    assert s["fell"]["n_episodes"] == 2 and s["fell"]["n_transitions"] == 30
    assert s["fell"]["return"]["mean"] == 25.0
    assert s["no_fall"]["return"]["mean"] == 90.0
    assert s["no_fall"]["share_transitions"] == 0.5
    # C4 identity: count-weighted group means reproduce the overall mean
    total = s["fell"]["return"]["mean"] * 2 + s["no_fall"]["return"]["mean"] * 1
    assert total / 3 == pytest.approx(s["return"]["mean"], rel=1e-12)


def test_empty_group_is_none_not_nan():
    s = eda.episode_stats([make_episode(0, 5, False)])
    assert s["fell"]["return"] is None and s["fell"]["n_episodes"] == 0
    assert eda.describe(np.array([])) is None


def test_state_stats_exclude_final_observation():
    ep = make_episode(0, 4, False)
    obs = ep.obs.copy()
    obs[-1] = 1e6  # a huge final observation must not move the stats
    ep = Episode(ep.id, obs, ep.act, ep.rew, ep.term, ep.trunc)
    stats = eda.state_stats([ep])
    np.testing.assert_allclose(stats["mean"], ep.obs[:-1].mean(0))


def test_velocity_clipping_fraction():
    ep = make_episode(0, 4, False)
    obs = np.zeros((5, 11))
    obs[0, 5], obs[1, 5], obs[2, 10] = 10.0, -10.0, 10.0
    obs[3, 6] = 9.999  # close, but not clipped
    obs[4, 5] = 10.0  # final observation: excluded
    ep = Episode(ep.id, obs, ep.act, ep.rew, ep.term, ep.trunc)
    clip = eda.velocity_clipping([ep])
    assert clip["obs[5]"] == 0.5 and clip["obs[10]"] == 0.25 and clip["obs[6]"] == 0.0
    assert set(clip) == {f"obs[{d}]" for d in range(5, 11)}


def test_action_stats_bound_fraction():
    ep = make_episode(0, 4, False)
    act = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]])
    stats = eda.action_stats([Episode(ep.id, ep.obs, act, ep.rew, ep.term, ep.trunc)])
    assert stats["frac_at_bound"] == [0.5, 0.25, 0.0]


def test_reward_decomposition_zero_residual_when_reward_is_rebuilt():
    ep = make_episode(0, 50, False)
    rew = 1.0 + ep.obs[1:, 5] - 1e-3 * (ep.act**2).sum(1)
    d = eda.reward_decomposition([Episode(ep.id, ep.obs, ep.act, rew, ep.term, ep.trunc)])
    assert d["residual_abs_max"] < 1e-12
    assert d["mean_healthy"] == 1.0
    assert d["mean_reward"] == pytest.approx(rew.mean())


def test_reward_decomposition_no_survival_bonus_on_fall_step():
    ep = make_episode(0, 20, True)  # Hopper-v5: the terminal step is unhealthy, bonus 0
    healthy = np.r_[np.ones(19), 0.0]
    rew = healthy + ep.obs[1:, 5] - 1e-3 * (ep.act**2).sum(1)
    t = eda.reward_terms([Episode(ep.id, ep.obs, ep.act, rew, ep.term, ep.trunc)])
    np.testing.assert_array_equal(t["healthy"], healthy)
    assert np.abs(t["residual"]).max() < 1e-12


def test_reward_decomposition_catches_doubled_reward():
    ep = make_episode(0, 50, False)
    rew = 2.0 * (1.0 + ep.obs[1:, 5] - 1e-3 * (ep.act**2).sum(1))
    d = eda.reward_decomposition([Episode(ep.id, ep.obs, ep.act, rew, ep.term, ep.trunc)])
    assert abs(d["residual_mean"]) > 0.5


def test_subsample_states_keeps_episode_ids_and_rows():
    eps = [make_episode(i, 30, False) for i in range(5)]
    x, a, ids = eda.subsample_states(eps, 40, np.random.default_rng(0))
    assert x.shape == (40, 11) and a.shape == (40, 3) and ids.shape == (40,)
    by_id = {ep.id: ep for ep in eps}
    for state, action, i in zip(x, a, ids, strict=True):
        row = np.flatnonzero((by_id[i].obs[:-1] == state).all(1))
        assert len(row) == 1 and (by_id[i].act[row[0]] == action).all()
    with pytest.raises(ValueError):
        eda.subsample_states(eps, 151, np.random.default_rng(0))


def test_split_by_episode_never_splits_an_episode():
    ids = np.repeat(np.arange(10), 7)
    mask = eda.split_by_episode(ids, 0.8, np.random.default_rng(0))
    assert mask.sum() == 56
    for i in range(10):
        assert mask[ids == i].all() or not mask[ids == i].any()


def test_pca_recovers_dominant_direction_with_sign_convention():
    rng = np.random.default_rng(0)
    direction = np.array([3.0, -4.0, 0.0]) / 5.0
    x = rng.normal(size=(2000, 1)) * 10 * direction + rng.normal(size=(2000, 3)) * 0.1
    comps, evr, proj = eda.pca_2d(x)
    assert abs(comps[0] @ direction) > 0.999
    assert comps[0, np.abs(comps[0]).argmax()] > 0  # largest-abs loading positive
    assert evr[0] > 0.99 and proj.shape == (2000, 2)
    comps_neg, _, _ = eda.pca_2d(-x)  # mirrored data: same components after the sign fix
    np.testing.assert_allclose(comps_neg, comps, atol=1e-9)


@pytest.mark.parametrize("block", [1, 7, 64, 1000])
def test_knn_indices_equals_brute_force(block):
    rng = np.random.default_rng(1)
    q, r = rng.normal(size=(50, 4)), rng.normal(size=(200, 4))
    brute = np.linalg.norm(q[:, None] - r[None], axis=2).argsort(1)[:, :5]
    np.testing.assert_array_equal(eda.knn_indices(q, r, 5, block=block), brute)


def test_roc_auc_known_values():
    labels = np.array([1, 1, 0, 0], dtype=bool)
    assert eda.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 1.0
    assert eda.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 0.0
    assert eda.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert eda.roc_auc(np.array([0.9, 0.3, 0.5, 0.1]), labels) == 0.75
    with pytest.raises(ValueError):
        eda.roc_auc(np.array([0.1, 0.2]), np.array([True, True]))


def _shifted(ep: Episode, shift: float, act_shift: float) -> Episode:
    return Episode(ep.id, ep.obs + shift, np.clip(ep.act + act_shift, -1, 1), ep.rew, ep.term, ep.trunc)


def test_separability_detects_separated_sources():
    expert = [_shifted(make_episode(i, 40, False), 5.0, 0.0) for i in range(20)]
    simple = [make_episode(100 + i, 40, True) for i in range(20)]
    numbers, arrays = eda.separability(expert, simple, 400, seed=0, k=5)
    assert numbers["auc"] == 1.0
    assert len(arrays["d_mix"]) == numbers["n_test_simple"]


def test_separability_same_distribution_is_near_chance():
    expert = [make_episode(i, 40, False, seed=0) for i in range(30)]
    simple = [make_episode(i, 40, False, seed=1000) for i in range(30)]
    numbers, _ = eda.separability(expert, simple, 600, seed=0, k=10)
    assert 0.35 < numbers["auc"] < 0.65
    assert 0.7 < numbers["ratio"] < 1.4


def test_separability_action_disagreement_at_shared_states():
    # Different episode lengths, so the expert and simple test sets have different sizes and
    # querying the wrong source's test states would change the array lengths.
    expert = [make_episode(i, 12, False, seed=0) for i in range(100)]
    simple = [_shifted(make_episode(i, 40, False, seed=1000), 0.0, 0.8) for i in range(30)]
    numbers, arrays = eda.separability(expert, simple, 600, seed=0, k=10)
    assert numbers["ratio"] > 2.0  # same states, different actions: supports H1
    assert len(arrays["d_mix"]) == numbers["n_test_simple"]
    assert len(arrays["d_same"]) == numbers["n_test_expert"]
    assert numbers["n_test_simple"] != numbers["n_test_expert"]
    # shared states: nearest expert and nearest simple neighbours are equally close
    ratio = numbers["nn_dist_simple_to_expert_median"] / numbers["nn_dist_simple_to_simple_median"]
    assert 0.7 < ratio < 1.4


def test_normalized_score_endpoints():
    assert eda.normalized_score(17.0, 17.0, 3858.0) == 0.0
    assert eda.normalized_score(3858.0, 17.0, 3858.0) == 100.0


def _lengths():
    return {eda.EXPERT: np.array([100, 80, 120, 50]), eda.SIMPLE: np.array([30, 60, 90, 40, 70])}


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
def test_sample_mixture_hits_budget_exactly(p):
    picks = eda.sample_mixture_episodes(_lengths(), p, 200, seed=3)
    n_e = sum(n for _, n in picks[eda.EXPERT])
    n_s = sum(n for _, n in picks[eda.SIMPLE])
    assert n_e == round(p * 200) and n_e + n_s == 200
    if p == 0.0:
        assert picks[eda.EXPERT] == []
    if p == 1.0:
        assert picks[eda.SIMPLE] == []
    for source, taken in picks.items():
        assert len({i for i, _ in taken}) == len(taken)  # no episode twice
        assert all(0 < n <= _lengths()[source][i] for i, n in taken)
        assert all(n == _lengths()[source][i] for i, n in taken[:-1])  # only the last is trimmed


def test_sample_mixture_deterministic_and_budget_error():
    a = eda.sample_mixture_episodes(_lengths(), 0.5, 200, seed=7)
    assert a == eda.sample_mixture_episodes(_lengths(), 0.5, 200, seed=7)
    with pytest.raises(ValueError):
        eda.sample_mixture_episodes(_lengths(), 1.0, 351, seed=0)


def test_mixture_preview_trimmed_return_and_scale():
    episodes = {
        eda.EXPERT: [make_episode(0, 10, False, reward=5.0)],
        eda.SIMPLE: [make_episode(1, 10, True, reward=1.0), make_episode(2, 10, True, reward=1.0)],
    }
    out = eda.mixture_preview(episodes, 0.5, 20, seed=0)  # 10 expert steps + 10 simple steps
    assert out["n_episodes"] == 2 and out["n_trimmed"] == 0
    assert out["mean_return"] == 30.0  # (50 + 10) / 2
    assert out["reward_scale"] == pytest.approx(1000.0 / 40.0)


def test_mixture_reward_scale_ignores_trimmed_tail():
    episodes = {
        eda.EXPERT: [make_episode(0, 10, False, reward=5.0)],
        eda.SIMPLE: [make_episode(i, 10, True, reward=1.0) for i in (1, 2, 3)],
    }
    out = eda.mixture_preview(episodes, 0.4, 25, seed=0)  # simple: 10 + 5 steps, one trimmed to 5
    assert out["n_trimmed"] == 1 and out["n_episodes"] == 3
    assert out["mean_return"] == pytest.approx((50 + 10 + 5) / 3)  # trimmed counted in the mean
    assert out["reward_scale"] == pytest.approx(1000.0 / 40.0)  # but not in the scale (else /45)


def test_sanity_bands(episodes):
    stats = eda.episode_stats(episodes)
    lo, hi = eda.sanity_bands(stats, r_random=0.0, r_expert=90.0)
    assert lo == pytest.approx(100 * 25 / 90) and hi == 100.0


def test_compare_numeric():
    a = {"x": 1.0, "y": [1, 2.0, "s"], "z": None, "b": True}
    assert eda.compare_numeric(a, {"x": 1.0 + 1e-12, "y": [1, 2.0, "s"], "z": None, "b": True}) == []
    diffs = eda.compare_numeric(a, {"x": 1.1, "y": [1, 2.0, "t"], "z": None, "b": False})
    assert len(diffs) == 3
    assert eda.compare_numeric({"x": 1}, {"y": 1}) == [": keys differ"]
