"""Mixture builder on synthetic episodes with known answers, plus traps for the classic bugs."""

import hashlib

import numpy as np
import pytest

from offrl.analysis import eda
from offrl.data import mixture as mx
from offrl.data.load import Episode


def make_episode(ep_id: int, n: int, fell: bool, base: float) -> Episode:
    """Every value encodes (episode, step), so a row from the wrong episode is detectable."""
    steps = np.arange(n + 1, dtype=np.float64)
    obs = base + ep_id * 1000.0 + steps[:, None] + np.zeros((1, 11))
    term = np.zeros(n, dtype=bool)
    trunc = np.zeros(n, dtype=bool)
    (term if fell else trunc)[-1] = True
    return Episode(ep_id, obs, np.full((n, 3), 0.1 * (ep_id % 7)), np.full(n, 1.0 + ep_id), term, trunc)


@pytest.fixture
def episodes():
    return {
        mx.EXPERT: [make_episode(i, 10 + i, fell=i % 2 == 0, base=0.0) for i in range(6)],
        mx.SIMPLE: [make_episode(i, 5 + i, fell=i % 3 == 0, base=1e6) for i in range(8)],
    }


def lengths(episodes):
    return {s: np.array([ep.length for ep in episodes[s]]) for s in mx.SOURCES}


def build(episodes, p, seed, n=60):
    picks = mx.sample_mixture_episodes(lengths(episodes), p, n, seed)
    return mx.flatten(episodes, picks), picks


def test_sources_order_pinned_and_shared_with_eda():
    assert mx.SOURCES == ("expert", "simple")
    assert eda.sample_mixture_episodes is mx.sample_mixture_episodes
    assert eda.SOURCES is mx.SOURCES


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
def test_budget_exact_and_pure_endpoints(episodes, p):
    arrays, picks = build(episodes, p, seed=1)
    assert len(arrays["obs"]) == 60
    assert int((arrays["source"] == 1).sum()) == round(p * 60)
    if p == 0.0:
        assert picks[mx.EXPERT] == []
    if p == 1.0:
        assert picks[mx.SIMPLE] == []


def test_flatten_never_crosses_episodes(episodes):
    arrays, _ = build(episodes, 0.5, seed=0)
    # next_obs is exactly one step after obs within the same episode (all dims encode the step)
    np.testing.assert_array_equal(arrays["next_obs"] - arrays["obs"], np.ones_like(arrays["obs"]))
    # and the episode encoded in the values matches the episode_id column
    encoded = (arrays["obs"][:, 0] % 1e6) // 1000
    np.testing.assert_array_equal(encoded, arrays["episode_id"])


def test_done_semantics_terminated_truncated_trimmed(episodes):
    arrays, picks = build(episodes, 0.5, seed=0)
    start = 0
    for source in mx.SOURCES:
        for i, k in picks[source]:
            ep = episodes[source][i]
            rows = slice(start, start + k)
            np.testing.assert_array_equal(arrays["done"][rows], ep.term[:k].astype(np.float32))
            if k < ep.length:  # trimmed: last row not terminal, next_obs real, not the next row's obs
                assert arrays["done"][start + k - 1] == 0
                if start + k < len(arrays["obs"]):
                    assert not np.array_equal(arrays["next_obs"][start + k - 1], arrays["obs"][start + k])
            if ep.trunc[-1] and k == ep.length:
                assert arrays["done"][start + k - 1] == 0  # truncation never sets done
            start += k
    assert any(k < episodes[s][i].length for s in mx.SOURCES for i, k in picks[s])  # a trim happened


def test_dtypes_and_columns(episodes):
    arrays, _ = build(episodes, 0.5, seed=0)
    for key in mx.MODEL_KEYS:
        assert arrays[key].dtype == np.float32
    assert arrays["source"].dtype == np.int8 and arrays["episode_id"].dtype == np.int32
    assert arrays["rew"].ndim == 1 and arrays["done"].ndim == 1


def test_picks_round_trip_through_columns(episodes):
    arrays, picks = build(episodes, 0.5, seed=3)
    assert mx.picks_from_columns(arrays["source"], arrays["episode_id"], arrays["t"]) == picks


def test_picks_from_columns_rejects_gaps():
    with pytest.raises(ValueError):
        mx.picks_from_columns(
            np.array([1, 1], np.int8), np.array([0, 0], np.int32), np.array([0, 2], np.int32)
        )


def test_nested_across_p_for_one_seed(episodes):
    small = mx.sample_mixture_episodes(lengths(episodes), 0.25, 60, seed=4)[mx.EXPERT]
    large = mx.sample_mixture_episodes(lengths(episodes), 0.75, 60, seed=4)[mx.EXPERT]
    assert [i for i, _ in small][:-1] == [i for i, _ in large][: len(small) - 1]  # prefix (last may trim)


def test_content_hash_deterministic_sensitive_and_container_free(episodes):
    a, _ = build(episodes, 0.5, seed=0)
    b, _ = build(episodes, 0.5, seed=0)
    c, _ = build(episodes, 0.5, seed=1)
    assert mx.content_sha256(a) == mx.content_sha256(b) != mx.content_sha256(c)
    assert mx.content_sha256(dict(reversed(list(a.items())))) == mx.content_sha256(a)  # key order
    changed = {**a, "rew": a["rew"].astype(np.float64)}
    assert mx.content_sha256(changed) != mx.content_sha256(a)  # dtype is part of the hash


def test_save_load_round_trip_and_file_hash(tmp_path, episodes):
    arrays, _ = build(episodes, 0.5, seed=0)
    sha1 = mx.save_mixture(tmp_path / "a.npz", arrays)
    sha2 = mx.save_mixture(tmp_path / "b.npz", arrays)
    assert sha1 == sha2 == hashlib.sha256((tmp_path / "a.npz").read_bytes()).hexdigest()
    loaded = mx.load_mixture(tmp_path / "a.npz")
    assert mx.content_sha256(loaded) == mx.content_sha256(arrays)


def test_mixture_name_format():
    assert mx.mixture_name(0.0, 0) == "mix_p0_s0"
    assert mx.mixture_name(0.05, 2) == "mix_p0.05_s2"
    assert mx.mixture_name(1.0, 100) == "mix_p1_s100"


def test_episode_summary(episodes):
    picks = {mx.EXPERT: [(1, 11), (2, 4)], mx.SIMPLE: [(0, 5)]}  # expert ep 2 trimmed (length 12)
    s = mx.episode_summary(episodes, picks)
    np.testing.assert_array_equal(s.returns, [11 * 2.0, 4 * 3.0, 5 * 1.0])
    np.testing.assert_array_equal(s.whole, [True, False, True])
    np.testing.assert_array_equal(s.fell, [False, False, True])  # ep1 truncated, ep2 trimmed, s0 fell
