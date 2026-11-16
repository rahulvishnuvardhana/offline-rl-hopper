"""The M1-M6 checks pass on a clean mixture and FAIL (never crash, never pass) on each corruption.

Mirrors the reviewer's corruption experiment on real data, on synthetic episodes so CI runs it.
"""

import numpy as np
import pytest

from offrl.data import mixture as mx
from offrl.data.load import Episode
from offrl.data.mixture_audit import SourceIndex, build_mixture, check_mixture, saco_cells

N, EPS = 120, 1e-3


def make_episode(ep_id: int, n: int, fell: bool, seed: int) -> Episode:
    rng = np.random.default_rng(seed * 100 + ep_id)
    term = np.zeros(n, dtype=bool)
    trunc = np.zeros(n, dtype=bool)
    (term if fell else trunc)[-1] = True
    rew = rng.uniform(0, 3, n) + (2.0 if seed == 1 else 0.0)
    return Episode(ep_id, rng.normal(size=(n + 1, 11)), rng.uniform(-1, 1, (n, 3)), rew, term, trunc)


@pytest.fixture(scope="module")
def setup():
    episodes = {
        mx.EXPERT: [make_episode(i, 15 + 3 * i, i % 2 == 0, seed=1) for i in range(8)],
        mx.SIMPLE: [make_episode(i, 8 + 2 * i, i % 3 != 0, seed=2) for i in range(12)],
    }
    index = {s: SourceIndex(episodes[s]) for s in mx.SOURCES}
    arrays, picks = build_mixture(episodes, 0.5, 0, N, EPS)
    return episodes, index, arrays, picks


def run(setup, arrays=None, picks=None):
    episodes, index, clean, clean_picks = setup
    checks = check_mixture("t", 0.5, arrays or clean, picks or clean_picks, index, episodes, N, EPS, None)
    return {c.name.split("[")[0]: c.passed for c in checks}


def corrupt(setup, **changes):
    arrays = {k: v.copy() for k, v in setup[2].items()}
    for key, fn in changes.items():
        arrays[key] = fn(arrays[key])
    return arrays


def test_clean_mixture_passes_everything(setup):
    result = run(setup)
    assert result and all(result.values()), result


def swap_rows(a):
    a = a.copy()
    a[[3, 40]] = a[[40, 3]]
    return a


@pytest.mark.parametrize(
    ("changes", "caught_by"),
    [
        ({"obs": swap_rows}, "M3_rows_equal_source"),
        ({"done": lambda d: np.where(np.arange(len(d)) == 5, 1 - d, d)}, "M3_rows_equal_source"),
        ({"next_obs": lambda x: np.roll(x, 1, axis=0)}, "M3_rows_equal_source"),
        ({"state_std": lambda s: s * 1.001}, "M6_stored_normalizer_and_scale"),
        ({"state_mean": lambda m: m + 1.0}, "M6_stored_normalizer_and_scale"),
        ({"reward_scale": lambda r: r * 1.5}, "M6_stored_normalizer_and_scale"),
        ({"state_std_eps": lambda e: e * 10}, "M6_stored_normalizer_and_scale"),
        ({k: swap_rows for k in (*mx.MODEL_KEYS, *mx.COLUMN_KEYS)}, "M5_picks_and_phase2"),
        (
            {"source": lambda s: np.where(np.arange(len(s)) == 0, 2, s).astype(np.int8)},
            "M3_rows_equal_source",
        ),
        (
            {"episode_id": lambda e: np.where(np.arange(len(e)) == 0, 999, e).astype(np.int32)},
            "M3_rows_equal_source",
        ),
    ],
)
def test_each_corruption_fails_without_crashing(setup, changes, caught_by):
    result = run(setup, arrays=corrupt(setup, **changes))
    assert result[caught_by] is False, result


def test_wrong_picks_fail_m5(setup):
    picks = {s: list(v) for s, v in setup[3].items()}
    picks[mx.SIMPLE] = picks[mx.SIMPLE][::-1]
    assert run(setup, picks=picks)["M5_picks_and_phase2"] is False


def test_phase2_reference_compared(setup):
    episodes, index, arrays, picks = setup
    summ = mx.episode_summary(episodes, picks)
    good = {"mean_return": float(summ.returns.mean()), "reward_scale": float(arrays["reward_scale"])}
    for ref, expected in ((good, True), ({**good, "mean_return": good["mean_return"] + 1e-6}, False)):
        checks = check_mixture("t", 0.5, arrays, picks, index, episodes, N, EPS, ref)
        assert {c.name.split("[")[0]: c.passed for c in checks}["M5_picks_and_phase2"] is expected


def test_saco_cells_counts_unique_cells():
    arrays = {"obs": np.zeros((4, 11)), "act": np.array([[0, 0, 0], [0, 0, 0], [5, 0, 0], [9, 0, 0]], float)}
    assert saco_cells(arrays, np.zeros(14), np.ones(14), 1.0) == 3
