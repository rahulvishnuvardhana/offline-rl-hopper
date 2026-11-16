import numpy as np
import pytest
import torch

from offrl.analysis import bc_sanity as bs
from offrl.data.load import Episode
from offrl.data.normalize import StateNormalizer
from offrl.nets import GaussianPolicy


def episode(i: int, n: int) -> Episode:
    obs = np.full((n + 1, 11), float(i))
    term = np.zeros(n, dtype=bool)
    return Episode(i, obs, np.full((n, 3), i / 10), np.ones(n), term, ~term)


def test_heldout_rows_are_exactly_the_unpicked_episodes():
    eps = [episode(i, n) for i, n in enumerate([3, 4, 5, 6])]
    held = bs.heldout_rows(eps, [[1, 4], [3, 2]])  # episode 3 trimmed: its tail is NOT held out
    assert int(held["n_episodes"]) == 2
    assert held["obs"].shape == (3 + 5, 11) and set(held["obs"][:, 0]) == {0.0, 2.0}
    assert len(held["obs"]) == len(held["act"])
    with pytest.raises(ValueError):
        bs.heldout_rows(eps, [[0, 3], [1, 4], [2, 5], [3, 6]])


def test_action_fit_perfect_policy_and_residual_std():
    torch.manual_seed(0)
    policy = GaussianPolicy(11, 3, (16, 16))
    norm = StateNormalizer(np.zeros(11), np.ones(11), 0.0)
    obs = np.random.default_rng(0).normal(size=(500, 11))
    exact = policy.act(torch.as_tensor(norm.transform(obs))).double().numpy()
    fit = bs.action_fit(policy, norm, obs, exact)
    assert fit["r2"] == pytest.approx(1.0) and fit["mse"] == pytest.approx(0.0, abs=1e-12)
    noisy = exact + np.random.default_rng(1).normal(0, 0.1, exact.shape)
    assert np.allclose(bs.action_fit(policy, norm, obs, noisy)["residual_std"], 0.1, atol=0.01)


def test_bands_use_fell_mean_and_dataset_best():
    summary = {
        "score": {"r_random": 0.0, "r_expert": 100.0},
        "sanity_bands": {"expert": [80.0, 107.0]},
        "datasets": {"expert": {"episodes": {"return": {"max": 113.0}}}},
    }
    assert bs.bands(summary) == {"expert": [80.0, 113.0]}


def test_no_fall_return_uses_final_steps_only():
    rows = [
        {"step": "5", "return": "1.0", "fell": "False"},
        {"step": "10", "return": "4.0", "fell": "False"},
        {"step": "10", "return": "2.0", "fell": "True"},
    ]
    assert bs.no_fall_return(rows, [10]) == {"n_episodes": 2, "n_no_fall": 1, "mean": 4.0}


def test_no_fall_match_tolerance_and_minimum_episodes():
    ok, _ = bs.no_fall_match({"n_no_fall": 42, "mean": 3934.0}, 4157.0)  # -5.4%
    assert ok
    ok, detail = bs.no_fall_match({"n_no_fall": 42, "mean": 3700.0}, 4157.0)  # -11%
    assert not ok and "-11.0%" in detail
    ok, detail = bs.no_fall_match({"n_no_fall": 9, "mean": 4157.0}, 4157.0)
    assert not ok and "cannot be evaluated" in detail
