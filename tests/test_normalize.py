import numpy as np
import pytest

from offrl.data import mixture as mx
from offrl.data.load import Episode
from offrl.data.normalize import StateNormalizer, beta_eff, fit_state_normalizer, reward_scale


def make_episode(ep_id: int, obs: np.ndarray) -> Episode:
    n = len(obs) - 1
    term = np.zeros(n, dtype=bool)
    term[-1] = True
    return Episode(ep_id, obs, np.zeros((n, 3)), np.ones(n), term, ~term)


def test_fit_uses_picked_rows_only_float64_ddof0():
    rng = np.random.default_rng(0)
    e0 = make_episode(0, rng.normal(size=(5, 11)))
    e1 = make_episode(1, rng.normal(size=(7, 11)) + 1e6)  # never picked: must not move the fit
    trimmed_tail = e0.obs.copy()
    episodes = {mx.EXPERT: [e0, e1], mx.SIMPLE: [make_episode(0, rng.normal(size=(4, 11)))]}
    picks = {mx.EXPERT: [(0, 3)], mx.SIMPLE: [(0, 3)]}  # e0 trimmed to 3 of 4 steps
    norm = fit_state_normalizer(episodes, picks)
    rows = np.concatenate([trimmed_tail[:3], episodes[mx.SIMPLE][0].obs[:3]])
    np.testing.assert_array_equal(norm.mean, rows.mean(0))
    np.testing.assert_array_equal(norm.std, rows.std(0))  # ddof = 0
    assert norm.mean.dtype == np.float64 and norm.std.dtype == np.float64


def test_transform_applies_eps_not_stored():
    norm = StateNormalizer(mean=np.zeros(2), std=np.array([0.0, 1.0]), eps=1e-3)
    out = norm.transform(np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(out, [[1000.0, 1.0 / 1.001]], rtol=1e-6)
    assert out.dtype == np.float32
    assert norm.std[0] == 0.0  # eps lives in the transform only


def test_reward_scale_ignores_trimmed():
    returns = np.array([100.0, 500.0, 3.0])
    whole = np.array([True, True, False])  # a 3-return tail must not set the minimum
    assert reward_scale(returns, whole) == pytest.approx(1000.0 / 400.0)
    with pytest.raises(ValueError):
        reward_scale(np.array([5.0, 9.0]), np.array([True, False]))
    with pytest.raises(ValueError):  # zero return range: no division by zero
        reward_scale(np.array([7.0, 7.0]), np.array([True, True]))


def test_beta_eff_reference():
    assert beta_eff(3.0, 0.2512) == pytest.approx(3.0)
    assert beta_eff(3.0, 0.368) == pytest.approx(4.39, abs=0.01)  # the ADR-9 p = 0 caveat
