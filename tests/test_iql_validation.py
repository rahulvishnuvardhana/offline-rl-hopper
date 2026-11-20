"""The Phase 5 validation checks (spec Section 8) on synthetic episodes, mixtures and runs."""

import numpy as np
import pytest

from offrl.analysis import iql_validation as iv
from offrl.data.load import Episode

COMMIT = "c" * 40


def episode(i, rew, fell):
    n = len(rew)
    term = np.zeros(n, dtype=bool)
    term[-1] = fell
    return Episode(i, np.zeros((n + 1, 11)), np.zeros((n, 3)), np.asarray(rew, dtype=float), term, ~term)


def test_reward_scale_and_terminals_from_raw_episodes():
    eps = {
        "expert": [episode(0, [1.0] * 10, False), episode(1, [2.0] * 10, True)],
        "simple": [episode(0, [0.5] * 4, True)],
    }
    # expert 1 trimmed to 5 steps: its return (10) and its fall must not count
    picks = {"expert": [[0, 10], [1, 5]], "simple": [[0, 4]]}
    assert iv.picked_returns(eps, picks).tolist() == [10.0, 2.0]
    assert iv.reward_scale_from_episodes(eps, picks) == pytest.approx(1000.0 / 8.0)
    assert iv.terminated_count(eps, picks) == 1
    assert iv.timeout_mixups(eps, picks) == 0


def test_timeout_mixups_catch_timeouts_marked_terminal():
    n = iv.EPISODE_STEPS
    timeout_as_fall = episode(0, [1.0] * n, True)  # term at the time limit
    both = episode(1, [1.0] * 5, True)
    both.trunc[-1] = True  # term and trunc on the same step
    eps = {"expert": [timeout_as_fall, both, episode(2, [1.0] * n, False), episode(3, [1.0] * 5, True)]}
    assert iv.timeout_mixups(eps, {"expert": [[0, n], [1, 5], [2, n], [3, 5]]}) == 2
    assert iv.timeout_mixups(eps, {"expert": [[0, 10]]}) == 0  # trimmed: its end is not the episode's


def test_value_bounds_include_zero():
    assert iv.value_bounds(np.array([0.5, 2.0]), 0.99) == pytest.approx((0.0, 200.0))
    assert iv.value_bounds(np.array([-1.0, -0.5]), 0.5) == pytest.approx((-2.0, 0.0))


def test_initial_states_and_discounted_returns():
    obs = np.arange(5, dtype=float)[:, None]
    rew = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    t = np.array([0, 1, 2, 0, 1])
    s0, ret = iv.initial_states_and_returns(obs, rew, t, scale=2.0, gamma=0.5)
    assert s0[:, 0].tolist() == [0.0, 3.0]
    assert ret.tolist() == pytest.approx([2 * (1 + 0.5 + 0.25), 2 * (2 + 1)])
    with pytest.raises(ValueError):
        iv.initial_states_and_returns(obs[1:], rew[1:], t[1:], 1.0, 0.5)


def rows(values, start=40_000, step=10_000, key="q_mean"):
    return [{"step": str(start + i * step), key: str(v)} for i, v in enumerate(values)]


def test_bounded_skips_early_rows_and_catches_nan_and_escape():
    ok, _ = iv.bounded(rows([1e9, 5.0, 6.0]), ["q_mean"], 0.0, 10.0)  # 40k is before 50k: ignored
    assert ok
    ok, detail = iv.bounded(rows([5.0, float("nan")], start=60_000), ["q_mean"], 0.0, 10.0)
    assert not ok and "first bad" in detail
    assert "first bad" in iv.bounded(rows([5.0, 11.0], start=60_000), ["q_mean"], 0.0, 10.0)[1]
    assert not iv.bounded(rows([5.0], start=10_000), ["q_mean"], 0.0, 10.0)[0]  # nothing after 50k
    assert iv.bounded(rows([1e9, 5.0], start=50_000), ["q_mean"], 0.0, 10.0)[0]  # 50k itself excluded


def test_q_drift_and_v0_ratio():
    assert iv.q_drift(rows([100.0, 109.0], start=200_000, step=100_000))[0]
    assert not iv.q_drift(rows([100.0, 111.0], start=200_000, step=100_000))[0]
    assert not iv.q_drift(rows([100.0], start=200_000))[0]
    assert iv.v0_ratio(80.0, 100.0)[0] and iv.v0_ratio(150.0, 100.0)[0]
    assert not iv.v0_ratio(79.0, 100.0)[0] and not iv.v0_ratio(151.0, 100.0)[0]


def test_drift_and_ratio_refuse_non_positive_references():
    # Unguarded, a = 0 divides by zero and a = -5 passes (drift 0%, ratio 1.0).
    for a in (0.0, -5.0):
        ok, detail = iv.q_drift(rows([a, a], start=200_000, step=100_000))
        assert not ok and "undefined" in detail
        ok, detail = iv.v0_ratio(a, a)
        assert not ok and "undefined" in detail


def run(algo, p, seed, score=90.0, **extra):
    r = {
        "algo": algo,
        "p": p,
        "seed": seed,
        "run": f"{algo}_p{p:g}_s{seed}",
        "mixture": f"mix_p{p:g}_s{seed}",
        "final": {"score": score},
        "reward_scale": 0.25,
        "provenance": {
            "git_commit": COMMIT,
            "git_dirty": False,
            "content_sha256": "h",
            "manifest_content_sha256": "h",
            "config_matches": True,
            "resume_records": [],
        },
    }
    if algo == "iql":
        r["v3"] = {"bounded": [True, ""], "drift": [True, ""]}
        if p == 1.0:
            r["v3"]["v_s0"] = [True, ""]
            r["no_fall_return"] = {"ok": True}
    return {**r, **extra}


def grid(**scores):
    cells = [("iql", 1.0), ("iql", 0.2), ("iql", 0.0), ("corl_iql", 1.0), ("corl_iql", 0.2)]
    cells += [("bc", 1.0), ("bc", 0.2)]
    return [run(a, p, s, score=scores.get(f"{a}_{p:g}", 90.0)) for a, p in cells for s in (100, 101)]


MIXTURES = {
    f"mix_p{p:g}_s{s}": {
        "reward_scale_recomputed": 0.25,
        "done_rows": 7,
        "terminated_episodes": 7,
        "content_sha256": "m",
        "content_sha256_file": "m",
        "timeout_mixups": 0,
    }
    for p in (1.0, 0.2, 0.0)
    for s in (100, 101)
}


def nf(no_fall):
    return no_fall["ok"], "no-fall"


def failed(checks):
    return {c["name"] for c in checks if not c["passed"]}


def test_all_pass_gives_38_checks():
    checks = iv.assemble_checks(grid(), MIXTURES, nf)
    assert len(checks) == 14 + 6 + 1 + 1 + 6 * 2 + 2 + 2
    assert not failed(checks)


def test_v1_is_relative_to_bc_and_v2_to_corl():
    checks = iv.assemble_checks(grid(iql_1=79.0, bc_1=90.0), MIXTURES, nf)
    assert failed(checks) == {"V1_iql_vs_bc_p1"}
    assert not failed(iv.assemble_checks(grid(iql_1=80.0, bc_1=90.0), MIXTURES, nf))
    checks = iv.assemble_checks(grid(**{"iql_0.2": 60.0, "corl_iql_0.2": 71.0}), MIXTURES, nf)
    assert failed(checks) == {"V2_ours_vs_corl_p0.2"}
    assert not failed(iv.assemble_checks(grid(**{"iql_0.2": 60.0, "corl_iql_0.2": 70.0}), MIXTURES, nf))
    checks = iv.assemble_checks(grid(**{"iql_0.2": 60.0, "corl_iql_0.2": 70.0001}), MIXTURES, nf)
    assert failed(checks) == {"V2_ours_vs_corl_p0.2"}


def test_v1_and_v2_need_exactly_the_validation_seeds():
    runs = [r for r in grid() if r["run"] != "bc_p1_s101"]  # a missing BC seed
    assert "V1_iql_vs_bc_p1" in failed(iv.assemble_checks(runs, MIXTURES, nf))
    runs = [*grid(), run("iql", 1.0, 102, score=0.0)]  # an extra seed would move the mean
    assert "V1_iql_vs_bc_p1" in failed(iv.assemble_checks(runs, MIXTURES, nf))
    runs = [r for r in grid() if r["algo"] != "corl_iql"]  # an empty cell
    assert "V2_ours_vs_corl_p0.2" in failed(iv.assemble_checks(runs, MIXTURES, nf))
    runs = [r for r in grid() if r["run"] != "corl_iql_p0.2_s101"]  # one seed: its mean still agrees
    assert "V2_ours_vs_corl_p0.2" in failed(iv.assemble_checks(runs, MIXTURES, nf))


def test_v0_needs_one_training_commit_and_the_validation_config():
    runs = grid()
    runs[5]["provenance"]["git_commit"] = "d" * 40
    names = {c["name"] for c in iv.assemble_checks(runs, MIXTURES, nf) if c["name"].startswith("V0_prov")}
    assert names == failed(iv.assemble_checks(runs, MIXTURES, nf))  # every run, not only runs[5]
    assert iv.training_commit(grid()) == COMMIT and iv.training_commit(runs) is None
    runs = grid()
    runs[2]["provenance"]["config_matches"] = False  # e.g. a 100k run read as a 300k one
    assert failed(iv.assemble_checks(runs, MIXTURES, nf)) == {"V0_provenance[iql_p0.2_s100]"}


def test_v0_catches_dirty_resume_scale_and_terminal_mismatch():
    runs = grid()
    runs[0]["provenance"]["resume_records"] = [{"git_commit": COMMIT, "git_dirty": True}]
    runs[3]["reward_scale"] = 0.25 * (1 + 1e-9)
    mixtures = {**MIXTURES, "mix_p0_s100": {**MIXTURES["mix_p0_s100"], "done_rows": 8}}
    mixtures["mix_p1_s101"] = {**MIXTURES["mix_p1_s101"], "content_sha256_file": "x"}
    mixtures["mix_p0_s101"] = {**MIXTURES["mix_p0_s101"], "timeout_mixups": 1}
    assert failed(iv.assemble_checks(runs, mixtures, nf)) == {
        "V0_provenance[iql_p1_s100]",
        "V0_data[mix_p0.2_s101]",
        "V0_data[mix_p0_s100]",
        "V0_data[mix_p1_s101]",
        "V0_data[mix_p0_s101]",
    }


def test_v3_and_v4_are_reported_per_iql_run():
    runs = grid()
    runs[2]["v3"]["drift"] = [False, "drift"]  # iql p0.2 s100
    runs[1]["no_fall_return"] = {"ok": False}  # iql p1 s101
    assert failed(iv.assemble_checks(runs, MIXTURES, nf)) == {
        "V3_drift[iql_p0.2_s100]",
        "V4_no_fall[iql_p1_s101]",
    }
