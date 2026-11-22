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


COMMIT = "a" * 40
NO_FALL_REF = {"expert": 4157.0, "simple": 3179.0}


def result(p, seed, score=100.0, in_band=True, no_fall=None, r2=0.9, resumes=(), dirty=False, run=None):
    ref = NO_FALL_REF["expert" if p == 1.0 else "simple"]
    return {
        "run": run or f"bc_p{p:g}_s{seed}",
        "p": p,
        "seed": seed,
        "final": {"score": score},
        "band": [80.0, 113.0],
        "in_band": in_band,
        "no_fall_return": {"n_no_fall": 50, "mean": ref if no_fall is None else no_fall},
        "heldout_fit": {"r2": r2},
        "provenance": {
            "git_commit": COMMIT,
            "git_dirty": dirty,
            "content_sha256": "h",
            "manifest_content_sha256": "h",
            "resume_records": list(resumes),
        },
    }


def clean(**extra):
    return {"git_commit": COMMIT, "git_dirty": False, **extra}


def clean_demo(**extra):
    return {"git_commit": COMMIT, "git_dirty": False, "resumes": [clean(from_step=25)], **extra}


def grid(overrides=None):
    overrides = overrides or {}
    out = []
    for s in (100, 101):
        for p in (0.0, 1.0):
            out.append(result(p, s, **overrides.get((p, s), {})))
    return out


def names(checks, passed):
    return {c["name"] for c in checks if c["passed"] is passed}


def test_assemble_checks_all_pass_gives_the_15_dod_checks():
    resume = {"metrics.csv": True, "eval.csv": True, "eval_episodes.csv": True}
    checks, p1_vs_p0 = bs.assemble_checks(grid(), resume, clean_demo(), COMMIT, NO_FALL_REF, [100, 101])
    assert len(checks) == 15 and all(c["passed"] for c in checks)
    assert p1_vs_p0 == {"s100": {"p1": 100.0, "p0": 100.0}, "s101": {"p1": 100.0, "p0": 100.0}}


def test_assemble_checks_judges_a_rerun_after_its_kept_miss():
    results = grid({(1.0, 100): {"score": 70.0, "in_band": False, "no_fall": 1000.0, "r2": 0.1}})
    rerun = result(1.0, 100, score=104.0, run="bc_p1_s100_300k")
    results.insert(2, rerun)  # D4: the rerun follows its miss in training order
    checks, p1_vs_p0 = bs.assemble_checks(results, {"m": True}, clean_demo(), COMMIT, NO_FALL_REF, [100, 101])
    assert all(c["passed"] for c in checks)  # band: any run in band; fits judged on the rerun
    band = next(c for c in checks if c["name"] == "band[p1_s100]")
    assert "MISS (kept, D4)" in band["detail"] and "_300k" in band["detail"]
    assert p1_vs_p0["s100"]["p1"] == 104.0


def test_assemble_checks_r5_reads_every_resume_record():
    dirty_resume = [clean(from_step=25, git_dirty=True)]
    results = grid({(0.0, 101): {"resumes": dirty_resume}, (1.0, 101): {"dirty": True}})
    checks, _ = bs.assemble_checks(results, {"m": True}, clean_demo(), COMMIT, NO_FALL_REF, [100, 101])
    assert names(checks, False) == {"R5_provenance[p0_s101]", "R5_provenance[p1_s101]"}


@pytest.mark.parametrize(
    "demo",
    [
        clean_demo(git_commit="b" * 40),  # a finished demo from an earlier job
        clean_demo(resumes=[clean(from_step=25, git_commit="b" * 40)]),
        clean_demo(resumes=[]),  # restarted, never resumed
        clean_demo(resumes=[clean(from_step=25), clean(from_step=50)]),
    ],
)
def test_assemble_checks_resume_demo_must_be_one_resume_at_this_commit(demo):
    checks, _ = bs.assemble_checks(grid(), {"m": True}, demo, COMMIT, NO_FALL_REF, [100, 101])
    assert names(checks, False) == {"resume_bitwise"}


def test_assemble_checks_resume_mismatch_fails():
    checks, _ = bs.assemble_checks(grid(), {"m": False}, clean_demo(), COMMIT, NO_FALL_REF, [100, 101])
    assert names(checks, False) == {"resume_bitwise"}
