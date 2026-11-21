"""The Phase 5 V1 follow-up rule (spec Section 8, D5 to D7) on synthetic run records."""

import math

import pytest

from offrl import train
from offrl.analysis import seed_followup as sf

C = "c" * 40
BAR = sf.V1_BAR


def run(score, **prov):
    base = {
        "git_commit": C,
        "git_dirty": False,
        "gpu": "Tesla T4",
        "device": "cuda",
        "content_sha256": "h",
        "manifest_content_sha256": "h",
        "config_matches": True,
        "resume_records": [],
    }
    return {"final": {"score": score}, "provenance": {**base, **prov}}


def runs(ours, corl):
    """{(algo, seed): run} from per-seed score lists for seeds 102, 103, 104."""
    out = {}
    for s, o, c in zip(sf.SEEDS, ours, corl, strict=True):
        out[("iql", s)], out[("corl_iql", s)] = run(o), run(c)
    return out


def test_bar_is_the_unrounded_spec_value():
    assert BAR == 105.08619608725604 - 10
    spec = (train.REPO_ROOT / "docs" / "specs" / "phase-5-iql.md").read_text(encoding="utf-8")
    assert "95.08619608725604" in spec and repr(BAR) == "95.08619608725604"


@pytest.mark.parametrize(
    ("ours", "corl", "a", "b", "action"),
    [
        ([100, 100, 100], [100, 100, 100], False, False, "proceed"),  # neither
        ([90, 90, 90], [90, 90, 90], True, False, "proceed"),  # CORL falls short, ours not below
        ([90, 90, 100], [100, 100, 100], False, True, "investigate"),  # ours below on 2 of 3
        ([80, 80, 90], [90, 90, 90], True, True, "investigate"),  # both
        ([90, 100, 100], [100, 100, 100], False, False, "proceed"),  # below on 1 of 3 only
    ],
)
def test_decision_table(ours, corl, a, b, action):
    d = sf.decide(runs(ours, corl), {}, C)
    assert (d["A"], d["B"], d["action"]) == (a, b, action)


def test_ties_are_unrounded_and_strict():
    gap5 = runs([95.0, 95.0, 100.0], [100.0, 100.0, 100.0])  # exactly 5: not below
    assert sf.decide(gap5, {}, C)["B"] is False
    just_over = runs([94.999999, 94.999999, 100.0], [100.0, 100.0, 100.0])
    assert sf.decide(just_over, {}, C)["B"] is True
    at_bar = runs([BAR] * 3, [BAR] * 3)  # exactly at the bar does not fall short
    assert sf.decide(at_bar, {}, C)["A"] is False
    # 95.09 rounds to the displayed 95.1 bar but is above the unrounded one
    assert sf.decide(runs([95.09] * 3, [95.09] * 3), {}, C)["A"] is False
    assert sf.decide(runs([95.08] * 3, [95.08] * 3), {}, C)["A"] is True


T4 = {"git_commit": C, "git_dirty": False, "gpu": "Tesla T4"}


def load_script():
    import importlib.util

    path = train.REPO_ROOT / "scripts" / "seed_followup.py"
    spec = importlib.util.spec_from_file_location("seed_followup_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        (None, "no final score"),
        (run(math.nan), "non-finite"),
        (run(100, git_commit="d" * 40), "trained at ddddddd"),
        (run(100, git_dirty=True), "dirty"),
        (run(100, gpu="NVIDIA A100-SXM4-40GB"), "GPU"),
        (run(100, gpu="NVIDIA T400"), "GPU"),  # exact name, not a substring
        (run(100, gpu=None), "GPU"),
        (run(100, device="cpu"), "trained on cpu"),  # a CPU run on a T4 node
        (run(100, config_matches=False), "config"),
        (run(100, content_sha256="x"), "mixture hash"),
        (run(100, resume_records=[{**T4, "gpu": "NVIDIA A100"}]), "GPU"),
        (run(100, resume_records=[{**T4, "git_commit": "e" * 40}]), "trained at eeeeeee"),
        (run(100, resume_records=[{**T4, "git_dirty": True}]), "dirty"),
    ],
)
def test_failed_runs_need_a_rerun(bad, why):
    first = runs([100] * 3, [100] * 3)
    first[("corl_iql", 103)] = bad
    d = sf.decide(first, {}, C)
    assert d["action"] == "rerun" and d["to_rerun"] == [("corl_iql", 103)]
    assert why in d["failures"]["corl_iql_s103"]["first"]


def test_several_failures_are_each_rerun_on_their_own():
    first = runs([100] * 3, [100] * 3)
    first[("iql", 102)] = first[("corl_iql", 104)] = None
    d = sf.decide(first, {}, C)
    assert d["action"] == "rerun" and d["to_rerun"] == [("iql", 102), ("corl_iql", 104)]
    one_back = sf.decide(first, {("iql", 102): run(100)}, C)  # the other rerun still pending
    assert one_back["action"] == "rerun" and one_back["to_rerun"] == [("corl_iql", 104)]


def test_a_finished_resume_on_t4_at_the_commit_is_complete():
    first = runs([100] * 3, [100] * 3)
    first[("iql", 102)] = run(100, resume_records=[T4])
    assert sf.decide(first, {}, C)["action"] == "proceed"


def test_a_good_rerun_is_used_and_a_second_failure_means_investigate():
    first = runs([100] * 3, [100] * 3)
    first[("iql", 104)] = None
    ok = sf.decide(first, {("iql", 104): run(90)}, C)
    assert ok["action"] == "proceed" and ok["per_seed"]["s104"]["ours"] == 90
    assert ok["per_seed"]["s104"]["from_rerun"] and not ok["per_seed"]["s102"]["from_rerun"]
    again = sf.decide(first, {("iql", 104): run(100, gpu="NVIDIA A100")}, C)
    assert again["action"] == "investigate" and "per_seed" not in again
    # the scores that exist are still reported (D7), outside the rule
    assert again["scores_reported"]["iql_s102_first"] == 100
    assert (
        again["scores_reported"]["iql_s104_rerun"] == 100 and "iql_s104_first" not in again["scores_reported"]
    )


def test_a_second_failure_wins_over_runs_still_awaiting_their_rerun():
    first = runs([100] * 3, [100] * 3)
    first[("iql", 102)] = first[("iql", 103)] = None
    d = sf.decide(first, {("iql", 102): None}, C)
    assert d["action"] == "investigate"


@pytest.mark.parametrize(
    ("ours", "rerun_dirs", "action", "code"),
    [
        ([100, 100, 100], set(), "proceed", 0),
        ([90, 90, 100], set(), "investigate", 10),
        (None, set(), "rerun", 11),  # iql s103 failed, no rerun directory yet: pending
        (None, {("iql", 103)}, "investigate", 10),  # its rerun directory exists, no final.json
    ],
)
def test_main_writes_the_report_and_exits_by_action(monkeypatch, tmp_path, ours, rerun_dirs, action, code):
    import json

    script = load_script()
    first = runs(ours or [100] * 3, [100] * 3)
    if ours is None:
        first[("iql", 103)] = None
    first[("corl_iql", 102)]["final"]["per_eval"] = [math.inf]  # non-finite extras still serialize

    def read_run(algo, seed, paths):
        is_rerun = any(p == "train.out_dir=/re" for p in paths)
        return None if is_rerun else first[(algo, seed)]

    def run_path(algo, seed, paths):
        return tmp_path if (algo, seed) in rerun_dirs else tmp_path / "absent"

    monkeypatch.setattr(script, "read_run", read_run)
    monkeypatch.setattr(script, "run_path", run_path)
    monkeypatch.setattr(script, "dirty_paths", lambda: [])
    monkeypatch.setattr(script, "collect_metadata", lambda: {})
    monkeypatch.setattr(script, "OUT", tmp_path / "out.json")
    argv = ["x", "--commit", C, "--set", "train.out_dir=/first", "--rerun-out-dir", "/re"]
    monkeypatch.setattr("sys.argv", argv)
    assert script.main() == code
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["decision"]["action"] == action
    assert report["runs"]["corl_iql_s102"]["first"]["final"]["per_eval"] == ["inf"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--commit", "abc"],
        ["--commit", C, "--set", "train.steps=5"],
        ["--commit", C, "--set", "train.out_dir=/same", "--rerun-out-dir", "/same"],
    ],
)
def test_main_refusals_are_not_outcomes(monkeypatch, argv):
    script = load_script()
    monkeypatch.setattr(script, "dirty_paths", lambda: [])
    monkeypatch.setattr("sys.argv", ["x", *argv])
    assert script.REFUSED not in script.EXIT.values()
    assert script.main() == script.REFUSED


def test_main_refuses_a_dirty_tree(monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "dirty_paths", lambda: ["x.py"])
    monkeypatch.setattr("sys.argv", ["x", "--commit", C])
    assert script.main() == script.REFUSED


def test_script_reads_a_real_tiny_run(small_run, monkeypatch):
    """scripts/seed_followup.py on a real 600-step CPU run: the fields the rule needs are read,
    the config matches with the run's own settings, the recorded GPU is the machine's, and only an
    exact Tesla T4 with training device cuda passes."""
    import gymnasium as gym
    import torch

    from test_iql_validate_script import SMALL

    make, manifests = small_run
    cfg = make("seed=102", algo="iql")
    env = gym.make("Hopper-v5")
    train.run(cfg, env=env, manifest_dir=manifests)
    env.close()
    script = load_script()
    monkeypatch.setattr(train, "MANIFEST_DIR", manifests)
    paths = [f"data.mixture_dir={cfg['data']['mixture_dir']}", f"train.out_dir={cfg['train']['out_dir']}"]

    got = script.read_run("iql", 102, [*paths, *SMALL])
    prov = got["provenance"]
    assert prov["config_matches"] is True and prov["content_sha256"] == prov["manifest_content_sha256"]
    assert math.isfinite(got["final"]["score"]) and prov["resume_records"] == []
    assert script.run_path("iql", 102, [*paths, *SMALL]).name == "iql_p1_s102"
    # meta.json records the machine's GPU, whatever train.device is: None on the laptop, the
    # node's GPU on Explorer (the test gate runs on a T4)
    assert prov["gpu"] == (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    clean = {**got, "provenance": {**prov, "git_dirty": False}}  # the laptop tree may be dirty
    assert prov["device"] == "cpu"  # where it trained, not what the machine has
    if prov["gpu"] == "Tesla T4":  # on the T4 test gate: the real CPU-on-T4 case, nothing overridden
        assert sf.failure(clean, prov["git_commit"]) == "trained on cpu, not the GPU"
    cases = [
        (None, "cuda", "GPU None, not a Tesla T4"),
        ("Tesla T4", "cpu", "trained on cpu, not the GPU"),  # this run, had it run on a T4 node
        ("Tesla T4", "cuda", None),
    ]
    for gpu, device, expected in cases:
        on = {**clean, "provenance": {**clean["provenance"], "gpu": gpu, "device": device}}
        assert sf.failure(on, prov["git_commit"]) == expected
    assert script.read_run("iql", 102, paths)["provenance"]["config_matches"] is False  # 300k asked
    assert script.read_run("corl_iql", 102, [*paths, *SMALL]) is None  # never ran
