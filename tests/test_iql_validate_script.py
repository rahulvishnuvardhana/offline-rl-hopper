"""scripts/iql_validate.py on a real tiny IQL run: analysis and figure run end to end."""

import importlib.util
import json
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from offrl import train
from offrl.algos.iql import IQLTrainer
from offrl.data import mixture as mx
from offrl.data.load import Episode
from offrl.data.normalize import reward_scale
from test_iql_validation import MIXTURES, grid

SCRIPT = train.REPO_ROOT / "scripts" / "iql_validate.py"
SMALL = [  # tests/conftest.py small_run overrides, minus the paths
    "train.steps=600",
    "train.batch_size=32",
    "train.log_every=50",
    "train.checkpoint_every=200",
    "train.device=cpu",
    "train.eval_every=100",
    "eval.episodes=2",
    "eval.final_episodes=3",
    "eval.final_score_last_n=2",
    "net.hidden=[32, 32]",
]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("iql_validate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_and_figure_on_a_tiny_run(small_run, script, tmp_path):
    make, manifests = small_run
    cfg = make(algo="iql")
    env = gym.make("Hopper-v5")
    run_dir = train.run(cfg, env=env, manifest_dir=manifests)
    env.close()
    paths = [f"data.mixture_dir={cfg['data']['mixture_dir']}", f"train.out_dir={cfg['train']['out_dir']}"]
    script.EXTRA.extend(paths)
    name = mx.mixture_name(1.0, 100)
    arrays = mx.load_mixture(train.resolve_path(cfg["data"]["mixture_dir"]) / f"{name}.npz")
    manifest = json.loads((manifests / f"{name}.json").read_text(encoding="utf-8"))
    mix = {"scaled_reward_min": 0.0, "scaled_reward_max": 1.0}

    # Paths only: the script asks for the 300k validation config, and this run is 600 steps.
    out = script.analyze("iql", 1.0, 100, mix, arrays, manifest)
    assert out["provenance"]["config_matches"] is False and out["provenance"]["steps"] == 600
    script.EXTRA.extend(SMALL)
    out = script.analyze("iql", 1.0, 100, mix, arrays, manifest)
    assert out["provenance"]["config_matches"] is True

    assert out["run"] == run_dir.name and out["provenance"]["content_sha256"] == manifest["content_sha256"]
    assert set(out["v3"]) == {"bounded", "drift", "v_s0"}
    assert out["v3"]["drift"][0] is False  # a 600-step run has no 200k and 300k rows
    assert "V(s0)" in out["v3"]["v_s0"][1] and out["no_fall_return"]["n_episodes"] == 6
    json.dumps(out, allow_nan=False)  # the report must serialize

    results = [{**out, "algo": a, "p": p} for a in ("iql", "corl_iql") for p in (1.0, 0.2)]
    script.save(script.figure(results), tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_v_at_initial_states_reads_v_from_the_checkpoint(small_run, script, make_mixture, tmp_path):
    make, _ = small_run
    cfg = make(algo="iql")
    arrays = make_mixture(seed=100)  # 20 episodes of 100 rows, reward_scale 0.25
    trainer = IQLTrainer(cfg, 11, 3, torch.device("cpu"))
    with torch.no_grad():  # V(s) = 2.5 everywhere
        trainer.v.v[-1].weight.zero_()
        trainer.v.v[-1].bias.fill_(2.5)
    ckpt = tmp_path / "run" / "ckpt"
    ckpt.mkdir(parents=True)
    torch.save({"trainer": trainer.state_dict()}, ckpt / "latest.pt")

    v, ret = script.v_at_initial_states(cfg, tmp_path / "run", arrays)

    g = cfg["algo"]["discount"]
    expected = [sum(float(arrays["rew"][100 * e + t]) * 0.25 * g**t for t in range(100)) for e in range(20)]
    assert v == pytest.approx(2.5, abs=1e-6)
    assert ret == pytest.approx(np.mean(expected), rel=1e-9)


def test_mixture_facts_on_a_flattened_mixture(script):
    """The builder's reward scale (``normalize.reward_scale``) and done column agree with V0's
    independent recomputation from the source episodes."""
    rng = np.random.default_rng(0)

    def ep(i: int, n: int, fell: bool) -> Episode:
        term = np.zeros(n, dtype=bool)
        term[-1] = fell
        trunc = np.zeros(n, dtype=bool)
        trunc[-1] = not fell
        obs, act = rng.normal(size=(n + 1, 11)), rng.normal(size=(n, 3))
        return Episode(i, obs, act, rng.uniform(0, 4, n), term, trunc)

    episodes = {mx.EXPERT: [ep(0, 30, False), ep(1, 20, True), ep(2, 25, True)], mx.SIMPLE: [ep(0, 10, True)]}
    picks = {mx.EXPERT: [(0, 30), (1, 20), (2, 7)], mx.SIMPLE: [(0, 10)]}  # expert 2 trimmed
    arrays = mx.flatten(episodes, picks)
    returns = [episodes[s][i].rew[:n].sum() for s in mx.SOURCES for i, n in picks[s]]
    whole = [n == episodes[s][i].length for s in mx.SOURCES for i, n in picks[s]]
    arrays["reward_scale"] = np.array(reward_scale(np.array(returns), np.array(whole)))
    manifest = {"picks": picks, "content_sha256": mx.content_sha256(arrays)}

    facts = script.mixture_facts("mix", arrays, manifest, episodes)

    assert facts["reward_scale_recomputed"] == pytest.approx(facts["reward_scale_file"], rel=1e-12)
    assert facts["done_rows"] == facts["terminated_episodes"] == 2  # the trimmed fall is not a fall
    assert facts["timeout_mixups"] == 0 and facts["content_sha256_file"] == facts["content_sha256"]


def test_main_wires_runs_checks_and_report(script, monkeypatch, tmp_path):
    """main() with the data, runs and figure stubbed: every cell is analyzed once, the checks are
    assembled on the training commit, and the report is written and exits 0."""
    runs = {(r["algo"], r["p"], r["seed"]): r for r in grid()}
    for r in runs.values():  # fields only the reported (not gated) section reads
        r["final"]["fall_rate"], r["beta_eff"] = 0.0, 3.0
    eda = {
        "datasets": {mx.EXPERT: {"episodes": {"no_fall": {"return": {"mean": 1.0}}}}},
        "dataset_ids": {mx.EXPERT: "e", mx.SIMPLE: "s"},
    }
    (tmp_path / "eda.json").write_text(json.dumps(eda), encoding="utf-8")
    monkeypatch.setattr(script, "EDA_SUMMARY", tmp_path / "eda.json")
    monkeypatch.setattr(script, "OUT", tmp_path / "out")
    monkeypatch.setattr(script, "FIGURE", script.ROOT / "report" / "figures" / "unused.png")
    monkeypatch.setattr(script, "dirty_paths", lambda: [])
    monkeypatch.setattr(script, "collect_metadata", lambda: {"git_commit": "a" * 40})
    monkeypatch.setattr(script, "load_episodes", lambda _id: [])
    monkeypatch.setattr(script.mx, "load_mixture", lambda _path: {})
    monkeypatch.setattr(script, "mixture_facts", lambda name, *_: MIXTURES[name])
    monkeypatch.setattr(script, "analyze", lambda algo, p, seed, *_: runs.pop((algo, p, seed)))
    monkeypatch.setattr(script.bs, "no_fall_match", lambda no_fall, _ref: (no_fall["ok"], "stub"))
    monkeypatch.setattr(script, "save", lambda fig, path: None)
    monkeypatch.setattr(script, "figure", lambda results: None)
    monkeypatch.setattr(script, "vendored_sha256", lambda commit: f"sha-at-{commit[:7]}")
    monkeypatch.setattr(sys, "argv", ["iql_validate.py"])

    assert script.main() == 0
    assert not runs  # each of the 14 runs analyzed exactly once
    report = json.loads((tmp_path / "out" / "iql_validation.json").read_text(encoding="utf-8"))
    assert len(report["checks"]) == 38 and report["all_passed"]
    assert report["training_commit"] == "c" * 40 and report["run"]["git_commit"] == "a" * 40
    assert report["vendored_corl_sha256"] == "sha-at-ccccccc"
    assert report["reported_not_gated"]["ours_vs_corl_p1"]["s100"] == {"ours": 90.0, "corl": 90.0}
