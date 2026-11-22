"""Phase 6 spec Section 6 and S3, S8, S9, S11 (shared parts): ``train.run_status`` on tiny real
runs, ``train.run`` agreeing with it on every case, ``preflight_config``, atomic ``write_json``
and the checkout check."""

import json
import os

import gymnasium as gym
import pytest

from offrl import train


@pytest.fixture(scope="module")
def env():
    e = gym.make("Hopper-v5")
    yield e
    e.close()


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    """Tests run on a developer's working copy, so report it clean (as test_train.py does)."""
    real = train.collect_metadata
    monkeypatch.setattr(train, "collect_metadata", lambda: {**real(), "git_dirty": False})


def status(cfg, device="cpu", threads=None, **now):
    meta_now = {**train.collect_metadata(), **now}
    return train.run_status(cfg, device, threads or train.torch.get_num_threads(), meta_now)


def edit_json(path, **changes):
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **changes}), encoding="utf-8")


def test_status_follows_a_run_through_its_life(small_run, env):
    make, manifests = small_run
    cfg = make()
    assert status(cfg) == ("new", "")
    run_dir = train.run(cfg, stop_at=100, env=env, manifest_dir=manifests)  # first checkpoint: 200
    assert status(cfg)[0] == "started"
    train.run(cfg, stop_at=200, env=env, manifest_dir=manifests)
    assert status(cfg) == ("partial", "")
    train.run(cfg, env=env, manifest_dir=manifests)
    assert status(cfg) == ("done", "")
    assert status(cfg, device="cuda", threads=1) == ("done", "")  # a finished run on any setup
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["repo_root"] == str(
        train.REPO_ROOT
    )


def finished(make, manifests, env):
    cfg = make()
    return cfg, train.run(cfg, env=env, manifest_dir=manifests)


def interrupted(make, manifests, env):
    cfg = make()
    return cfg, train.run(cfg, stop_at=200, env=env, manifest_dir=manifests)


@pytest.mark.parametrize(
    ("build", "spoil", "match"),
    [
        (finished, lambda d: edit_json(d / "meta.json", git_commit="deadbeef"), "code has changed"),
        (interrupted, lambda d: edit_json(d / "meta.json", git_commit="deadbeef"), "code has changed"),
        (finished, lambda d: edit_json(d / "meta.json", git_dirty=True), "dirty tree"),
        (interrupted, lambda d: edit_json(d / "meta.json", git_dirty=True), "dirty tree"),
        (finished, lambda d: (d / "final.json").write_text('{"run": "trunc', encoding="utf-8"), "final.json"),
        (finished, lambda d: edit_json(d / "final.json", run="another_run"), "final.json"),
        (finished, lambda d: (d / "meta.json").unlink(), "meta.json"),
        (finished, lambda d: edit_json(d / "final.json", steps=[1, 2]), "final.json"),
        (finished, lambda d: (d / "config.yaml").unlink(), "config.yaml"),
        (interrupted, lambda d: (d / "config.yaml").unlink(), "config.yaml"),
        (interrupted, lambda d: edit_json(d / "meta.json", torch_threads=999), "same setup"),
        (interrupted, lambda d: edit_json(d / "meta.json", device="cuda"), "same setup"),
    ],
)
def test_conflicts_are_refused_by_status_and_by_run(small_run, env, build, spoil, match):
    make, manifests = small_run
    cfg, run_dir = build(make, manifests, env)
    spoil(run_dir)
    got, reason = status(cfg)
    assert got == "conflict" and match in reason
    with pytest.raises(train.PreflightError, match=match):
        train.run(cfg, env=env, manifest_dir=manifests)


def test_other_config_is_a_conflict_in_every_state(small_run, env):
    make, manifests = small_run
    cfg = make()
    for stop in (100, 200, None):  # started, partial, done
        train.run(cfg, stop_at=stop, env=env, manifest_dir=manifests)
        got, reason = status(make("algo.beta=0.5"))
        assert got == "conflict" and "different config" in reason


@pytest.mark.parametrize("now", [{"git_dirty": True}, {"git_dirty": None}, {"git_commit": None}])
def test_resume_needs_a_known_clean_tree(small_run, env, now):
    make, manifests = small_run
    cfg, _ = interrupted(make, manifests, env)
    assert status(cfg, **now)[0] == "conflict"


def test_preflight_config_passes_and_refuses_log_every(small_run):
    """S9: log_every 80 passes every step check (600 % 100, 600 % 200) and fails only its own."""
    make, manifests = small_run
    assert train.preflight_config(make(), manifests) == []
    errors = train.preflight_config(make("train.log_every=80"), manifests)
    assert errors == ["checkpoint_every and eval_every must be multiples of log_every"]


def test_preflight_config_needs_no_arrays(small_run):
    make, manifests = small_run
    cfg = make("seed=0")  # a final-grid mixture with development evaluation seeds
    assert any("final evaluation seeds" in e for e in train.preflight_config(cfg, manifests))


def test_write_json_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "final.json"
    train.write_json(path, {"a": 1})

    def crash(*_):
        raise OSError("killed mid-write")

    monkeypatch.setattr(train.os, "replace", crash)
    with pytest.raises(OSError):
        train.write_json(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}  # the old file, whole
    monkeypatch.undo()
    train.write_json(path, {"a": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 3}
    assert sorted(os.listdir(tmp_path)) == ["final.json"]


def test_run_refuses_code_from_another_checkout(small_run, env, tmp_path):
    make, manifests = small_run
    cfg = make()
    out = train.resolve_path(cfg["train"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps({"repo_root": str(tmp_path / "elsewhere")}), encoding="utf-8")
    with pytest.raises(train.PreflightError, match="belongs to checkout"):
        train.run(cfg, env=env, manifest_dir=manifests)
    for broken in ('{"repo_root": "trunc', "[]", "{}"):  # truncated, not an object, no repo_root
        (out / "sweep.json").write_text(broken, encoding="utf-8")
        with pytest.raises(train.PreflightError, match="unreadable or without repo_root"):
            train.run(cfg, env=env, manifest_dir=manifests)
    (out / "sweep.json").write_text(json.dumps({"repo_root": str(train.REPO_ROOT)}), encoding="utf-8")
    assert (train.run(cfg, env=env, manifest_dir=manifests) / "final.json").exists()


def test_started_run_restarts_with_cleared_logs(small_run, env):
    make, manifests = small_run
    cfg = make()
    run_dir = train.run(cfg, stop_at=100, env=env, manifest_dir=manifests)  # before the first checkpoint
    assert status(cfg)[0] == "started"
    rows_before = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    train.run(cfg, stop_at=150, env=env, manifest_dir=manifests)
    rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert rows[: len(rows_before)] == rows_before and len(rows) == len(rows_before) + 1  # 50, 100, 150
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["resumes"] == []


def test_non_finite_final_score_still_writes_final_json(small_run, env, monkeypatch):
    """A run that finishes is a result whatever its score: NaN is written as a string, the run is
    done, and the sweep's --check flags it (Phase 6 spec Section 2, run outcomes)."""
    make, manifests = small_run
    cfg = make()
    real = train.evaluate

    def nan_scores(*args, **kwargs):
        summary, rows = real(*args, **kwargs)
        return {**summary, "score_mean": float("nan")}, rows

    monkeypatch.setattr(train, "evaluate", nan_scores)
    run_dir = train.run(cfg, env=env, manifest_dir=manifests)
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    assert final["score"] == "nan" and final["per_eval"][0]["score_mean"] == "nan"
    assert status(cfg) == ("done", "")
