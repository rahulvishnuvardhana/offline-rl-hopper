"""R1 to R4 (spec Section 7) on a synthetic development mixture with the real Hopper-v5 env."""

import json
import time

import gymnasium as gym
import pytest
import torch

from offrl import train
from offrl.algos.bc import make_policy
from offrl.config import unknown_keys
from offrl.data.buffer import ReplayBuffer
from offrl.nets import TwinQ, ValueNet
from offrl.runlog import truncate_csv
from offrl.seeding import derive_seeds, seed_everything

COMPARED = ("metrics.csv", "eval.csv", "eval_episodes.csv")


@pytest.fixture(scope="module")
def env():
    e = gym.make("Hopper-v5")
    yield e
    e.close()


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    """Resume refuses a dirty tree; tests run on a developer's working copy, so report it clean."""
    real = train.collect_metadata
    monkeypatch.setattr(train, "collect_metadata", lambda: {**real(), "git_dirty": False})


def read(run_dir, name):
    return (run_dir / name).read_bytes()


def test_r1_kill_and_resume_is_bitwise_identical(small_run, env):
    make, manifests = small_run
    full = train.run(make(out="a"), env=env, manifest_dir=manifests)
    cfg = make(out="b")
    part = train.run(cfg, stop_at=350, env=env, manifest_dir=manifests)  # last checkpoint: 200
    assert not (part / "final.json").exists()
    assert any(line.startswith(b"300,") for line in read(part, "eval.csv").splitlines())  # to discard
    resumed = train.run(cfg, env=env, manifest_dir=manifests)
    for name in COMPARED:
        assert read(full, name) == read(resumed, name), name
    a = torch.load(full / "ckpt" / "latest.pt", weights_only=True)
    b = torch.load(resumed / "ckpt" / "latest.pt", weights_only=True)  # R4: safe loading works
    for key in ("policy", "opt", "sched"):
        assert repr(a["trainer"][key]) == repr(b["trainer"][key]), key
    assert json.loads(read(full, "final.json")) == json.loads(read(resumed, "final.json"))
    meta = json.loads(read(resumed, "meta.json"))
    assert [r["from_step"] for r in meta["resumes"]] == [200]


def test_final_json_and_schedule(small_run, env):
    make, manifests = small_run
    run_dir = train.run(make(), env=env, manifest_dir=manifests)
    final = json.loads(read(run_dir, "final.json"))
    assert final["steps"] == [500, 600] and final["episodes_per_eval"] == 3
    evals = read(run_dir, "eval.csv").decode().splitlines()
    n_eps = [int(line.split(",")[1]) for line in evals[1:]]
    assert n_eps == [2, 2, 2, 2, 3, 3]  # 10-episode schedule, 50 at the final evaluations (scaled)
    assert train.run(make(), env=env, manifest_dir=manifests) == run_dir  # finished: no-op


def test_killed_after_last_checkpoint_writes_final_on_restart(small_run, env):
    make, manifests = small_run
    cfg = make()
    run_dir = train.run(cfg, stop_at=600, env=env, manifest_dir=manifests)
    assert not (run_dir / "final.json").exists()
    train.run(cfg, env=env, manifest_dir=manifests)
    assert json.loads(read(run_dir, "final.json"))["steps"] == [500, 600]


def test_r2_same_seed_same_losses_other_seed_differs(small_run, env):
    make, manifests = small_run
    a = train.run(make(out="a"), env=env, manifest_dir=manifests)
    b = train.run(make(out="b"), env=env, manifest_dir=manifests)
    c = train.run(make("seed=101", out="c"), env=env, manifest_dir=manifests)
    assert read(a, "metrics.csv") == read(b, "metrics.csv")
    assert read(a, "metrics.csv") != read(c, "metrics.csv")


def test_r2_bc_and_iql_shaped_trainers_start_identically(make_mixture):
    cfg = {"net": {"hidden": [32, 32], "log_std_min": -20.0, "log_std_max": 2.0}}
    init, buf_seed, _ = derive_seeds(7)
    policies = []
    for extra_nets in (False, True):
        seed_everything(init)
        policies.append(make_policy(cfg, 11, 3, torch.device("cpu")))  # always built first
        if extra_nets:  # IQL builds its value networks after the policy
            TwinQ(11, 3, (32, 32)), ValueNet(11, (32, 32))
    for p, q in zip(policies[0].parameters(), policies[1].parameters(), strict=True):
        assert torch.equal(p, q)
    arrays = make_mixture(n=100)
    x = ReplayBuffer.from_mixture(arrays, False, "cpu", buf_seed).sample(16)
    y = ReplayBuffer.from_mixture(arrays, True, "cpu", buf_seed).sample(16)  # IQL scales rewards
    assert torch.equal(x.obs, y.obs)


def test_derived_seeds_are_independent_streams():
    init, buf, diag = derive_seeds(100)
    assert len({100, init, buf, diag}) == 4
    assert derive_seeds(100) == [init, buf, diag]


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("eval.seed_offset=10000",), "reserved for the final grid"),
        (("eval.seed_offset=10001",), "reserved for the final grid"),
        (("eval.seed_offset=150",), "collection seeds"),
        (("data.state_std_eps=0.01",), "state_std_eps"),
        (("data.normalize_states=false",), "normalize_states"),
        (("train.eval_every=250",), "multiple"),
        (("train.checkpoint_every=130",), "multiple"),
    ],
)
def test_r3_preflight_refuses(small_run, env, extra, match):
    make, manifests = small_run
    with pytest.raises(train.PreflightError, match=match):
        train.run(make(*extra), env=env, manifest_dir=manifests)


def test_r3_final_mixture_needs_final_seeds(small_run, env):
    make, manifests = small_run
    with pytest.raises(train.PreflightError, match="final evaluation seeds"):
        train.run(make("seed=0"), env=env, manifest_dir=manifests)


def test_r3_hash_and_manifest_mismatch(small_run, env):
    make, manifests = small_run
    cfg = make()
    path = manifests / "mix_p1_s100.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "content_sha256": "0" * 64}))
    with pytest.raises(train.PreflightError, match="content_sha256"):
        train.run(cfg, env=env, manifest_dir=manifests)
    path.write_text(json.dumps({**manifest, "p_target": 0.5}))
    with pytest.raises(train.PreflightError, match="manifest is"):
        train.run(cfg, env=env, manifest_dir=manifests)


def test_r3_resume_refuses_changed_config_or_commit(small_run, env, monkeypatch):
    make, manifests = small_run
    cfg = make()
    train.run(cfg, stop_at=200, env=env, manifest_dir=manifests)
    with pytest.raises(train.PreflightError, match="different config"):
        train.run(make("train.lr=0.001"), env=env, manifest_dir=manifests)
    real = train.collect_metadata
    monkeypatch.setattr(train, "collect_metadata", lambda: {**real(), "git_commit": "deadbeef"})
    with pytest.raises(train.PreflightError, match="code has changed"):
        train.run(cfg, env=env, manifest_dir=manifests)


def test_truncate_csv_drops_later_and_half_written_rows(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("step,loss\n100,0.5\n200,0.4\n300,0.3\n40", encoding="utf-8")
    assert truncate_csv(path, 200) == 2
    assert path.read_text(encoding="utf-8") == "step,loss\n100,0.5\n200,0.4\n"


def test_yaml_typos_are_rejected():
    base = {"train": {"steps": 1}, "algo": {"name": "bc"}}
    assert unknown_keys(base, {"train": {"stpes": 2}, "algo": {"beta": 0}}, free=("algo",)) == ["train.stpes"]


def test_r3_final_mixture_needs_exactly_the_final_seeds(small_run):
    make, manifests = small_run
    for n, ok in ((50, True), (100, False)):
        cfg = make("seed=0", "eval.seed_offset=10000", f"eval.final_episodes={n}")
        name = train.mx.mixture_name(1.0, 0)
        arrays = train.mx.load_mixture(train.resolve_path(cfg["data"]["mixture_dir"]) / f"{name}.npz")
        if ok:
            assert train.preflight(cfg, arrays, manifests)["passed"]
        else:
            with pytest.raises(train.PreflightError, match="final evaluation seeds"):
                train.preflight(cfg, arrays, manifests)


def test_resume_refuses_another_device(small_run, env, monkeypatch):
    make, manifests = small_run
    cfg = make()
    train.run(cfg, stop_at=200, env=env, manifest_dir=manifests)
    monkeypatch.setattr(train, "pick_device", lambda _: torch.device("cuda"))
    with pytest.raises(train.PreflightError, match="same setup"):
        train.run(cfg, env=env, manifest_dir=manifests)


def test_restart_before_first_checkpoint_clears_tensorboard(small_run, env):
    make, manifests = small_run
    cfg = make()
    run_dir = train.run(cfg, stop_at=100, env=env, manifest_dir=manifests)
    (run_dir / "tb" / "stale").write_text("x", encoding="utf-8")
    train.run(cfg, stop_at=150, env=env, manifest_dir=manifests)
    assert not (run_dir / "tb" / "stale").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CI and the laptop are CPU-only")
def test_r1_resume_on_cuda_is_bitwise_identical(small_run, env):
    make, manifests = small_run
    full = train.run(make("train.device=cuda", out="a"), env=env, manifest_dir=manifests)
    cfg = make("train.device=cuda", out="b")
    train.run(cfg, stop_at=350, env=env, manifest_dir=manifests)
    resumed = train.run(cfg, env=env, manifest_dir=manifests)
    for name in COMPARED:
        assert read(full, name) == read(resumed, name), name


def test_finished_run_refuses_changed_config_or_commit(small_run, env, monkeypatch):
    make, manifests = small_run
    cfg = make()
    train.run(cfg, env=env, manifest_dir=manifests)
    with pytest.raises(train.PreflightError, match="different config"):
        train.run(make("train.lr=0.001"), env=env, manifest_dir=manifests)
    real = train.collect_metadata
    monkeypatch.setattr(train, "collect_metadata", lambda: {**real(), "git_commit": "deadbeef"})
    with pytest.raises(train.PreflightError, match="code has changed"):
        train.run(cfg, env=env, manifest_dir=manifests)


def test_resume_refuses_dirty_tree(small_run, env, monkeypatch):
    make, manifests = small_run
    cfg = make()
    train.run(cfg, stop_at=200, env=env, manifest_dir=manifests)
    real = train.collect_metadata
    monkeypatch.setattr(train, "collect_metadata", lambda: {**real(), "git_dirty": True})
    with pytest.raises(train.PreflightError, match="dirty tree"):
        train.run(cfg, env=env, manifest_dir=manifests)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("data.scale_rewards=true",), "scale_rewards must be false for bc"),
        (("algo.name=nope",), "unknown algo.name"),
    ],
)
def test_preflight_reward_scaling_and_algo_name(small_run, env, extra, match):
    make, manifests = small_run
    with pytest.raises(train.PreflightError, match=match):
        train.run(make(*extra), env=env, manifest_dir=manifests)


def test_update_timing_excludes_checkpoint_writes(small_run, env, monkeypatch):
    make, manifests = small_run
    real = train.save_checkpoint

    def slow(*args):
        time.sleep(0.5)
        real(*args)

    monkeypatch.setattr(train, "save_checkpoint", slow)
    run_dir = train.run(make(), env=env, manifest_dir=manifests)
    rows = (run_dir / "timing.csv").read_text(encoding="utf-8").splitlines()[1:]
    train_s = [float(r.split(",")[2]) for r in rows if r.split(",")[1] == "train"]
    assert len(train_s) == 12 and max(train_s) < 0.5
