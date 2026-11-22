"""Phase 6 spec S2 to S6 (and S3's status rows through the sweep): ``offrl.sweep`` on tiny real
runs, trained in-process by an injected launcher (the real one starts ``python -m offrl.train``)."""

import json
import math
import os
from types import SimpleNamespace

import gymnasium as gym
import pytest
import yaml

from offrl import sweep, train


@pytest.fixture(scope="module")
def env():
    e = gym.make("Hopper-v5")
    yield e
    e.close()


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    real = train.collect_metadata
    fake = lambda: {**real(), "git_dirty": False}  # noqa: E731
    monkeypatch.setattr(train, "collect_metadata", fake)
    monkeypatch.setattr(sweep, "collect_metadata", fake)


@pytest.fixture
def grid(tmp_path, small_run):
    """A 2-algorithm, 2-seed grid on synthetic development mixtures, with its output directory."""
    make, manifests = small_run
    for seed in (100, 101):
        make(f"seed={seed}")  # writes the mixture and its manifest
    mix_dir = train.resolve_path(make()["data"]["mixture_dir"])
    small = {
        "train": {
            "steps": 600,
            "batch_size": 32,
            "log_every": 50,
            "checkpoint_every": 200,
            "eval_every": 100,
        },
        "eval": {"episodes": 2, "final_episodes": 3, "final_score_last_n": 2},
        "net": {"hidden": [32, 32]},
    }
    (tmp_path / "small.yaml").write_text(yaml.safe_dump(small), encoding="utf-8")
    spec = {
        "name": "t",
        "pin": "t",
        "extra_configs": [str(tmp_path / "small.yaml")],
        "axes": {"algo": ["bc", "iql"], "data.p_expert": [1.0], "seed": [100, 101]},
    }
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    g = sweep.load_grid(path)
    out = tmp_path / "out"
    runs = sweep.expand(g, out, [f"data.mixture_dir={mix_dir.as_posix()}"])
    return g, out, runs, manifests


def inproc(env, manifests, fail=()):
    """The injected launcher: train.run in this process; names in ``fail`` exit 1."""
    calls = []

    def launch(runs, cmds, launch):
        codes = []
        for r in runs:
            calls.append(r.name)
            if r.name in fail:
                codes.append(1)
                continue
            cfg = json.loads(json.dumps(r.cfg))
            cfg["train"]["device"], cfg["train"]["num_threads"] = launch.device, launch.threads
            try:
                train.run(cfg, env=env, manifest_dir=manifests)
                codes.append(0)
            except Exception:  # noqa: BLE001 - a failed run is a non-zero exit, as a process would give
                codes.append(1)
        return codes

    launch.calls = calls
    return launch


def L(tmp_path):
    (tmp_path / "logs").mkdir(exist_ok=True)
    return sweep.Launch("cpu", train.torch.get_num_threads(), tmp_path / "logs")


def never_alive(_):
    return False


def always_alive(_):
    return True


def test_expand_order_and_names(grid):
    _, _, runs, _ = grid
    assert [r.name for r in runs] == ["bc_p1_s100", "bc_p1_s101", "iql_p1_s100", "iql_p1_s101"]


def test_duplicate_run_names_refuse_the_grid(grid, tmp_path):
    g, out, _, _ = grid
    dup = {**g, "axes": {"algo": ["bc"], "seed": [100, 100]}}
    with pytest.raises(sweep.SweepError, match="duplicate"):
        sweep.expand(dup, out)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_bundles_cover_each_run_once_and_never_mix(grid, k):
    _, _, runs, _ = grid
    groups = sweep.bundles(runs, {"bc": k, "iql": k})
    flat = [r.name for gs in groups.values() for g in gs for r in g]
    assert sorted(flat) == sorted(r.name for r in runs) and len(flat) == len(runs)
    for algo, gs in groups.items():
        assert all({r.algo for r in g} == {algo} for g in gs)
        assert len(gs) == math.ceil(2 / k)


def test_bundle_trains_then_skips_done(grid, env, tmp_path):
    _, out, runs, manifests = grid
    launch = inproc(env, manifests)
    bc = runs[:2]
    assert sweep.run_bundle(bc, out, L(tmp_path), [], launch, never_alive) == 0
    assert launch.calls == ["bc_p1_s100", "bc_p1_s101"]
    assert sweep.run_bundle(bc, out, L(tmp_path), [], launch, never_alive) == 0
    assert launch.calls == ["bc_p1_s100", "bc_p1_s101"]  # done: no process started again
    assert not list((out / "locks").glob("*.lock"))  # released


def test_partial_resumes_and_started_restarts(grid, env, tmp_path):
    _, out, runs, manifests = grid
    a, b = runs[2], runs[3]
    train.run(a.cfg, stop_at=200, env=env, manifest_dir=manifests)  # partial
    train.run(b.cfg, stop_at=100, env=env, manifest_dir=manifests)  # started
    assert sweep.status(a, out, "cpu", train.torch.get_num_threads(), never_alive)[0] == "partial"
    assert sweep.status(b, out, "cpu", train.torch.get_num_threads(), never_alive)[0] == "started"
    assert sweep.run_bundle([a, b], out, L(tmp_path), [], inproc(env, manifests), never_alive) == 0
    meta = json.loads((train.resolve_path(a.cfg["train"]["out_dir"]) / a.name / "meta.json").read_text())
    assert [r["from_step"] for r in meta["resumes"]] == [200]


def test_conflict_refuses_the_whole_bundle_before_any_run(grid, env, tmp_path):
    _, out, runs, manifests = grid
    a, b = runs[:2]
    run_dir = train.run(a.cfg, stop_at=200, env=env, manifest_dir=manifests)
    meta = json.loads((run_dir / "meta.json").read_text())
    (run_dir / "meta.json").write_text(json.dumps({**meta, "git_commit": "deadbeef"}))
    launch = inproc(env, manifests)
    assert sweep.run_bundle([a, b], out, L(tmp_path), [], launch, never_alive) == sweep.EXIT_CONFLICT
    assert launch.calls == []  # b was not started either


def test_running_run_is_skipped_with_exit_4(grid, env, tmp_path):
    _, out, runs, manifests = grid
    a, b = runs[:2]
    lock = sweep.lock_path(out, a)
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"token": "999_1"}))
    launch = inproc(env, manifests)
    assert sweep.run_bundle([a, b], out, L(tmp_path), [], launch, always_alive) == sweep.EXIT_RUNNING
    assert launch.calls == [b.name]
    assert json.loads(lock.read_text())["token"] == "999_1"  # another job's lock is left alone


def test_stale_lock_is_broken_by_one_contender(tmp_path):
    lock = tmp_path / "locks" / "r.lock"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"token": "dead_0"}))
    assert sweep.acquire(lock, "a_1", never_alive)
    assert json.loads(lock.read_text())["token"] == "a_1"
    assert not sweep.acquire(lock, "b_2", lambda t: t == "a_1")  # a_1 is alive and holds it
    assert sweep.acquire(lock, "a_1", always_alive)  # a requeue of a_1 takes its own lock over


def test_partner_failure_does_not_stop_the_other(grid, env, tmp_path):
    _, out, runs, manifests = grid
    a, b = runs[:2]
    launch = inproc(env, manifests, fail={a.name})
    assert sweep.run_bundle([a, b], out, L(tmp_path), [], launch, never_alive) == 1
    assert (train.resolve_path(b.cfg["train"]["out_dir"]) / b.name / "final.json").exists()


@pytest.mark.parametrize(("codes", "want"), [([0, 0], 0), ([0, 2], 2), ([-9, 0], 137), ([0, -15], 143)])
def test_exit_codes(codes, want):
    assert sweep.exit_code(codes) == want


HW_CPU = {"kind": "cpu", "partition": "short", "constraint": "cascadelake", "model": "X"}
SETS = ["data.mixture_dir=/m"]


def test_init_once_and_setup_must_match(grid, tmp_path):
    g, out, _, _ = grid
    commit = sweep.head_commit()
    setup = sweep.parse_algo_setup(["bc:1:4:8:0.5", "iql:2:8:12:2.6"])
    record = sweep.setup_record(g, commit, HW_CPU, setup, SETS, {"bc": 7, "iql": 10}, 8)
    assert sweep.init(out, record, tmp_path / "logs") is True
    assert (tmp_path / "logs" / "t" / commit[:12]).is_dir()
    assert sweep.init(out, record, tmp_path / "logs") is False  # a rerun with the same setup
    with pytest.raises(sweep.SweepError, match="another setup"):
        sweep.init(out, {**record, "max_submit": 9}, tmp_path / "logs")
    got = sweep.read_setup(out, g, commit, SETS)
    assert got["algos"]["iql"]["time"] == "2:40:00" and got["throttle"] == {"bc": 7, "iql": 10}
    for grid_, commit_, sets_ in (
        ({**g, "sha256": "0" * 64}, commit, SETS),
        (g, "f" * 40, SETS),
        (g, commit, ["data.mixture_dir=/x"]),
    ):
        with pytest.raises(sweep.SweepError, match="does not match"):
            sweep.read_setup(out, grid_, commit_, sets_)


def test_algo_setup_refuses_uneven_cpus():
    with pytest.raises(sweep.SweepError, match="divide"):
        sweep.parse_algo_setup(["iql:3:8:12:2"])


def test_init_needs_every_algorithm_and_one_device_kind(grid, tmp_path):
    base = [
        "--grid", str(tmp_path / "grid.yaml"), "--root", str(tmp_path / "root"), "--init",
        "--hardware", "cpu", "--partition", "short", "--model", "X", "--max-submit", "8",
    ]  # fmt: skip
    both = ["--algo-setup", "bc:1:4:8:1", "iql:1:4:8:1", "--throttle", "bc:7", "iql:10"]
    assert sweep.main([*base, "--constraint", "c", "--algo-setup", "bc:1:4:8:1", "--throttle", "bc:7"]) == 3
    assert sweep.main([*base, *both]) == sweep.EXIT_CONFLICT  # no --constraint for a CPU grid
    assert sweep.main([*base, "--constraint", "c", *both]) == 0
    assert sweep.main([*base, "--constraint", "c", *both]) == 0  # the same setup again: accepted


def test_one_setup_serves_a_grid_with_fewer_algorithms(grid, tmp_path):
    """The gate passes one setup (bc and iql) to every grid; an iql-only grid keeps its own part."""
    g, _, _, _ = grid
    iql_only = tmp_path / "iql.yaml"
    text = (tmp_path / "grid.yaml").read_text().replace("- bc\n  - iql", "- iql")
    iql_only.write_text(text.replace("name: t", "name: ti"))
    argv = [
        "--grid", str(iql_only), "--root", str(tmp_path / "root"), "--init", "--hardware", "cpu",
        "--partition", "short", "--constraint", "c", "--model", "X", "--max-submit", "8",
        "--algo-setup", "bc:1:4:8:1", "iql:1:4:8:1", "--throttle", "bc:7", "iql:8",
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    path = next((tmp_path / "root").glob("ti/*/sweep.json"))
    record = json.loads(path.read_text())
    assert set(record["algos"]) == {"iql"} and record["throttle"] == {"iql": 8}


def test_init_only_options_are_refused_elsewhere(tmp_path):
    argv = ["--grid", "g.yaml", "--root", str(tmp_path), "--throttle", "iql:50", "--list"]
    assert sweep.main(argv) == sweep.EXIT_CONFLICT


def test_unfinished_lists_bundles_and_fails_on_conflict(grid, env, tmp_path):
    _, out, runs, manifests = grid
    groups = sweep.bundles(runs, {"bc": 1, "iql": 1})["bc"]
    train.run(runs[0].cfg, env=env, manifest_dir=manifests)
    threads = train.torch.get_num_threads()
    assert sweep.unfinished(groups, out, "cpu", threads, never_alive) == ([1], {})
    run_dir = train.resolve_path(runs[0].cfg["train"]["out_dir"]) / runs[0].name
    meta = json.loads((run_dir / "meta.json").read_text())
    (run_dir / "meta.json").write_text(json.dumps({**meta, "git_dirty": True}))
    assert list(sweep.unfinished(groups, out, "cpu", threads, never_alive)[1]) == [runs[0].name]


def test_submit_plan_interleaves_and_cuts_waves():
    main = [("main", "bc", i) for i in range(3)]
    abl = [("abl", "iql", i) for i in range(2)]
    waves = sweep.submit_plan([main, abl], max_submit=2)
    assert waves == [
        [("main", "bc", 0), ("abl", "iql", 0)],
        [("main", "bc", 1), ("abl", "iql", 1)],
        [("main", "bc", 2)],
    ]


def test_sbatch_args_come_from_the_record(tmp_path):
    record = {
        "grid": "main",
        "commit": "a" * 40,
        "hardware": HW_CPU,
        "throttle": {"iql": 10},
        "algos": sweep.parse_algo_setup(["iql:1:4:8:3.1"]),
    }
    args = sweep.sbatch_args(record, "iql", [0, 3], tmp_path)
    assert "--array=0,3%10" in args and "--constraint=cascadelake" in args and "--time=3:10:00" in args
    assert "--mem=8G" in args and "--cpus-per-task=4" in args and "--partition=short" in args
    assert not any(a.startswith("--gres") for a in args)


def test_check_accepts_flags_and_refuses_provenance(grid, env, tmp_path, monkeypatch):
    g, out, runs, manifests = grid
    r = runs[0]
    run_dir = train.run(r.cfg, env=env, manifest_dir=manifests)
    monkeypatch.setattr(sweep, "PIN_DIR", tmp_path / "pins")
    (tmp_path / "pins").mkdir()
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    (tmp_path / "pins" / "t_bc.yaml").write_text(yaml.safe_dump(sweep.pin_view(saved)))
    meta = json.loads((run_dir / "meta.json").read_text())
    record = {
        "commit": meta["git_commit"],
        "repo_root": str(train.REPO_ROOT),
        "hardware": {"kind": "cpu", "model": meta["cpu_model"]},
    }
    final = json.loads((run_dir / "final.json").read_text())

    def with_final(**changes):
        (run_dir / "final.json").write_text(json.dumps({**final, **changes}))
        return sweep.check_run(r, g, record)

    assert with_final(score=95.0, fall_rate=0.1) == ([], [])
    errors, flags = with_final(score="nan")
    assert errors == [] and "non-finite" in flags[0]  # a result, flagged, not refused
    assert "collapsed" in with_final(score=5.0, fall_rate=0.0)[1][0]
    assert "high-fall" in with_final(score=17.0, fall_rate=0.9)[1][0]  # Phase 5's p = 0.2 kind
    with_final(score=95.0, fall_rate=0.1)
    errors, _ = sweep.check_run(r, g, {**record, "hardware": {"kind": "cpu", "model": "Other CPU"}})
    assert any("cpu_model" in e for e in errors)
    errors, _ = sweep.check_run(r, g, {**record, "hardware": {"kind": "gpu", "model": "Tesla T4"}})
    assert any("gpu" in e for e in errors) and any("device" in e for e in errors)
    (run_dir / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "cpu_model"}))
    assert any("cpu_model" in e for e in sweep.check_run(r, g, record)[0])  # a missing key refuses
    (run_dir / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "repo_root"}))
    assert any("trained from" in e for e in sweep.check_run(r, g, record)[0])
    resume = {
        "from_step": 200,
        "git_commit": meta["git_commit"],
        "git_dirty": False,
        "cpu_model": "Other CPU",
    }
    (run_dir / "meta.json").write_text(json.dumps({**meta, "resumes": [resume]}))
    assert any("cpu_model" in e for e in sweep.check_run(r, g, record)[0])  # resume records checked too
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "pins" / "t_bc.yaml").write_text(yaml.safe_dump({**sweep.pin_view(saved), "seed": 7}))
    assert any("pin" in e for e in sweep.check_run(r, g, record)[0])


def test_set_accepts_only_the_mixture_dir(tmp_path):
    argv = ["--grid", "configs/grid_x.yaml", "--root", str(tmp_path), "--set", "train.steps=10", "--list"]
    assert sweep.main(argv) == sweep.EXIT_CONFLICT


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "alive"),
    [
        (0, "12345_1 gpu offrl R", "", True),
        (0, "", "", False),
        (1, "", "slurm_load_jobs error: Invalid job id specified", False),  # gone from slurmctld
        (1, "", "Socket timed out", True),  # not proof the job is gone
    ],
)
def test_squeue_alive(returncode, stdout, stderr, alive):
    def run(*_, **__):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    assert sweep.squeue_alive("12345_1", run=run) is alive


def test_unreadable_lock_is_stale_only_once_old(tmp_path):
    lock = tmp_path / "locks" / "r.lock"
    lock.parent.mkdir()
    lock.write_text('{"tok')  # killed mid-write
    assert not sweep.acquire(lock, "a_1", never_alive)  # may be being written right now
    old = lock.stat().st_mtime - 2 * sweep.FRESH_LOCK_S
    os.utime(lock, (old, old))
    assert sweep.acquire(lock, "a_1", never_alive)


def test_breaking_a_stale_lock_never_steals_a_fresh_one(tmp_path):
    """B finds a stale lock; while B asks squeue, A breaks it and takes the lock. B's rename then
    moves A's fresh lock: B must give it back and report the run as held."""
    lock = tmp_path / "locks" / "r.lock"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"token": "dead_0"}))

    def slow_squeue(token):
        if token == "dead_0" and not (tmp_path / "a_done").exists():
            (tmp_path / "a_done").write_text("")
            assert sweep.acquire(lock, "a_1", never_alive)  # A, meanwhile
        return False

    assert not sweep.acquire(lock, "b_2", slow_squeue)
    assert json.loads(lock.read_text())["token"] == "a_1"


def test_paired_axes_expand_the_ablation(grid):
    g, out, _, _ = grid
    axes = {
        "algo": ["iql"],
        "algo.expectile,train.run_suffix": [[0.5, "_tau0.5"], [0.9, "_tau0.9"]],
        "data.p_expert": [1.0],
        "seed": [100, 101],
    }
    runs = sweep.expand({**g, "axes": axes}, out, ["data.mixture_dir=/m"])
    names = ["iql_p1_s100_tau0.5", "iql_p1_s101_tau0.5", "iql_p1_s100_tau0.9", "iql_p1_s101_tau0.9"]
    assert [r.name for r in runs] == names
    assert [r.cfg["algo"]["expectile"] for r in runs] == [0.5, 0.5, 0.9, 0.9]
    bad = {"algo": ["iql"], "algo.expectile,train.run_suffix": [[0.5]]}
    with pytest.raises(sweep.SweepError, match="zipped axis"):
        sweep.expand({**g, "axes": bad}, out)


def test_sigterm_releases_the_bundle_locks(grid, tmp_path):
    """Slurm's SIGTERM (time-out, scancel) leaves through SystemExit, so the locks are released."""
    _, out, runs, _ = grid

    def killed(runs_, cmds, launch):
        sweep._on_sigterm(15, None)

    with pytest.raises(SystemExit) as e:
        sweep.run_bundle(runs[:2], out, L(tmp_path), [], killed, never_alive)
    assert e.value.code == 143
    assert not list((out / "locks").glob("*.lock"))


@pytest.mark.requires_data  # a child train process recovers its env from the Minari dataset
def test_s7_subprocess_launcher_trains_a_new_bundle(grid, tmp_path):
    """S7: the real launcher, two `python -m offrl.train` processes in parallel, each with its own
    log; the hidden --manifest-dir points the children at the test manifests. Runs in the
    Explorer test gate (datasets present), skipped in CI."""
    _, out, runs, manifests = grid
    launch = L(tmp_path)
    bundle = runs[:2]
    cmds = [[*c, f"--manifest-dir={manifests.as_posix()}"] for c in sweep.build_commands(bundle, [], launch)]
    codes = sweep.subprocess_launch(bundle, cmds, launch)
    logs = sorted(p.name for p in launch.log_dir.iterdir())
    assert codes == [0, 0], [(p, p.read_text()[-2000:]) for p in launch.log_dir.iterdir()]
    assert logs == sorted(f"{sweep.job_token()}_{r.name}.log" for r in bundle)
    for r in bundle:
        assert (train.resolve_path(r.cfg["train"]["out_dir"]) / r.name / "final.json").exists()
