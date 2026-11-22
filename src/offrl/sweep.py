"""Grid runner (Phase 6 spec Section 6): expand a grid file, run one bundle of it per array task,
skip what is done, resume what is partial, and refuse what does not belong.

    python -m offrl.sweep --grid configs/grid_main.yaml --root /scratch/$USER/runs_grid \\
        --set data.mixture_dir=/scratch/$USER/mixtures --init --hardware gpu --gres gpu:t4:1 \\
        --model "Tesla T4" --algo-setup bc:4:16:20:1.5 iql:2:8:12:2.6
    python -m offrl.sweep --grid G --root R --set ... --algo iql --bundle 3
    python -m offrl.sweep --grid G --root R --set ... --list | --summary | --check | --index
    python -m offrl.sweep --grid G --root R --set ... --algo iql --unfinished | --sbatch-args
    python -m offrl.sweep --grid G [--grid G2] --root R --set ... --submit-plan [--max-submit 8]

Runs go to ``<root>/<grid name>/<HEAD[:12]>``; ``sweep.json`` there fixes the setup (``--init``,
in the gate job). A run's status comes from ``train.run_status``; ``running`` is added here, from
locks in ``<out>/locks/`` (kept out of the run directories, so a lock never makes a ``new`` run
look ``started``). Exit codes: 0 every run of the bundle done; 3 a conflict (or the wrong node,
or a setup mismatch), refused before any run starts; 4 a run skipped as ``running``; 128 + n a
run killed by signal n; otherwise the first non-zero exit of a run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from offrl import train
from offrl.config import Config, load_checked
from offrl.runmeta import REPO_ROOT, collect_metadata, cpu_model

EXIT_CONFLICT, EXIT_RUNNING = 3, 4
STATUSES = ("done", "running", "partial", "started", "new", "conflict")
SET_KEYS = ("data.mixture_dir=",)  # the only per-site value; everything else is in the grid file
# Keys a golden pin ignores (Phase 6 spec Section 2): where and as which seed a run runs.
PIN_IGNORED = (
    "seed",
    "data.p_expert",
    "train.run_suffix",
    "train.device",
    "train.num_threads",
    "train.out_dir",
    "data.mixture_dir",
)
PIN_DIR = REPO_ROOT / "configs" / "pins"


class SweepError(RuntimeError):
    """A refusal: setup mismatch, conflict, wrong node. Exits with EXIT_CONFLICT."""


@dataclass(frozen=True)
class Run:
    algo: str
    overrides: tuple[str, ...]
    cfg: Config

    @property
    def name(self) -> str:
        return train.run_name(self.cfg)


# ---------- grid ----------


def load_grid(path: str | Path) -> dict[str, Any]:
    grid = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = {"name", "pin", "axes"} - set(grid)
    if missing:
        raise SweepError(f"{path}: grid file lacks {sorted(missing)}")
    if "algo" not in grid["axes"]:
        raise SweepError(f"{path}: axes must include algo")
    grid.setdefault("extra_configs", [])
    grid["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return grid


def expand(grid: dict[str, Any], out: Path, sets: Sequence[str] = ()) -> list[Run]:
    """The Cartesian product of the axes in the order written, last axis fastest."""
    keys = list(grid["axes"])
    runs = []
    for combo in itertools.product(*(grid["axes"][k] for k in keys)):
        values = dict(zip(keys, combo, strict=True))
        algo = values.pop("algo")
        overrides = (*(f"{k}={v}" for k, v in values.items()), *sets, f"train.out_dir={out.as_posix()}")
        files = [train.BASE_CONFIG.parent / "algo" / f"{algo}.yaml"]
        files += [REPO_ROOT / f for f in grid["extra_configs"]]
        runs.append(Run(algo, overrides, load_checked(train.BASE_CONFIG, *files, overrides=overrides)))
    names = [r.name for r in runs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SweepError(f"grid {grid['name']}: duplicate run names {dupes}")
    return runs


def bundles(runs: Sequence[Run], k: dict[str, int]) -> dict[str, list[list[Run]]]:
    """Per algorithm, consecutive runs in groups of ``k[algo]``; never two algorithms in one."""
    out: dict[str, list[list[Run]]] = {}
    for algo in dict.fromkeys(r.algo for r in runs):
        mine = [r for r in runs if r.algo == algo]
        out[algo] = [mine[i : i + k[algo]] for i in range(0, len(mine), k[algo])]
    return out


def out_dir(root: str | Path, grid: dict[str, Any], commit: str) -> Path:
    return Path(root) / grid["name"] / commit[:12]


# ---------- setup record ----------


def setup_record(
    grid: dict[str, Any], commit: str, hardware: dict[str, str], algo_setup: dict[str, dict]
) -> dict:
    return {
        "grid": grid["name"],
        "grid_sha256": grid["sha256"],
        "commit": commit,
        "repo_root": str(REPO_ROOT),
        "hardware": hardware,
        "algos": algo_setup,
    }


def init(out: Path, record: dict[str, Any], log_root: Path) -> None:
    """Create ``sweep.json`` once (``O_EXCL``) and the log directory; refuse an existing record."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(out / "sweep.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SweepError(f"{out / 'sweep.json'} exists; a setup is fixed once per grid and commit") from None
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    (log_root / record["grid"] / record["commit"][:12]).mkdir(parents=True, exist_ok=True)


def read_setup(out: Path, grid: dict[str, Any], commit: str) -> dict[str, Any]:
    path = out / "sweep.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SweepError(f"{path} missing or unreadable; run --init first (in the gate job)") from None
    expected = {"grid_sha256": grid["sha256"], "commit": commit, "repo_root": str(REPO_ROOT)}
    wrong = {k: (record.get(k), v) for k, v in expected.items() if record.get(k) != v}
    if wrong:
        raise SweepError(f"{path} does not match this call (recorded, now): {wrong}")
    return record


def node_model(kind: str) -> str | None:
    if kind == "gpu":
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return cpu_model()


# ---------- locks ----------


def job_token() -> str:
    array, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if array and task:
        return f"{array}_{task}"
    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}@{socket.gethostname()}"


def squeue_alive(token: str) -> bool:
    """Whether the job holding a lock still exists. Without squeue (CI, laptop), assume it does."""
    if token.startswith("local-"):
        return True
    try:
        out = subprocess.run(["squeue", "-h", "-j", token], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return True
    return out.returncode != 0 or bool(out.stdout.strip())  # an error is not proof the job is gone


def lock_path(out: Path, run: Run) -> Path:
    return out / "locks" / f"{run.name}.lock"


def lock_holder(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["token"]
    except (OSError, ValueError, KeyError, TypeError):
        return "unreadable"


def acquire(path: Path, token: str, alive: Callable[[str], bool]) -> bool:
    """Take a run's lock; False if another live job holds it. A stale lock is broken by an atomic
    rename (only one contender's rename succeeds); a lock with this task's own token (a requeue)
    is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(path)
            if holder == token:
                return True
            if alive(holder):
                return False
            try:
                os.rename(path, path.with_name(f"{path.name}.stale.{token}"))
            except OSError:
                return False  # another contender broke it first and holds it now
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"token": token, "host": socket.gethostname(), "time": time.time()}, f)
        return True
    return False


def release(path: Path, token: str) -> None:
    if lock_holder(path) == token:
        path.unlink(missing_ok=True)


def status(run: Run, out: Path, device: str, threads: int, alive: Callable[[str], bool]) -> tuple[str, str]:
    """Read-only: ``running`` if a live job other than this one holds the lock, else the files'."""
    path = lock_path(out, run)
    if path.exists():
        holder = lock_holder(path)
        if holder != job_token() and alive(holder):
            return "running", f"locked by {holder}"
    return train.run_status(run.cfg, device, threads)


# ---------- running a bundle ----------


@dataclass(frozen=True)
class Launch:
    """What one bundle's processes run with."""

    device: str
    threads: int
    log_dir: Path


def build_commands(runs: Sequence[Run], extra_configs: Sequence[str], launch: Launch) -> list[list[str]]:
    cmds = []
    for r in runs:
        files = [f"configs/algo/{r.algo}.yaml", *extra_configs]
        cmds.append(
            [
                sys.executable,
                "-m",
                "offrl.train",
                *files,
                *r.overrides,
                f"train.device={launch.device}",
                f"train.num_threads={launch.threads}",
            ]
        )
    return cmds


def subprocess_launch(runs: Sequence[Run], cmds: Sequence[Sequence[str]], launch: Launch) -> list[int]:
    """One process per run, in parallel; each one's output to its own log file."""
    env = {**os.environ, "OMP_NUM_THREADS": str(launch.threads), "MKL_NUM_THREADS": str(launch.threads)}
    tag = job_token()
    procs = []
    for r, cmd in zip(runs, cmds, strict=True):
        log = open(launch.log_dir / f"{tag}_{r.name}.log", "w", encoding="utf-8")  # noqa: SIM115
        procs.append(
            (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT), log)
        )
    codes = []
    for p, log in procs:
        codes.append(p.wait())
        log.close()
    return codes


def exit_code(codes: Iterable[int]) -> int:
    for c in codes:
        if c < 0:
            return 128 - c  # killed by signal -c
        if c:
            return c
    return 0


def run_bundle(
    runs: Sequence[Run],
    out: Path,
    launch: Launch,
    extra_configs: Sequence[str],
    launcher: Callable[[Sequence[Run], Sequence[Sequence[str]], Launch], list[int]] = subprocess_launch,
    alive: Callable[[str], bool] = squeue_alive,
) -> int:
    """Locks first, then statuses, then the runs that need training, in parallel."""
    token = job_token()
    locked = {r.name: acquire(lock_path(out, r), token, alive) for r in runs}
    try:
        states = {}
        for r in runs:
            states[r.name] = (
                ("running", "locked")
                if not locked[r.name]
                else train.run_status(r.cfg, launch.device, launch.threads)
            )
        conflicts = {n: why for n, (s, why) in states.items() if s == "conflict"}
        if conflicts:
            for n, why in conflicts.items():
                print(f"CONFLICT {n}: {why}")
            return EXIT_CONFLICT
        todo = [r for r in runs if states[r.name][0] in ("partial", "started", "new")]
        for r in runs:
            print(f"{states[r.name][0].upper():8s} {r.name}")
        codes = launcher(todo, build_commands(todo, extra_configs, launch), launch) if todo else []
        code = exit_code(codes)
        if code:
            return code
        if any(states[r.name][0] == "running" for r in runs):
            return EXIT_RUNNING
        final = [train.run_status(r.cfg, launch.device, launch.threads)[0] for r in runs]
        return 0 if all(s == "done" for s in final) else 1
    finally:
        for r in runs:
            if locked[r.name]:
                release(lock_path(out, r), token)


# ---------- checks and index ----------


def run_type(run: Run) -> str:
    return run.algo if run.algo == "bc" else f"{run.algo}_tau{run.cfg['algo']['expectile']:g}"


def without(cfg: Config, dotted: Iterable[str]) -> Config:
    out = json.loads(json.dumps(cfg))
    for key in dotted:
        *parents, leaf = key.split(".")
        node = out
        for p in parents:
            node = node.get(p, {})
        node.pop(leaf, None)
    return out


def pin_for(grid: dict[str, Any], run: Run) -> Config:
    path = PIN_DIR / f"{grid['pin']}_{run_type(run)}.yaml"
    if not path.exists():
        raise SweepError(f"no golden pin {path.relative_to(REPO_ROOT)}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pin_view(cfg: Config) -> Config:
    return without(cfg, PIN_IGNORED)


def check_run(run: Run, grid: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(errors, flags)`` for one run (Phase 6 spec Section 6, --check). Errors are provenance or
    configuration failures; flags (a non-finite final score) are accepted and reported (Section 2,
    run outcomes)."""
    errors, flags = [], []
    run_dir = train.resolve_path(run.cfg["train"]["out_dir"]) / run.name
    if train.run_status(run.cfg, "unused", 0, collect_metadata())[0] != "done":
        return [f"{run.name}: not done"], flags
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    hw = record["hardware"]
    records = [meta, *meta.get("resumes", [])]
    for rec in records:
        if rec.get("git_commit") != record["commit"] or rec.get("git_dirty") is not False:
            errors.append(f"{run.name}: trained at {rec.get('git_commit')} dirty={rec.get('git_dirty')}")
        key = "gpu" if hw["kind"] == "gpu" else "cpu_model"
        if rec.get(key) != hw["model"]:
            errors.append(f"{run.name}: {key} {rec.get(key)!r}, not {hw['model']!r}")
    want_device = "cuda" if hw["kind"] == "gpu" else "cpu"
    if not str(meta.get("device", "")).startswith(want_device):
        errors.append(f"{run.name}: device {meta.get('device')}, not {want_device}")
    if Path(meta.get("repo_root", "")).resolve() != Path(record["repo_root"]).resolve():
        errors.append(f"{run.name}: trained from {meta.get('repo_root')}, not {record['repo_root']}")
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if pin_view(saved) != pin_for(grid, run):
        errors.append(f"{run.name}: resolved config differs from pin {grid['pin']}_{run_type(run)}")
    tr, ev = run.cfg["train"], run.cfg["eval"]
    rows = train.read_csv(run_dir / "eval.csv")
    if len(rows) != tr["steps"] // tr["eval_every"]:
        errors.append(f"{run.name}: {len(rows)} evaluations, not {tr['steps'] // tr['eval_every']}")
    last = rows[-ev["final_score_last_n"] :]
    if any(int(r["n_episodes"]) != ev["final_episodes"] for r in last):
        errors.append(f"{run.name}: a final evaluation without {ev['final_episodes']} episodes")
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    if not isinstance(final.get("score"), (int, float)) or not math.isfinite(final["score"]):
        flags.append(f"{run.name}: non-finite final score {final.get('score')!r}")
    return errors, flags


INDEX_FIELDS = [
    "run",
    "algo",
    "p",
    "seed",
    "tau",
    "suffix",
    "final_score",
    "fall_rate",
    "flags",
    "commit",
    "dirty",
    "device",
    "gpu",
    "cpu_model",
    "threads",
    "resumes",
    "slurm_job_id",
    "mixture_sha256",
]


def index_rows(runs: Sequence[Run], flags: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        run_dir = train.resolve_path(r.cfg["train"]["out_dir"]) / r.name
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": r.name,
                "algo": r.algo,
                "p": r.cfg["data"]["p_expert"],
                "seed": r.cfg["seed"],
                "tau": r.cfg["algo"].get("expectile", ""),
                "suffix": r.cfg["train"]["run_suffix"],
                "final_score": str(final.get("score")),
                "fall_rate": str(final.get("fall_rate")),
                "flags": "; ".join(flags.get(r.name, [])),
                "commit": meta.get("git_commit"),
                "dirty": meta.get("git_dirty"),
                "device": meta.get("device"),
                "gpu": meta.get("gpu"),
                "cpu_model": meta.get("cpu_model"),
                "threads": meta.get("torch_threads"),
                "resumes": len(meta.get("resumes", [])),
                "slurm_job_id": meta.get("slurm_job_id"),
                "mixture_sha256": meta.get("preflight", {}).get("content_sha256"),
            }
        )
    return rows


# ---------- submission ----------


def unfinished(
    groups: Sequence[Sequence[Run]], out: Path, device: str, threads: int, alive: Callable[[str], bool]
) -> tuple[list[int], dict[str, str]]:
    todo, conflicts = [], {}
    for i, group in enumerate(groups):
        for r in group:
            s, why = status(r, out, device, threads, alive)
            if s == "conflict":
                conflicts[r.name] = why
            elif s in ("partial", "started", "new") and i not in todo:
                todo.append(i)
    return todo, conflicts


def submit_plan(
    tasks: Sequence[Sequence[tuple[str, str, int]]], max_submit: int
) -> list[list[tuple[str, str, int]]]:
    """Interleave the (grid, algo, bundle) lists round-robin, then cut into waves of ``max_submit``
    tasks (a submit limit counts each array task)."""
    order = [t for group in itertools.zip_longest(*tasks) for t in group if t is not None]
    return [order[i : i + max_submit] for i in range(0, len(order), max_submit)]


def sbatch_args(
    record: dict[str, Any], algo: str, indices: Sequence[int], throttle: int, log_root: Path
) -> list[str]:
    a, hw = record["algos"][algo], record["hardware"]
    logs = log_root / record["grid"] / record["commit"][:12]
    args = [
        f"--array={','.join(map(str, indices))}%{throttle}",
        f"--time={a['time']}",
        f"--cpus-per-task={a['cpus']}",
        f"--mem={a['mem_gb']}G",
        f"--partition={hw['partition']}",
        f"--output={logs.as_posix()}/%x_%A_%a.out",
        f"--error={logs.as_posix()}/%x_%A_%a.err",
    ]
    args.append(f"--gres={hw['gres']}" if hw["kind"] == "gpu" else f"--constraint={hw['constraint']}")
    return args


# ---------- CLI ----------


def head_commit() -> str:
    commit = collect_metadata()["git_commit"]
    if commit is None:
        raise SweepError("git commit unknown; refusing to place runs")
    return commit


def parse_algo_setup(items: Sequence[str]) -> dict[str, dict]:
    """``algo:k:cpus:mem_gb:time`` (time as hours, e.g. 2.6, written to Slurm as H:MM:00)."""
    out = {}
    for item in items:
        algo, k, cpus, mem, hours = item.split(":")
        if int(cpus) % int(k):
            raise SweepError(f"{algo}: {cpus} CPUs do not divide into {k} runs")
        minutes = math.ceil(float(hours) * 6) * 10  # rounded up to 10 minutes
        out[algo] = {
            "k": int(k),
            "cpus": int(cpus),
            "mem_gb": int(mem),
            "time": f"{minutes // 60}:{minutes % 60:02d}:00",
        }
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    alive: Callable[[str], bool] = squeue_alive,
    launcher: Callable[..., list[int]] = subprocess_launch,
) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", action="append", required=True)
    p.add_argument("--root", required=True, help="output root; runs go to <root>/<grid>/<HEAD[:12]>")
    p.add_argument("--log-root", default=None, help="default: <root>/../logs")
    p.add_argument("--set", nargs="*", default=[], help="only data.mixture_dir=")
    p.add_argument("--commit", default=None, help="expected HEAD (refused if it differs)")
    p.add_argument("--algo")
    mode = p.add_mutually_exclusive_group(required=True)
    for m in ("init", "list", "summary", "check", "index", "unfinished", "sbatch_args", "submit_plan"):
        mode.add_argument(f"--{m.replace('_', '-')}", action="store_true")
    mode.add_argument("--bundle", type=int)
    p.add_argument("--hardware", choices=["gpu", "cpu"])
    p.add_argument("--partition")
    p.add_argument("--gres")
    p.add_argument("--constraint")
    p.add_argument("--model")
    p.add_argument("--algo-setup", nargs="*", default=[])
    p.add_argument("--throttle", type=int, default=4)
    p.add_argument("--max-submit", type=int, default=8)
    a = p.parse_args(argv)
    try:
        return _main(a, alive, launcher)
    except SweepError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return EXIT_CONFLICT


def _main(a: argparse.Namespace, alive: Callable[[str], bool], launcher: Callable[..., list[int]]) -> int:
    bad = [s for s in a.set if not s.startswith(SET_KEYS)]
    if bad:
        raise SweepError(f"--set takes only {SET_KEYS}: {bad}")
    commit = head_commit()
    if a.commit and a.commit != commit:
        raise SweepError(f"HEAD is {commit}, not --commit {a.commit}")
    log_root = Path(a.log_root) if a.log_root else Path(a.root).parent / "logs"
    grids = [load_grid(g) for g in a.grid]
    if a.submit_plan:
        tasks = []
        for g in grids:
            out = out_dir(a.root, g, commit)
            record = read_setup(out, g, commit)
            groups = bundles(expand(g, out, a.set), {x: v["k"] for x, v in record["algos"].items()})
            for algo, gs in groups.items():
                tasks.append([(g["name"], algo, i) for i in range(len(gs))])
        for n, wave in enumerate(submit_plan(tasks, a.max_submit)):
            print(f"wave {n}: " + " ".join(f"{g}/{algo}/{i}" for g, algo, i in wave))
        return 0
    if len(grids) != 1:
        raise SweepError("one --grid per call except for --submit-plan")
    grid = grids[0]
    out = out_dir(a.root, grid, commit)
    if a.init:
        if not (a.hardware and a.model and a.partition and a.algo_setup):
            raise SweepError("--init needs --hardware, --partition, --model and --algo-setup")
        hw = {"kind": a.hardware, "partition": a.partition, "model": a.model}
        hw["gres" if a.hardware == "gpu" else "constraint"] = a.gres if a.hardware == "gpu" else a.constraint
        if not hw.get("gres") and not hw.get("constraint"):
            raise SweepError("--init needs --gres (gpu) or --constraint (cpu)")
        init(out, setup_record(grid, commit, hw, parse_algo_setup(a.algo_setup)), log_root)
        print(f"initialised {out / 'sweep.json'}")
        return 0
    record = read_setup(out, grid, commit)
    train.check_checkout(out)
    runs = expand(grid, out, a.set)
    k = {x: v["k"] for x, v in record["algos"].items()}
    groups = bundles(runs, k)
    device = "cuda" if record["hardware"]["kind"] == "gpu" else "cpu"
    threads_of = {x: v["cpus"] // v["k"] for x, v in record["algos"].items()}

    if a.bundle is not None:
        if a.algo not in groups or not 0 <= a.bundle < len(groups[a.algo]):
            raise SweepError(f"no bundle {a.algo}/{a.bundle}")
        model = node_model(record["hardware"]["kind"])
        if model != record["hardware"]["model"]:
            raise SweepError(f"this node has {model!r}, the grid requires {record['hardware']['model']!r}")
        launch = Launch(device, threads_of[a.algo], log_root / grid["name"] / commit[:12])
        launch.log_dir.mkdir(parents=True, exist_ok=True)
        return run_bundle(groups[a.algo][a.bundle], out, launch, grid["extra_configs"], launcher, alive)
    if a.list or a.summary:
        counts: dict[str, int] = dict.fromkeys(STATUSES, 0)
        for algo, gs in groups.items():
            for i, group in enumerate(gs):
                for r in group:
                    s, why = status(r, out, device, threads_of[algo], alive)
                    counts[s] += 1
                    if a.list:
                        print(f"{algo}/{i:<3d} {r.name:32s} {s:8s} {why}")
                    elif s in ("running", "partial", "started"):
                        metrics = train.resolve_path(r.cfg["train"]["out_dir"]) / r.name / "metrics.csv"
                        step, age = "-", "-"
                        if metrics.exists():
                            lines = metrics.read_text(encoding="utf-8").splitlines()
                            step = lines[-1].split(",")[0] if len(lines) > 1 else "0"
                            age = f"{(time.time() - metrics.stat().st_mtime) / 60:.0f} min"
                        print(f"{s:8s} {r.name:32s} last step {step}, metrics written {age} ago")
        print("  ".join(f"{s} {n}" for s, n in counts.items()))
        return 0
    if a.unfinished or a.sbatch_args:
        if a.algo not in groups:
            raise SweepError(f"--algo must be one of {sorted(groups)}")
        todo, conflicts = unfinished(groups[a.algo], out, device, threads_of[a.algo], alive)
        if conflicts:
            for n, why in conflicts.items():
                print(f"CONFLICT {n}: {why}", file=sys.stderr)
            return EXIT_CONFLICT
        if a.unfinished:
            print(",".join(map(str, todo)))
        elif todo:
            print(" ".join(sbatch_args(record, a.algo, todo, a.throttle, log_root)))
        return 0
    errors, flags = [], {}
    for r in runs:
        e, f = check_run(r, grid, record)
        errors += e
        if f:
            flags[r.name] = f
    for e in errors:
        print(f"FAIL {e}")
    for f in flags.values():
        print(f"FLAG {'; '.join(f)}")
    if errors:
        return 1
    print(f"all {len(runs)} runs done and consistent ({len(flags)} flagged)")
    if a.index:
        path = out / "runs_index.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
            w.writeheader()
            w.writerows(index_rows(runs, flags))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
