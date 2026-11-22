"""Grid runner (Phase 6 spec Section 6): expand a grid file, run one bundle of it per array task,
skip what is done, resume what is partial, and refuse what does not belong.

    python -m offrl.sweep --grid configs/grid_main.yaml --root /scratch/$USER/runs_grid \\
        --set data.mixture_dir=/scratch/$USER/mixtures --init --hardware gpu --partition gpu \\
        --gres gpu:t4:1 --model "Tesla T4" --algo-setup bc:4:16:20:1.5 iql:2:8:12:2.6 \\
        --throttle bc:7 iql:10 --max-submit 8
    python -m offrl.sweep --grid G --root R --set ... --algo iql --bundle 3
    python -m offrl.sweep --grid G --root R --set ... --list | --summary | --check | --index
    python -m offrl.sweep --grid G --root R --set ... --algo iql --unfinished
    python -m offrl.sweep --grid G --root R --set ... --algo iql --sbatch-args --indices 0,1,2
    python -m offrl.sweep --grid G --grid G2 --root R --set ... --submit-plan

Runs go to ``<root>/<grid name>/<HEAD[:12]>``; ``sweep.json`` there fixes the setup (``--init``,
in the gate job), the per-array throttle and the submit limit included. A run's status comes from
``train.run_status``; ``running`` is added here, from locks in ``<out>/locks/`` (kept out of the
run directories, so a lock never makes a ``new`` run look ``started``).

Exit codes of ``--bundle``, in this order of precedence: 3 a conflict, the wrong node or a setup
mismatch, refused before any run starts; then a run process's failure: 128 + n if it was killed
by signal n, else its exit code; then 4 if a run was skipped as ``running``; then 1 if every
process succeeded but a run is still not done; else 0 (every run of the bundle done).
``--sbatch-args`` exits 5 when the requested wave holds no unfinished bundle.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import signal
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

EXIT_CONFLICT, EXIT_RUNNING, EXIT_NOTHING = 3, 4, 5
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
# Pre-registration Section 9: collapsed and high-fall runs are kept, with their scores, and flagged.
COLLAPSE_SCORE, HIGH_FALL_RATE = 10.0, 0.5
UNREADABLE = "unreadable"
FRESH_LOCK_S = 60  # an unreadable lock younger than this may be one being written


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


def _axis_values(key: str, value: Any) -> dict[str, Any]:
    """An axis key ``a,b`` is a zipped axis: each value is a list ``[a_value, b_value]`` (the
    ablation pairs ``algo.expectile`` with its ``train.run_suffix``)."""
    keys = [k.strip() for k in key.split(",")]
    if len(keys) == 1:
        return {key: value}
    if not isinstance(value, list) or len(value) != len(keys):
        raise SweepError(f"zipped axis {key!r} needs lists of {len(keys)} values, got {value!r}")
    return dict(zip(keys, value, strict=True))


def expand(grid: dict[str, Any], out: Path, sets: Sequence[str] = ()) -> list[Run]:
    """The Cartesian product of the axes in the order written, last axis fastest."""
    keys = list(grid["axes"])
    runs = []
    for combo in itertools.product(*(grid["axes"][k] for k in keys)):
        values: dict[str, Any] = {}
        for k, v in zip(keys, combo, strict=True):
            values.update(_axis_values(k, v))
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
    """Per algorithm, consecutive runs in groups of ``k[algo]``; never two algorithms in one.
    Bundle indices are per algorithm (0 to n - 1), as are array indices."""
    out: dict[str, list[list[Run]]] = {}
    for algo in dict.fromkeys(r.algo for r in runs):
        mine = [r for r in runs if r.algo == algo]
        out[algo] = [mine[i : i + k[algo]] for i in range(0, len(mine), k[algo])]
    return out


def out_dir(root: str | Path, grid: dict[str, Any], commit: str) -> Path:
    return Path(root).resolve() / grid["name"] / commit[:12]


# ---------- setup record ----------


def setup_record(
    grid: dict[str, Any],
    commit: str,
    hardware: dict[str, str],
    algo_setup: dict[str, dict],
    sets: Sequence[str],
    throttle: dict[str, int],
    max_submit: int,
) -> dict:
    return {
        "grid": grid["name"],
        "grid_sha256": grid["sha256"],
        "commit": commit,
        "repo_root": str(REPO_ROOT),
        "sets": list(sets),
        "hardware": hardware,
        "algos": algo_setup,
        "throttle": throttle,
        "max_submit": max_submit,
    }


def parse_throttle(items: list[str]) -> dict[str, int]:
    """``algo:n`` items, n concurrent tasks of that algorithm's array."""
    throttle = {}
    for t in items:
        algo, _, n = t.partition(":")
        if not algo or not n.isdigit() or int(n) < 1:
            raise SweepError(f"--throttle takes algo:n with n >= 1, got {t!r}")
        throttle[algo] = int(n)
    return throttle


def allocated_cpus() -> int | None:
    """CPUs of this job: Slurm's count when --cpus-per-task was given, else the CPUs this process
    may run on (Linux), else unknown."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None:
        return int(cpus)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def init(out: Path, record: dict[str, Any], log_root: Path) -> bool:
    """Create ``sweep.json`` once (``O_EXCL``) and the log directory. An existing record equal to
    this one is accepted (a rerun of a gate that stopped after it), so the call returns False;
    a different one is refused: a setup is fixed once per grid and commit."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(out / "sweep.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            same = json.loads((out / "sweep.json").read_text(encoding="utf-8")) == record
        except (OSError, ValueError):
            same = False
        if not same:
            raise SweepError(
                f"{out / 'sweep.json'} exists with another setup; it is fixed once per grid and commit"
            ) from None
        (log_root / record["grid"] / record["commit"][:12]).mkdir(parents=True, exist_ok=True)
        return False
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    (log_root / record["grid"] / record["commit"][:12]).mkdir(parents=True, exist_ok=True)
    return True


def read_setup(out: Path, grid: dict[str, Any], commit: str, sets: Sequence[str]) -> dict[str, Any]:
    path = out / "sweep.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SweepError(f"{path} missing or unreadable; run --init first (in the gate job)") from None
    expected = {
        "grid_sha256": grid["sha256"],
        "commit": commit,
        "repo_root": str(REPO_ROOT),
        "sets": list(sets),
    }
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


def squeue_alive(token: str, run: Callable[..., Any] = subprocess.run) -> bool:
    """Whether the job holding a lock still exists. A job squeue no longer knows ("Invalid job
    id", once it has left slurmctld) is gone; any other error, or no squeue at all (CI, the
    laptop), is not proof that it is gone, so the lock counts as held."""
    if token.startswith("local-"):
        return True
    try:
        out = run(["squeue", "-h", "-j", token], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return True
    if out.returncode != 0:
        return "invalid job id" not in (out.stderr or "").lower()
    return bool(out.stdout.strip())


def lock_path(out: Path, run: Run) -> Path:
    return out / "locks" / f"{run.name}.lock"


def lock_holder(path: Path) -> str | None:
    """The token in a lock; None if there is no lock; ``UNREADABLE`` if it does not parse."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["token"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return UNREADABLE


def _held(path: Path, holder: str, alive: Callable[[str], bool]) -> bool:
    if holder == UNREADABLE:  # possibly being written right now: stale only once it is old
        try:
            return time.time() - path.stat().st_mtime < FRESH_LOCK_S
        except OSError:
            return False
    return alive(holder)


def acquire(path: Path, token: str, alive: Callable[[str], bool]) -> bool:
    """Take a run's lock; False if another live job holds it. A lock with this task's own token
    (a requeue) is taken over. A stale lock is broken by renaming it aside; if the renamed file
    turns out not to be the stale one (another contender broke it and took the lock first), it
    is put back with ``os.link``, which never replaces a file, and the lock counts as held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(path)
            if holder is None:
                continue  # released meanwhile: try again
            if holder == token:
                return True
            if _held(path, holder, alive):
                return False
            aside = path.with_name(f"{path.name}.stale.{token}")
            try:
                os.rename(path, aside)
            except OSError:
                return False
            if lock_holder(aside) != holder:  # renamed someone's fresh lock: give it back
                with contextlib.suppress(OSError):  # a newer lock took its place: leave that one
                    os.link(aside, path)
                aside.unlink(missing_ok=True)
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"token": token, "host": socket.gethostname(), "time": time.time()}, f)
        return True
    return False


def release(path: Path, token: str) -> None:
    if lock_holder(path) == token:
        path.unlink(missing_ok=True)


def status(run: Run, out: Path, device: str, threads: int, alive: Callable[[str], bool]) -> tuple[str, str]:
    """Read-only: ``running`` if another live job holds the lock, else the files' status. A run
    under a stale lock is classified by its files."""
    path = lock_path(out, run)
    holder = lock_holder(path)
    if holder is not None and holder != job_token() and _held(path, holder, alive):
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


_children: list[subprocess.Popen] = []


def _on_sigterm(signum: int, _frame: Any) -> None:
    """Slurm ends a job (time-out, scancel) with SIGTERM: pass it on to the run processes, then
    leave through ``SystemExit`` so ``run_bundle``'s ``finally`` releases the locks."""
    for p in _children:
        if p.poll() is None:
            p.send_signal(signum)
    raise SystemExit(128 + signum)


def subprocess_launch(runs: Sequence[Run], cmds: Sequence[Sequence[str]], launch: Launch) -> list[int]:
    """One process per run, in parallel; each one's output to its own log file."""
    env = {**os.environ, "OMP_NUM_THREADS": str(launch.threads), "MKL_NUM_THREADS": str(launch.threads)}
    tag = job_token()
    logs = []
    for r, cmd in zip(runs, cmds, strict=True):
        log = open(launch.log_dir / f"{tag}_{r.name}.log", "w", encoding="utf-8")  # noqa: SIM115
        logs.append(log)
        _children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT))
    try:
        return [p.wait() for p in _children]
    finally:
        for log in logs:
            log.close()
        _children.clear()


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
    held: list[Path] = []
    try:
        states = {}
        for r in runs:
            path = lock_path(out, r)
            if acquire(path, token, alive):
                held.append(path)
                states[r.name] = train.run_status(r.cfg, launch.device, launch.threads)
            else:
                states[r.name] = ("running", f"locked by {lock_holder(path)}")
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
        for path in held:
            release(path, token)


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
        raise SweepError(f"no golden pin {path.name} in {PIN_DIR}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pin_view(cfg: Config) -> Config:
    return without(cfg, PIN_IGNORED)


def _number(value: Any) -> float | None:
    """A finite float, or None for anything else (a non-finite score is written as a string)."""
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def check_run(run: Run, grid: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(errors, flags)`` for one run (Phase 6 spec Section 6, --check). Errors are provenance or
    configuration failures; flags (a non-finite or collapsed final score) are accepted and
    reported (spec Section 2, run outcomes; pre-registration Section 9)."""
    errors, flags = [], []
    run_dir = train.resolve_path(run.cfg["train"]["out_dir"]) / run.name
    if train.run_status(run.cfg, "unused", 0, collect_metadata())[0] != "done":
        return [f"{run.name}: not done"], flags
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    hw = record["hardware"]
    key = "gpu" if hw["kind"] == "gpu" else "cpu_model"
    for rec in [meta, *meta.get("resumes", [])]:
        if rec.get("git_commit") != record["commit"] or rec.get("git_dirty") is not False:
            errors.append(f"{run.name}: trained at {rec.get('git_commit')} dirty={rec.get('git_dirty')}")
        if rec.get(key) != hw["model"]:
            errors.append(f"{run.name}: {key} {rec.get(key)!r}, not {hw['model']!r}")
    want_device = "cuda" if hw["kind"] == "gpu" else "cpu"
    if not str(meta.get("device", "")).startswith(want_device):
        errors.append(f"{run.name}: device {meta.get('device')}, not {want_device}")
    root = meta.get("repo_root")
    if not root or Path(root).resolve() != Path(record["repo_root"]).resolve():
        errors.append(f"{run.name}: trained from {root!r}, not {record['repo_root']}")
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if pin_view(saved) != pin_for(grid, run):
        errors.append(f"{run.name}: resolved config differs from pin {grid['pin']}_{run_type(run)}")
    tr, ev = run.cfg["train"], run.cfg["eval"]
    rows = train.read_csv(run_dir / "eval.csv")
    if len(rows) != tr["steps"] // tr["eval_every"]:
        errors.append(f"{run.name}: {len(rows)} evaluations, not {tr['steps'] // tr['eval_every']}")
    if any(int(r["n_episodes"]) != ev["final_episodes"] for r in rows[-ev["final_score_last_n"] :]):
        errors.append(f"{run.name}: a final evaluation without {ev['final_episodes']} episodes")
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    score, fall = _number(final.get("score")), _number(final.get("fall_rate"))
    if score is None:
        flags.append(f"{run.name}: non-finite final score {final.get('score')!r} (scored 0)")
    elif score < COLLAPSE_SCORE:
        flags.append(f"{run.name}: collapsed (score {score:.1f})")
    if fall is not None and fall >= HIGH_FALL_RATE:  # instability, not failure: an expert-like gait
        flags.append(f"{run.name}: high-fall (fall rate {fall:.2f})")  # falling half the time scores ~94
    return errors, flags


INDEX_FIELDS = [
    "run", "algo", "p", "seed", "tau", "suffix", "final_score", "fall_rate", "flags", "commit",
    "dirty", "device", "gpu", "cpu_model", "threads", "resumes", "slurm_job_id", "mixture_sha256",
]  # fmt: skip


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


def sbatch_args(record: dict[str, Any], algo: str, indices: Sequence[int], log_root: Path) -> list[str]:
    a, hw = record["algos"][algo], record["hardware"]
    logs = log_root / record["grid"] / record["commit"][:12]
    args = [
        f"--array={','.join(map(str, indices))}%{record['throttle'][algo]}",
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
    meta = collect_metadata()
    if meta["git_commit"] is None:
        raise SweepError("git commit unknown; refusing to place runs")
    return meta["git_commit"]


def require_clean() -> None:
    if collect_metadata()["git_dirty"] is not False:
        raise SweepError("the checkout has uncommitted changes (or git failed); refusing")


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


INIT_ONLY = ("hardware", "partition", "gres", "constraint", "model", "algo_setup", "throttle", "max_submit")


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
    p.add_argument("--indices", help="one wave of --submit-plan, for --sbatch-args")
    mode = p.add_mutually_exclusive_group(required=True)
    for m in ("init", "list", "summary", "check", "index", "unfinished", "sbatch_args", "submit_plan"):
        mode.add_argument(f"--{m.replace('_', '-')}", action="store_true")
    mode.add_argument("--bundle", type=int)
    p.add_argument("--hardware", choices=["gpu", "cpu"])
    p.add_argument("--partition")
    p.add_argument("--gres")
    p.add_argument("--constraint")
    p.add_argument("--model")
    p.add_argument("--algo-setup", nargs="*")
    p.add_argument("--throttle", nargs="*", help="algo:n, concurrent tasks of that algorithm's array")
    p.add_argument("--max-submit", type=int)
    a = p.parse_args(argv)
    if a.bundle is not None:  # Slurm ends a job with SIGTERM; release the locks on the way out
        signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        return _main(a, alive, launcher)
    except SweepError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return EXIT_CONFLICT


def _main(a: argparse.Namespace, alive: Callable[[str], bool], launcher: Callable[..., list[int]]) -> int:
    bad = [s for s in a.set if not s.startswith(SET_KEYS)]
    if bad:
        raise SweepError(f"--set takes only {SET_KEYS}: {bad}")
    if not a.init and any(getattr(a, k) is not None for k in INIT_ONLY):
        raise SweepError(f"{', '.join('--' + k.replace('_', '-') for k in INIT_ONLY)} are for --init only")
    commit = head_commit()
    if a.commit and a.commit != commit:
        raise SweepError(f"HEAD is {commit}, not --commit {a.commit}")
    log_root = Path(a.log_root).resolve() if a.log_root else Path(a.root).resolve().parent / "logs"
    grids = [load_grid(g) for g in a.grid]
    if a.submit_plan:
        tasks, limits = [], set()
        for g in grids:
            out = out_dir(a.root, g, commit)
            record = read_setup(out, g, commit, a.set)
            limits.add(record["max_submit"])
            groups = bundles(expand(g, out, a.set), {x: v["k"] for x, v in record["algos"].items()})
            for algo, gs in groups.items():
                tasks.append([(g["name"], algo, i) for i in range(len(gs))])
        if len(limits) != 1:
            raise SweepError(f"the grids record different submit limits {sorted(limits)}")
        for n, wave in enumerate(submit_plan(tasks, limits.pop())):
            print(f"wave {n}: " + " ".join(f"{g}/{algo}/{i}" for g, algo, i in wave))
        return 0
    if len(grids) != 1:
        raise SweepError("one --grid per call except for --submit-plan")
    grid = grids[0]
    out = out_dir(a.root, grid, commit)
    if a.init:
        return _init(a, grid, commit, out, log_root)
    record = read_setup(out, grid, commit, a.set)
    train.check_checkout(out)
    runs = expand(grid, out, a.set)
    groups = bundles(runs, {x: v["k"] for x, v in record["algos"].items()})
    device = "cuda" if record["hardware"]["kind"] == "gpu" else "cpu"
    threads_of = {x: v["cpus"] // v["k"] for x, v in record["algos"].items()}

    if a.bundle is not None:
        if a.algo not in groups or not 0 <= a.bundle < len(groups[a.algo]):
            raise SweepError(f"no bundle {a.algo}/{a.bundle}")
        require_clean()
        cpus = allocated_cpus()
        if cpus is not None and cpus != record["algos"][a.algo]["cpus"]:
            raise SweepError(
                f"allocated {cpus} CPUs, the grid records {record['algos'][a.algo]['cpus']}: "
                "submit with the arguments of --sbatch-args"
            )
        model = node_model(record["hardware"]["kind"])
        if model != record["hardware"]["model"]:
            raise SweepError(f"this node has {model!r}, the grid requires {record['hardware']['model']!r}")
        launch = Launch(device, threads_of[a.algo], log_root / grid["name"] / commit[:12])
        launch.log_dir.mkdir(parents=True, exist_ok=True)
        return run_bundle(groups[a.algo][a.bundle], out, launch, grid["extra_configs"], launcher, alive)
    if a.list or a.summary:
        return _list(a, groups, out, device, threads_of, alive)
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
            return 0
        wave = [int(i) for i in a.indices.split(",")] if a.indices else todo
        chosen = [i for i in wave if i in todo]
        if not chosen:
            print("nothing unfinished in this wave", file=sys.stderr)
            return EXIT_NOTHING
        print(" ".join(sbatch_args(record, a.algo, chosen, log_root)))
        return 0
    return _check(a, grid, record, runs, out)


def _init(a: argparse.Namespace, grid: dict[str, Any], commit: str, out: Path, log_root: Path) -> int:
    if not (a.hardware and a.model and a.partition and a.algo_setup and a.throttle and a.max_submit):
        raise SweepError(
            "--init needs --hardware, --partition, --model, --algo-setup, --throttle, --max-submit"
        )
    require_clean()
    hw = {"kind": a.hardware, "partition": a.partition, "model": a.model}
    if a.hardware == "gpu" and a.gres and not a.constraint:
        hw["gres"] = a.gres
    elif a.hardware == "cpu" and a.constraint and not a.gres:
        hw["constraint"] = a.constraint
    else:
        raise SweepError("--init needs --gres for a GPU grid or --constraint for a CPU grid, not both")
    algos = set(grid["axes"]["algo"])
    setup = parse_algo_setup(a.algo_setup)  # one setup may serve several grids: keep this grid's
    throttle = parse_throttle(a.throttle or [])
    for name, given in (("--algo-setup", setup), ("--throttle", throttle)):
        if not algos <= set(given):
            raise SweepError(f"{name} covers {sorted(given)}, the grid has {sorted(algos)}")
    setup = {k: v for k, v in setup.items() if k in algos}
    throttle = {k: v for k, v in throttle.items() if k in algos}
    created = init(out, setup_record(grid, commit, hw, setup, a.set, throttle, a.max_submit), log_root)
    print(f"{'initialised' if created else 'already initialised, identical'} {out / 'sweep.json'}")
    return 0


def _list(a, groups, out, device, threads_of, alive) -> int:
    counts: dict[str, int] = dict.fromkeys(STATUSES, 0)
    for algo, gs in groups.items():
        for i, group in enumerate(gs):
            for r in group:
                s, why = status(r, out, device, threads_of[algo], alive)
                counts[s] += 1
                if a.list:
                    print(f"{algo}/{i:<3d} {r.name:32s} {s:8s} {why}")
                elif s in ("running", "partial", "started"):
                    run_dir = train.resolve_path(r.cfg["train"]["out_dir"]) / r.name
                    meta = train._read_json(run_dir / "meta.json") or {}
                    job = lock_holder(lock_path(out, r)) or meta.get("slurm_job_id")
                    step, age = "-", "-"
                    metrics = run_dir / "metrics.csv"
                    if metrics.exists():
                        lines = metrics.read_text(encoding="utf-8").splitlines()
                        step = lines[-1].split(",")[0] if len(lines) > 1 else "0"
                        age = f"{(time.time() - metrics.stat().st_mtime) / 60:.0f} min"
                    print(f"{s:8s} {r.name:32s} job {job}, last step {step}, metrics written {age} ago")
    print("  ".join(f"{s} {n}" for s, n in counts.items()))
    return 0


def _check(a, grid, record, runs, out) -> int:
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
