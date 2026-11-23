"""Run metadata: enough to say exactly which code, packages and machine produced a result."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

PACKAGES = ("torch", "numpy", "gymnasium", "mujoco", "minari", "pandas", "matplotlib")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    # cwd pinned to this repo: launched from elsewhere, git would describe a different repo.
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10, cwd=REPO_ROOT
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """The CPU's model name, e.g. ``Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz``: from
    ``/proc/cpuinfo`` on Linux (the cluster), else what ``platform`` reports. A CPU-only grid
    requires one model across its runs, as a GPU grid requires one GPU (Phase 6 spec D1)."""
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    except OSError:
        pass
    return platform.processor() or None


def collect_metadata() -> dict[str, Any]:
    """Git commit and dirty flag, package versions, host, CPU and GPU model and Slurm job id."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "hostname": platform.node(),
        "cpu_model": cpu_model(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_metadata(run_dir: str | Path) -> dict[str, Any]:
    """Write ``meta.json`` into the run folder and return it."""
    meta = collect_metadata()
    path = Path(run_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta
