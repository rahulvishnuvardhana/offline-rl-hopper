"""Phase 5 V1 follow-up: apply the seed 102 to 104 rule, write report/phase5/seed_followup.json.

Spec: docs/specs/phase-5-iql.md Section 8 (user decisions D5 to D7). Runs on Explorer after the
follow-up array (slurm/iql_seeds.sbatch), on a clean tree. ``--commit`` is the commit the array
was pinned to; every run (and every resume) must be at it:
    python scripts/seed_followup.py --commit <sha> --set data.mixture_dir=... train.out_dir=... \
        [--rerun-out-dir <dir of the reruns>]
The runs are read, never trained here. Exit codes: 0 proceed, 10 investigate, 11 failed runs must
be rerun first (listed); anything else (1, 2, 3) is a usage, tree or runtime error, not an outcome.
A rerun counts only once its run directory exists in ``--rerun-out-dir``; until then it is pending.
Run it only after the training (and any rerun) job has ended: a rerun still training has a
directory but no ``final.json`` yet, which reads as a second failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from offrl import train
from offrl.analysis import seed_followup as sf
from offrl.config import Config, load_checked
from offrl.data import mixture as mx
from offrl.provenance import dirty_paths
from offrl.runmeta import collect_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "phase5" / "seed_followup.json"
STEPS = 300_000
P = 1.0
EXIT = {"proceed": 0, "investigate": 10, "rerun": 11}
REFUSED = 3


def config(algo: str, seed: int, paths: list[str]) -> Config:
    return load_checked(
        train.BASE_CONFIG,
        ROOT / "configs" / "algo" / f"{algo}.yaml",
        overrides=[f"seed={seed}", f"data.p_expert={P}", f"train.steps={STEPS}", *paths],
    )


def comparable(cfg: Config) -> Config:
    """The config minus where it ran and where its files live (as iql_validate.validation_view)."""
    view = train._comparable(cfg)
    return {
        **view,
        "data": {k: v for k, v in view["data"].items() if k != "mixture_dir"},
        "train": {k: v for k, v in view["train"].items() if k != "out_dir"},
    }


def run_path(algo: str, seed: int, paths: list[str]) -> Path:
    located = config(algo, seed, paths)
    return train.resolve_path(located["train"]["out_dir"]) / train.run_name(located)


def finite(obj: Any) -> Any:
    """``obj`` with non-finite floats as strings, so the report always serializes."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


def read_run(algo: str, seed: int, paths: list[str]) -> dict[str, Any] | None:
    """The run's score and provenance, or None if it has no ``final.json``."""
    located = config(algo, seed, paths)
    run_dir = run_path(algo, seed, paths)
    if not (run_dir / "final.json").is_file():
        return None
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (train.MANIFEST_DIR / f"{mx.mixture_name(P, seed)}.json").read_text(encoding="utf-8")
    )
    return {
        "run_dir": run_dir.as_posix(),
        "final": json.loads((run_dir / "final.json").read_text(encoding="utf-8")),
        "provenance": {
            "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"],
            "gpu": meta["gpu"],
            "content_sha256": meta["preflight"]["content_sha256"],
            "manifest_content_sha256": manifest["content_sha256"],
            "config_matches": comparable(cfg) == comparable(located),
            "resume_records": [
                {k: r.get(k) for k in ("from_step", "git_commit", "git_dirty", "gpu")}
                for r in meta["resumes"]
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--commit", required=True, help="full sha the follow-up array was pinned to")
    parser.add_argument("--set", nargs="*", default=[], help="paths for every run: mixture_dir, out_dir")
    parser.add_argument("--rerun-out-dir", help="out_dir of the reruns (fresh directory), if any")
    args = parser.parse_args()
    allowed = ("data.mixture_dir=", "train.out_dir=")
    if any(not o.startswith(allowed) for o in args.set):
        print(f"--set accepts only {allowed}")
        return REFUSED
    if len(args.commit) != 40:
        print("--commit must be the full 40-character sha")
        return REFUSED
    out_dirs = [o.split("=", 1)[1] for o in args.set if o.startswith("train.out_dir=")]
    if args.rerun_out_dir and any(
        train.resolve_path(args.rerun_out_dir) == train.resolve_path(d) for d in out_dirs
    ):
        print("--rerun-out-dir must be a fresh directory, not train.out_dir")
        return REFUSED
    dirty = dirty_paths()
    if dirty:
        print(f"refusing to run on a dirty tree (commit first): {dirty[:5]}")
        return REFUSED

    keys = [(a, s) for s in sf.SEEDS for a in sf.ALGOS]
    first = {k: read_run(*k, args.set) for k in keys}
    rerun: dict[sf.Key, dict[str, Any] | None] = {}
    if args.rerun_out_dir:
        rerun_paths = [o for o in args.set if not o.startswith("train.out_dir=")]
        rerun_paths.append(f"train.out_dir={args.rerun_out_dir}")
        for k in keys:  # a rerun whose directory does not exist yet is pending, not failed
            if sf.failure(first[k], args.commit) is not None and run_path(*k, rerun_paths).is_dir():
                rerun[k] = read_run(*k, rerun_paths)
    decision = sf.decide(first, rerun, args.commit)

    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": "docs/specs/phase-5-iql.md Section 8 (D5 to D7)",
        "run": {**collect_metadata(), "dirty_paths": dirty},
        "runs": {f"{a}_s{s}": {"first": first[(a, s)], "rerun": rerun.get((a, s))} for a, s in keys},
        "decision": decision,
    }
    report = finite(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    for name, why in decision["failures"].items():
        print(f"FAILED  {name}: {why}")
    if "per_seed" in decision:
        print(f"V1 bar {decision['v1_bar']:.1f}; below CORL = CORL - ours > {decision['gap']:g}")
        for s, v in decision["per_seed"].items():
            print(
                f"{s}: ours {v['ours']:.1f}  CORL {v['corl']:.1f}  gap {v['corl_minus_ours']:.1f}  "
                f"CORL falls short {v['corl_falls_short']}  ours below CORL {v['ours_below_corl']}"
            )
        print(
            f"A (CORL falls short on >= 1 seed): {decision['A']}  "
            f"B (ours below CORL on >= 2): {decision['B']}"
        )
    print(f"\nACTION: {decision['action']}: {decision['reason']}; report written to {OUT.as_posix()}")
    return EXIT[decision["action"]]


if __name__ == "__main__":
    sys.exit(main())
