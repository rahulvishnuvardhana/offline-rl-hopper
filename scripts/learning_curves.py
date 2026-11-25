"""Extract learning curves from the archived grid runs into one committed JSON file.

    python scripts/learning_curves.py --runs <archive>/runs [--out report/phase7/learning_curves.json]

For every run, the score at each evaluation (every 5k steps) on the 10 evaluation seeds that every
evaluation shares (``sub_score_mean``; the last 5 evaluations also ran 40 more episodes, which are
left out here so the curve has one definition throughout), plus its fall rate on the same seeds.
Read-only on the runs; ``offrl.analysis.plots`` draws the figure from the JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("report/phase7/learning_curves.json"))
    args = ap.parse_args()
    runs = {}
    for grid in ("main", "ablation"):
        for d in sorted(p for p in (args.runs / grid).iterdir() if (p / "eval.csv").exists()):
            cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
            with (d / "eval.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            runs[d.name] = {
                "grid": grid,
                "algo": cfg["algo"]["name"],
                "tau": cfg["algo"].get("expectile"),
                "p": cfg["data"]["p_expert"],
                "seed": cfg["seed"],
                "step": [int(r["step"]) for r in rows],
                "score": [round(float(r["sub_score_mean"]), 3) for r in rows],
                "fall_rate": [round(float(r["sub_fall_rate"]), 3) for r in rows],
            }
    if len(runs) != 70 or any(len(r["step"]) != 60 for r in runs.values()):
        raise SystemExit(f"expected 70 runs of 60 evaluations, got {len(runs)}")
    note = "score and fall_rate on the 10 evaluation seeds shared by every evaluation (sub_* columns)"
    args.out.write_text(
        json.dumps({"note": note, "runs": runs}, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}: {len(runs)} runs x 60 evaluations")


if __name__ == "__main__":
    main()
