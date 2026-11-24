"""Run the pre-registered analysis (docs/preregistration.md) on the archived Phase 7 and 8 grids.

    python scripts/prereg_analysis.py --runs <archive>/runs --commit <sha> [--out <json>]

<archive>/runs holds main/ and ablation/, one directory per run with config.yaml, meta.json,
final.json and eval.csv. Every run's provenance is checked first; any mismatch stops the analysis.
It assumes every run finished: a run failed twice (Section 9 rule 5, scored 0 and flagged) has no
final.json and stops it, to be handled by hand. Learning curves come from Phase 9's plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offrl.analysis import prereg as pr  # noqa: E402

CPU_MODEL = "Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz"  # D1
FINAL_STEPS = [280000, 285000, 290000, 295000, 300000]
R_RANDOM, R_EXPERT = 17.02882347293913, 3857.803603002423  # ADR-6
EXPECTED = {"main": 50, "ablation": 20}


def finite(x: object) -> bool:
    return not isinstance(x, str) and math.isfinite(x)


def load_run(d: Path, commit: str) -> dict:
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fin = json.loads((d / "final.json").read_text(encoding="utf-8"))
    algo = cfg["algo"]["name"]
    errs = []
    for rec in (meta, *meta.get("resumes", [])):
        if rec.get("git_commit") != commit or rec.get("git_dirty") is not False:
            errs.append(f"commit/dirty {rec.get('git_commit')} {rec.get('git_dirty')}")
        if rec.get("cpu_model") != CPU_MODEL:
            errs.append(f"cpu {rec.get('cpu_model')}")
    if meta.get("device") != "cpu":
        errs.append(f"device {meta.get('device')}")
    if not str(meta.get("repo_root", "")).rstrip("/").endswith(commit):
        errs.append(f"repo_root {meta.get('repo_root')}")
    suffix = f"_tau{cfg['algo']['expectile']:g}" if cfg["train"].get("run_suffix") else ""
    if d.name != f"{algo}_p{cfg['data']['p_expert']:g}_s{cfg['seed']}{suffix}" or fin.get("run") != d.name:
        errs.append("directory name does not match config.yaml or final.json")
    if cfg["train"]["steps"] != 300000 or fin["steps"] != FINAL_STEPS or fin["episodes_per_eval"] != 50:
        errs.append("not the 300k / last-5 / 50-episode final score")
    if cfg["eval"]["seed_offset"] != 10000:
        errs.append("evaluation seeds do not start at 10000")
    if (cfg["score"]["r_random"], cfg["score"]["r_expert"]) != (R_RANDOM, R_EXPERT):
        errs.append("score constants")
    per = [e["score_mean"] for e in fin["per_eval"]]
    exact = finite(fin["score"]) and all(finite(x) for x in per)
    if exact and not math.isclose(fin["score"], float(np.mean(per)), rel_tol=1e-9, abs_tol=1e-9):
        errs.append("final score is not the mean of its 5 evaluations")
    with (d / "eval.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_step = {int(r["step"]): r for r in rows}
    if len(rows) != 60 or sorted(by_step) != list(range(5000, 300001, 5000)):
        errs.append("eval.csv is not 60 evaluations at 5k to 300k")
    elif any((by_step[s]["n_episodes"], by_step[s]["sub_n_episodes"]) != ("50", "10") for s in FINAL_STEPS):
        errs.append("a final evaluation is not 50 episodes with a 10-episode subset")
    elif any(by_step[s]["n_episodes"] != "10" for s in range(255000, 275001, 5000)):
        errs.append("evaluations 51 to 55 are not 10 episodes (the plateau's shared seeds)")
    if errs:
        raise SystemExit(f"{d.name}: provenance FAIL: {'; '.join(errs)}")
    first = [float(by_step[s]["score_mean"]) for s in range(255000, 275001, 5000)]  # evals 51 to 55
    last = [float(by_step[s]["sub_score_mean"]) for s in range(280000, 300001, 5000)]  # evals 56 to 60
    ok = finite(fin["score"])
    plateau_ok = ok and all(math.isfinite(x) for x in first + last)
    return {
        "name": d.name,
        "algo": algo,
        "tau": float(cfg["algo"]["expectile"]) if algo == "iql" else None,
        "p": float(cfg["data"]["p_expert"]),
        "seed": int(cfg["seed"]),
        "score": pr.final_score(fin["score"]),
        "finite": ok,
        "fall_rate": float(fin["fall_rate"]),
        "return_mean": float(fin["return_mean"]) if finite(fin["return_mean"]) else None,
        "plateau": float(np.mean(last) - np.mean(first)) if plateau_ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, default=Path("report/phase7/prereg_analysis.json"))
    args = ap.parse_args()

    runs = []
    for grid, n in EXPECTED.items():
        dirs = sorted(d for d in (args.runs / grid).iterdir() if (d / "final.json").exists())
        if len(dirs) != n:
            raise SystemExit(f"{grid}: {len(dirs)} finished runs, expected {n}")
        runs += [load_run(d, args.commit) for d in dirs]
    by_cell: dict[pr.Cell, dict[int, dict]] = {}
    for r in runs:
        key = pr.bc(r["p"]) if r["algo"] == "bc" else pr.iql(r["tau"], r["p"])
        if r["seed"] in by_cell.setdefault(key, {}):
            raise SystemExit(f"{pr.cell_name(key)}: seed {r['seed']} twice")
        by_cell[key][r["seed"]] = r
    want = {pr.bc(p) for p in pr.P_GRID} | {pr.iql(pr.MAIN_TAU, p) for p in pr.P_GRID}
    want |= {pr.iql(t, p) for t in pr.ABL_TAUS for p in pr.ABL_PS}
    if set(by_cell) != want:
        raise SystemExit(f"cells {sorted(map(pr.cell_name, by_cell))} are not the 14 pre-registered ones")
    for key, seeds in by_cell.items():
        if sorted(seeds) != list(pr.SEEDS):
            raise SystemExit(f"{pr.cell_name(key)}: seeds {sorted(seeds)}")

    score = {k: [v[s]["score"] for s in pr.SEEDS] for k, v in by_cell.items()}
    idx = pr.boot_index()
    cells = {}
    for k in sorted(by_cell, key=lambda c: (c[0], c[1] or 0.0, c[2])):
        v = [by_cell[k][s] for s in pr.SEEDS]
        plateaus = [r["plateau"] for r in v if r["plateau"] is not None]
        rets = [r["return_mean"] for r in v if r["return_mean"] is not None]
        cells[pr.cell_name(k)] = {
            **pr.describe(score[k], idx),
            "fall_rate_per_seed": [r["fall_rate"] for r in v],
            "return_mean": float(np.mean(rets)) if rets else None,
            "return_mean_left_out_nonfinite": len(v) - len(rets),
            "plateau": float(np.mean(plateaus)) if plateaus else None,
            "plateau_left_out_nonfinite": len(v) - len(plateaus),
            "still_climbing": bool(plateaus) and float(np.mean(plateaus)) > pr.CLIMBING,
            "collapsed_seeds": [r["seed"] for r in v if r["score"] < pr.COLLAPSED],
            "high_fall_seeds": [r["seed"] for r in v if r["fall_rate"] >= pr.HIGH_FALL],
            "nonfinite_seeds": [r["seed"] for r in v if not r["finite"]],
        }
    iql_vs_bc = {  # descriptive only (Section 8)
        f"{p:g}": pr.compare(score[pr.iql(pr.MAIN_TAU, p)], score[pr.bc(p)], idx) for p in pr.P_GRID
    }
    out = {
        "commit": args.commit,
        "runs": len(runs),
        "note": pr.INTERVAL_NOTE + ". Each verdict's false-positive rate (1/32) is per hypothesis, "
        "uncorrected for the family of three.",
        "H1": pr.h1(score, idx),
        "H2": pr.h2(score, idx),
        "H3": pr.h3(score, idx),
        "cells": cells,
        "iql_minus_bc_descriptive": iql_vs_bc,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")

    f = lambda xs: ", ".join(f"{x:.1f}" for x in xs)  # noqa: E731
    print(f"{len(runs)} runs, provenance PASS ({args.commit[:12]}, {CPU_MODEL}, clean, 300k)")
    print(f"{'cell':22} {'mean':>6} {'median':>6}  {'per seed':33} falls>=0.5 collapsed plateau")
    for name, c in cells.items():
        plat = "n/a" if c["plateau"] is None else f"{c['plateau']:+.1f}"
        print(
            f"{name:22} {c['mean']:6.1f} {c['median']:6.1f}  {f(c['per_seed']):33} "
            f"{len(c['high_fall_seeds'])}/5        {len(c['collapsed_seeds'])}/5      {plat}"
        )
    print(f"intervals: {pr.INTERVAL_NOTE}")
    for h in ("H1", "H2", "H3"):
        r = out[h]
        c = r.get("comparison") or r.get("contrast")
        extra = f"; test {r['test']}: mean {c['mean']:+.1f}, p_greater {c['p_greater']:.3f}" if c else ""
        print(f"{h}: {r['verdict'].upper()}{extra}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
