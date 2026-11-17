"""Phase 4 DoD: train the BC sanity runs, check them, write report/phase4/bc_sanity.json + figure.

Spec: docs/specs/phase-4-bc-eval.md Section 8. Runs on an Explorer GPU (slurm/bc_sanity.sbatch):
    python scripts/build_mixtures.py --p 0 1 --seeds 100 101     # once; commit report/mixtures/
    python scripts/bc_sanity.py --rerun-misses --set train.device=cuda data.mixture_dir=... train.out_dir=...
``--set`` overrides apply to every run (device and paths only; never hyperparameters).

Runs: BC at p in {0, 1} x development mixture seeds {100, 101}, plus one resume demonstration
(the p = 1, seed 100 run stopped mid-way and resumed, compared bitwise with the uninterrupted one).
A missed band is kept and reported next to its 300k rerun, never replaced (user decision, D4).
Provenance order (as Phases 2 and 3): commit the code, run on the clean tree, commit report/.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure

from offrl import train
from offrl.algos.bc import make_policy
from offrl.analysis import bc_sanity as bs
from offrl.analysis.eda_plots import COLORS, MUTED, _style, save
from offrl.config import Config, load_checked
from offrl.data import mixture as mx
from offrl.data.load import load_episodes
from offrl.data.mixture import EXPERT, SIMPLE
from offrl.data.normalize import normalizer_from_arrays
from offrl.evaluate import evaluate, make_env, sampled_policy
from offrl.provenance import dirty_paths
from offrl.runlog import read_csv
from offrl.runmeta import collect_metadata
from offrl.seeding import derive_seeds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "phase4"
FIGURE = ROOT / "report" / "figures" / "bc_sanity_curves.png"
EDA_SUMMARY = ROOT / "report" / "eda_summary.json"
P_VALUES = (0.0, 1.0)
RERUN_STEPS = 300_000
RESUME_STOP_FRACTION = 0.375  # stop between checkpoints, so logged rows must be discarded
TRAIN_FIT_ROWS = 50_000


EXTRA: list[str] = []  # --set overrides, applied to every run


def config(p: float, seed: int, steps: int, suffix: str = "") -> Config:
    return load_checked(
        train.BASE_CONFIG,
        ROOT / "configs" / "algo" / "bc.yaml",
        overrides=[
            f"seed={seed}",
            f"data.p_expert={p}",
            f"train.steps={steps}",
            f"train.run_suffix={suffix!r}",
            *EXTRA,
        ],
    )


def source_of(p: float) -> str:
    return EXPERT if p == 1.0 else SIMPLE


def analyze(cfg: Config, run_dir: Path, episodes, env, band: list[float]) -> dict[str, Any]:
    """Score vs band, held-out and training action fit, sampled diagnostic, no-fall return."""
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    p, seed = cfg["data"]["p_expert"], cfg["seed"]
    name = mx.mixture_name(p, seed)
    arrays = mx.load_mixture(train.resolve_path(cfg["data"]["mixture_dir"]) / f"{name}.npz")
    manifest = json.loads((train.MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
    normalizer = normalizer_from_arrays(arrays)
    policy = make_policy(cfg, arrays["obs"].shape[1], arrays["act"].shape[1], torch.device("cpu"))
    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(state["trainer"]["policy"])
    policy.eval()

    src = source_of(p)
    held = bs.heldout_rows(episodes[src], manifest["picks"][src])
    rows = np.random.default_rng(seed).choice(len(arrays["obs"]), TRAIN_FIT_ROWS, replace=False)
    fit_held = bs.action_fit(policy, normalizer, held["obs"], held["act"])
    fit_train = bs.action_fit(policy, normalizer, arrays["obs"][rows], arrays["act"][rows])

    gen = torch.Generator().manual_seed(derive_seeds(seed)[2])
    sampled_final, _ = evaluate(
        env, sampled_policy(policy, normalizer, torch.device("cpu"), gen), cfg, cfg["eval"]["final_episodes"]
    )
    no_fall = bs.no_fall_return(read_csv(run_dir / "eval_episodes.csv"), final["steps"])
    timing = read_csv(run_dir / "timing.csv")
    train_s = [float(r["seconds"]) for r in timing if r["kind"] == "train"]
    eval_s = {
        n: [float(r["seconds"]) for r in timing if r["kind"] == "eval" and int(r["step"]) in steps]
        for n, steps in (
            ("final", set(final["steps"])),
            ("periodic", {int(r["step"]) for r in timing if int(r["step"]) not in set(final["steps"])}),
        )
    }
    return {
        "run": run_dir.name,
        "p": p,
        "seed": seed,
        "steps": cfg["train"]["steps"],
        "final": final,
        "band": band,
        "in_band": band[0] <= final["score"] <= band[1],
        "heldout_fit": {"n_episodes": int(held["n_episodes"]), **fit_held},
        "train_fit": fit_train,
        "no_fall_return": no_fall,
        "sampled_diagnostic_descriptive_only": {
            "score_mean": sampled_final["score_mean"],
            "fall_rate": sampled_final["fall_rate"],
            "return_mean": sampled_final["return_mean"],
            "n_episodes": sampled_final["n_episodes"],
        },
        "timing": {
            "ms_per_update": 1000 * float(np.mean(train_s)) / cfg["train"]["log_every"],
            "s_per_eval_10": float(np.mean(eval_s["periodic"])) if eval_s["periodic"] else None,
            "s_per_eval_50": float(np.mean(eval_s["final"])) if eval_s["final"] else None,
        },
        "provenance": {
            "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"],
            "content_sha256": meta["preflight"]["content_sha256"],
            "manifest_content_sha256": manifest["content_sha256"],
            "resumes": [r["from_step"] for r in meta["resumes"]],
            "resume_records": [
                {k: r[k] for k in ("from_step", "git_commit", "git_dirty")} for r in meta["resumes"]
            ],
        },
        "run_dir": run_dir.as_posix(),
    }


def figure(results: list[dict[str, Any]], bands: dict[str, list[float]], ref: dict[str, float]) -> Figure:
    """Score (10-seed subset, so no variance jump at the final evaluations) and fall rate vs step."""
    fig = Figure(figsize=(9.6, 4.0))
    ax_s, ax_f = fig.subplots(1, 2)
    for src in (EXPERT, SIMPLE):
        ax_s.axhspan(*bands[src], color=COLORS[src], alpha=0.10, linewidth=0)
        ax_s.axhline(ref[src], color=COLORS[src], linewidth=1, linestyle=":")
    for r in results:
        rows = read_csv(Path(r["run_dir"]) / "eval.csv")
        steps = [int(x["step"]) / 1000 for x in rows]
        style = "-" if r["seed"] == min(x["seed"] for x in results) else "--"
        label = f"p={r['p']:g}, seed {r['seed']}" + (
            f", {r['steps'] // 1000}k" if r["steps"] != 100_000 else ""
        )
        color = COLORS[source_of(r["p"])]
        ax_s.plot(steps, [float(x["sub_score_mean"]) for x in rows], style, color=color, lw=2, label=label)
        ax_f.plot(steps, [float(x["sub_fall_rate"]) for x in rows], style, color=color, lw=2, label=label)
    ax_s.set(xlabel="gradient steps (thousands)", ylabel="normalized score", title="BC score (bands shaded)")
    ax_f.set(
        xlabel="gradient steps (thousands)", ylabel="fall rate", title="BC fall rate", ylim=(-0.02, 1.02)
    )
    ax_s.legend(frameon=False, fontsize=8)
    for ax in (ax_s, ax_f):
        _style(ax)
    ax_s.text(
        0.99, 0.01, "dotted: dataset means", transform=ax_s.transAxes, ha="right", fontsize=8, color=MUTED
    )
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--rerun-misses", action="store_true", help="D4: train a 300k rerun of a band miss")
    parser.add_argument("--allow-dirty", action="store_true", help="development only; the report records it")
    parser.add_argument("--set", nargs="*", default=[], help="overrides for every run: device and paths")
    args = parser.parse_args()
    allowed = ("train.device=", "data.mixture_dir=", "train.out_dir=", "train.num_threads=")
    if any(not o.startswith(allowed) for o in args.set):
        print(f"--set accepts only {allowed}")
        return 1
    EXTRA.extend(args.set)

    dirty = dirty_paths()
    if dirty and not args.allow_dirty:
        print(f"refusing to run on a dirty tree (commit first): {dirty[:5]}")
        return 1
    meta = collect_metadata()
    eda = json.loads(EDA_SUMMARY.read_text(encoding="utf-8"))
    bands = bs.bands(eda)
    ref = {EXPERT: 100.0, SIMPLE: eda["score"]["simple_dataset_score"]}
    no_fall_ref = {s: eda["datasets"][s]["episodes"]["no_fall"]["return"]["mean"] for s in (EXPERT, SIMPLE)}

    cells = [(p, s) for s in args.seeds for p in P_VALUES]
    for p, s in cells:
        print(f"training BC p={p:g} seed {s} ({args.steps} steps)", flush=True)
        train.run(config(p, s, args.steps))

    # Resume demonstration on the real data (spec Section 8): stop between checkpoints, resume.
    demo_cfg = config(1.0, args.seeds[0], args.steps, "_resume_demo")
    demo_dir = train.resolve_path(demo_cfg["train"]["out_dir"]) / train.run_name(demo_cfg)
    # A finished demo is reused; resume_bitwise fails it unless it ran at this commit, clean.
    if not (demo_dir / "final.json").exists():
        shutil.rmtree(demo_dir, ignore_errors=True)  # a job killed mid-demo: start the demo afresh
        print("resume demonstration: stopping mid-run, then resuming", flush=True)
        train.run(demo_cfg, stop_at=int(RESUME_STOP_FRACTION * args.steps))
        train.run(demo_cfg)
    reference = train.resolve_path(demo_cfg["train"]["out_dir"]) / train.run_name(
        config(1.0, args.seeds[0], args.steps)
    )
    resume = {
        name: (demo_dir / name).read_bytes() == (reference / name).read_bytes()
        for name in ("metrics.csv", "eval.csv", "eval_episodes.csv")
    }
    demo_meta = json.loads((demo_dir / "meta.json").read_text(encoding="utf-8"))

    episodes = {
        EXPERT: load_episodes(eda["dataset_ids"][EXPERT]),
        SIMPLE: load_episodes(eda["dataset_ids"][SIMPLE]),
    }
    env = make_env(eda["dataset_ids"][EXPERT])
    results = []
    for p, s in cells:
        cfg = config(p, s, args.steps)
        r = analyze(
            cfg,
            train.resolve_path(cfg["train"]["out_dir"]) / train.run_name(cfg),
            episodes,
            env,
            bands[source_of(p)],
        )
        results.append(r)
        if not r["in_band"] and args.rerun_misses:  # D4: keep the miss, add the 300k rerun
            rcfg = config(p, s, RERUN_STEPS, "_300k")
            print(f"band miss at p={p:g} seed {s}: rerun at {RERUN_STEPS} steps", flush=True)
            run_dir = train.run(rcfg)
            results.append(
                {**analyze(rcfg, run_dir, episodes, env, bands[source_of(p)]), "rerun_of": r["run"]}
            )
    env.close()
    for r in results:
        src = source_of(r["p"])
        r["no_fall_return"]["dataset_no_fall_mean"] = no_fall_ref[src]

    checks, p1_vs_p0 = bs.assemble_checks(
        results, resume, demo_meta, meta["git_commit"], no_fall_ref, args.seeds
    )

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    save(figure(results, bands, ref), FIGURE)
    passed = all(c["passed"] for c in checks)
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": "docs/specs/phase-4-bc-eval.md",
        "run": {**meta, "allow_dirty": args.allow_dirty, "dirty_paths": dirty},
        "bands": bands,
        "dataset_reference": {"score": ref, "no_fall_return": no_fall_ref},
        "results": results,
        "resume_demo": {
            "run": demo_dir.name,
            "matches_uninterrupted": resume,
            "resumes": demo_meta["resumes"],
        },
        "checks": checks,
        "all_passed": passed,
        "reported_not_gated": {"score_p1_vs_p0": p1_vs_p0},
        "figure": FIGURE.relative_to(ROOT).as_posix(),
    }
    (OUT / "bc_sanity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    verdict = f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
    print(f"\n{verdict}; report written to {(OUT / 'bc_sanity.json').relative_to(ROOT).as_posix()}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
