"""Phase 5 validation: check the 14 runs (V0 to V4), write report/phase5/iql_validation.json + figure.

Spec: docs/specs/phase-5-iql.md Section 8. Runs on Explorer after the training array
(slurm/iql_validate.sbatch), on a clean tree; V0 checks that all runs share one training commit,
so the analysis may run at a later commit (both are recorded):
    python scripts/iql_validate.py --set data.mixture_dir=... train.out_dir=...
``--set`` takes paths only. The runs are read, never trained here: a missing ``final.json`` fails.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure

from offrl import train
from offrl.algos.iql import IQLTrainer
from offrl.analysis import bc_sanity as bs
from offrl.analysis import iql_validation as iv
from offrl.analysis.eda_plots import COLORS, MUTED, _style, save
from offrl.config import Config, load_checked
from offrl.data import mixture as mx
from offrl.data.load import load_episodes
from offrl.data.mixture import EXPERT, SIMPLE
from offrl.data.normalize import normalizer_from_arrays
from offrl.provenance import dirty_paths
from offrl.runlog import read_csv
from offrl.runmeta import collect_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "phase5"
FIGURE = ROOT / "report" / "figures" / "iql_validation_curves.png"
EDA_SUMMARY = ROOT / "report" / "eda_summary.json"
PHASE4 = ROOT / "report" / "phase4" / "bc_sanity.json"
STEPS = 300_000
SEEDS = (100, 101)
RUNS = [("iql", p) for p in (1.0, 0.2, 0.0)] + [("corl_iql", p) for p in (1.0, 0.2)]
RUNS += [("bc", p) for p in (1.0, 0.2)]
ROADMAP_P1_TARGET = 90.0  # reported only (D3)
VALUE_KEYS = ("q_mean", "q_target_mean", "v_mean")

EXTRA: list[str] = []  # --set overrides: paths only


def config(algo: str, p: float, seed: int) -> Config:
    return load_checked(
        train.BASE_CONFIG,
        ROOT / "configs" / "algo" / f"{algo}.yaml",
        overrides=[f"seed={seed}", f"data.p_expert={p}", f"train.steps={STEPS}", *EXTRA],
    )


VENDORED = "src/offrl/_vendor/corl_iql.py"


def vendored_sha256(commit: str | None) -> tuple[str | None, str | None]:
    """``(hash, error)``: the vendored CORL file at the training commit, hashed without its first
    line (our notice), as `_vendor/README.md` and test I12 record it; or ``(None, reason)`` if
    there is no single training commit or git cannot read the file there (e.g. a shallow clone)."""
    if commit is None:
        return None, "no single training commit"
    got = subprocess.run(["git", "show", f"{commit}:{VENDORED}"], capture_output=True, cwd=ROOT)
    if got.returncode != 0:  # report it, do not lose the analysis
        reason = f"unreadable at {commit[:7]}: {got.stderr.decode().strip()}"
        print(f"vendored CORL file {reason}")
        return None, reason
    return hashlib.sha256(got.stdout.split(b"\n", 1)[1]).hexdigest(), None


def validation_view(cfg: Config) -> Config:
    """What makes a run the validation run: the config minus where it ran (device, threads) and
    where its files live (the mixture is identified by its content hash, checked in V0)."""
    view = train._comparable(cfg)
    return {
        **view,
        "data": {k: v for k, v in view["data"].items() if k != "mixture_dir"},
        "train": {k: v for k, v in view["train"].items() if k != "out_dir"},
    }


def mixture_facts(
    name: str, arrays: dict[str, np.ndarray], manifest: dict[str, Any], episodes
) -> dict[str, Any]:
    scale = float(arrays["reward_scale"])
    scaled = np.asarray(arrays["rew"], dtype=np.float64) * scale
    return {
        "reward_scale_file": scale,
        "reward_scale_recomputed": iv.reward_scale_from_episodes(episodes, manifest["picks"]),
        "done_rows": int(np.asarray(arrays["done"]).sum()),
        "terminated_episodes": iv.terminated_count(episodes, manifest["picks"]),
        "timeout_mixups": iv.timeout_mixups(episodes, manifest["picks"]),
        "scaled_reward_min": float(scaled.min()),
        "scaled_reward_max": float(scaled.max()),
        "content_sha256": manifest["content_sha256"],
        "content_sha256_file": mx.content_sha256(arrays),  # the copy this analysis read
    }


@torch.no_grad()
def v_at_initial_states(cfg: Config, run_dir: Path, arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    """Mean V(s0) of the final checkpoint and the mean discounted scaled return from s0."""
    gamma, scale = cfg["algo"]["discount"], float(arrays["reward_scale"])
    s0, ret = iv.initial_states_and_returns(arrays["obs"], arrays["rew"], arrays["t"], scale, gamma)
    trainer = IQLTrainer(cfg, arrays["obs"].shape[1], arrays["act"].shape[1], torch.device("cpu"))
    trainer.load_state_dict(train.load_checkpoint(run_dir / "ckpt" / "latest.pt")["trainer"])
    x = torch.as_tensor(normalizer_from_arrays(arrays).transform(s0), dtype=torch.float32)
    return float(trainer.v(x).double().mean()), float(ret.mean())


def at_steps(rows: list[dict[str, str]], key: str, every: int = 50_000) -> dict[str, float]:
    return {r["step"]: float(r[key]) for r in rows if int(r["step"]) % every == 0}


def analyze(algo: str, p: float, seed: int, mix: dict[str, Any], arrays, manifest) -> dict[str, Any]:
    located = config(algo, p, seed)
    run_dir = train.resolve_path(located["train"]["out_dir"]) / train.run_name(located)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))  # what was trained
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "algo": algo,
        "p": p,
        "seed": seed,
        "run": run_dir.name,
        "mixture": mx.mixture_name(p, seed),
        "final": final,
        "reward_scale": meta["reward_scale"],
        "beta_eff": meta["beta_eff"],
        "provenance": {
            "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"],
            "gpu": meta["gpu"],
            "content_sha256": meta["preflight"]["content_sha256"],
            "manifest_content_sha256": manifest["content_sha256"],
            # The trained config must be the validation config (300k steps etc.): run names
            # carry no step count, so a stray Phase 4 run in out_dir would otherwise be read.
            "config_matches": validation_view(cfg) == validation_view(located),
            "steps": cfg["train"]["steps"],
            "resume_records": [
                {k: r[k] for k in ("from_step", "git_commit", "git_dirty")} for r in meta["resumes"]
            ],
        },
        "run_dir": run_dir.as_posix(),
    }
    timing = read_csv(run_dir / "timing.csv")
    train_s = [float(r["seconds"]) for r in timing if r["kind"] == "train"]
    out["ms_per_update_two_per_gpu"] = 1000 * float(np.mean(train_s)) / cfg["train"]["log_every"]
    if algo == "iql":
        rows = read_csv(run_dir / "metrics.csv")
        lo, hi = iv.value_bounds(
            np.array([mix["scaled_reward_min"], mix["scaled_reward_max"]]), cfg["algo"]["discount"]
        )
        out["v3"] = {"bounded": list(iv.bounded(rows, VALUE_KEYS, lo, hi)), "drift": list(iv.q_drift(rows))}
        out["value_bounds"] = [lo, hi]
        out["frac_clipped"] = at_steps(rows, "frac_clipped")
        out["adv_std"] = at_steps(rows, "adv_std")
        if p == 1.0:
            v_s0, ret_s0 = v_at_initial_states(cfg, run_dir, arrays)
            out["v3"]["v_s0"] = list(iv.v0_ratio(v_s0, ret_s0))
            out["no_fall_return"] = bs.no_fall_return(read_csv(run_dir / "eval_episodes.csv"), final["steps"])
    return out


def figure(results: list[dict[str, Any]]) -> Figure:
    """Ours (solid) vs. CORL (dashed) at p = 1 and p = 0.2: score (10-episode subset, seed mean)
    and our Q and V (seed mean). One axis per panel."""
    fig = Figure(figsize=(9.6, 4.0))
    ax_s, ax_v = fig.subplots(1, 2)
    colors = {1.0: COLORS[EXPERT], 0.2: COLORS[SIMPLE]}

    def seed_mean(algo: str, p: float, key: str, name: str) -> tuple[list[float], np.ndarray]:
        tables = [read_csv(Path(r["run_dir"]) / name) for r in results if r["algo"] == algo and r["p"] == p]
        steps = [int(x["step"]) / 1000 for x in tables[0]]
        return steps, np.mean([[float(x[key]) for x in t] for t in tables], axis=0)

    for p, color in colors.items():
        for algo, style, label in (("iql", "-", "ours"), ("corl_iql", "--", "CORL")):
            x, y = seed_mean(algo, p, "sub_score_mean", "eval.csv")
            ax_s.plot(x, y, style, color=color, lw=1.6, label=f"{label}, p={p:g}")
        for key, style in (("q_mean", "-"), ("v_mean", ":")):
            x, y = seed_mean("iql", p, key, "metrics.csv")
            ax_v.plot(x, y, style, color=color, lw=1.6, label=f"{key.split('_')[0].upper()}, p={p:g}")
    ax_s.set(
        xlabel="gradient steps (thousands)",
        ylabel="normalized score (10-episode subset)",
        title="Score, ours vs. CORL",
    )
    ax_v.set(xlabel="gradient steps (thousands)", ylabel="value (scaled reward units)", title="Our Q and V")
    for ax in (ax_s, ax_v):
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    ax_s.text(
        0.99, 0.01, "mean of seeds 100, 101", transform=ax_s.transAxes, ha="right", fontsize=8, color=MUTED
    )
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--set", nargs="*", default=[], help="paths for every run: mixture_dir, out_dir")
    args = parser.parse_args()
    allowed = ("data.mixture_dir=", "train.out_dir=")
    if any(not o.startswith(allowed) for o in args.set):
        print(f"--set accepts only {allowed}")
        return 1
    EXTRA.extend(args.set)

    dirty = dirty_paths()
    if dirty:
        print(f"refusing to run on a dirty tree (commit first): {dirty[:5]}")
        return 1
    meta = collect_metadata()
    eda = json.loads(EDA_SUMMARY.read_text(encoding="utf-8"))
    no_fall_ref = eda["datasets"][EXPERT]["episodes"]["no_fall"]["return"]["mean"]
    episodes = {s: load_episodes(eda["dataset_ids"][s]) for s in (EXPERT, SIMPLE)}

    mixtures: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for p in (1.0, 0.2, 0.0):
        for seed in SEEDS:
            name = mx.mixture_name(p, seed)
            mix_dir = train.resolve_path(config("iql", p, seed)["data"]["mixture_dir"])
            arrays = mx.load_mixture(mix_dir / f"{name}.npz")
            manifest = json.loads((train.MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
            mixtures[name] = mixture_facts(name, arrays, manifest, episodes)
            for algo, p_run in RUNS:
                if p_run == p:
                    print(f"analyzing {algo} p={p:g} seed {seed}", flush=True)
                    results.append(analyze(algo, p, seed, mixtures[name], arrays, manifest))

    checks = iv.assemble_checks(results, mixtures, lambda nf: bs.no_fall_match(nf, no_fall_ref), SEEDS)
    commit = iv.training_commit(results)

    def score(algo: str, p: float, seed: int) -> float:
        return next(r["final"]["score"] for r in results if (r["algo"], r["p"], r["seed"]) == (algo, p, seed))

    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    reported = {
        "iql_p1_vs_roadmap_90": {f"s{s}": score("iql", 1.0, s) for s in SEEDS}
        | {"target": ROADMAP_P1_TARGET},
        "iql_vs_bc_p0.2": {f"s{s}": {"iql": score("iql", 0.2, s), "bc": score("bc", 0.2, s)} for s in SEEDS},
        "iql_p0": {
            f"s{r['seed']}": {
                "score": r["final"]["score"],
                "fall_rate": r["final"]["fall_rate"],
                "beta_eff": r["beta_eff"],
            }
            for r in results
            if (r["algo"], r["p"]) == ("iql", 0.0)
        },
        "ours_vs_corl_p1": {
            f"s{s}": {"ours": score("iql", 1.0, s), "corl": score("corl_iql", 1.0, s)} for s in SEEDS
        },
        "bc_p1_300k_vs_phase4_100k": {
            f"s{s}": {
                "phase5_300k": score("bc", 1.0, s),
                "phase4_100k": next(
                    r["final"]["score"] for r in phase4["results"] if r["run"] == f"bc_p1_s{s}"
                ),
            }
            for s in SEEDS
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    save(figure(results), FIGURE)
    passed = all(c["passed"] for c in checks)
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": "docs/specs/phase-5-iql.md",
        "run": {**meta, "dirty_paths": dirty},
        "training_commit": commit,
        "note": "run: this analysis run (its commit may be later); training_commit: the one commit "
        "all runs were trained at (V0)",
        **dict(zip(("vendored_corl_sha256", "vendored_corl_error"), vendored_sha256(commit), strict=True)),
        "dataset_reference": {"expert_no_fall_return": no_fall_ref},
        "mixtures": mixtures,
        "results": results,
        "checks": checks,
        "all_passed": passed,
        "reported_not_gated": reported,
        "figure": FIGURE.relative_to(ROOT).as_posix(),
    }
    (OUT / "iql_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    verdict = f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
    print(f"\n{verdict}; report written to {(OUT / 'iql_validation.json').as_posix()}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
