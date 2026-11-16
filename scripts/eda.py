"""Phase 2B EDA: report/eda_summary.json + report/figures/*.png (spec: docs/specs/phase-2b-eda.md).

Usage:
    python scripts/eda.py                  # analyse, check, write report/ (exit 1 if a check fails)
    python scripts/eda.py --write-config   # also write the ADR-6 score values to configs/base.yaml

Provenance order: commit the code, run this on the clean tree, then commit report/ and
configs/base.yaml together, so ``git_commit`` in the summary names the code that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from offrl.analysis import eda
from offrl.analysis import eda_plots as plots
from offrl.analysis.eda import EXPERT, SIMPLE, SOURCES
from offrl.config import load_config
from offrl.data.load import Episode, file_hashes, load_episodes
from offrl.data.validate import Check
from offrl.rollout import run_policy
from offrl.runmeta import collect_metadata

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
AUDIT = REPORT / "data_audit.json"
SUMMARY = REPORT / "eda_summary.json"
FIGURES = REPORT / "figures"
BASE_CONFIG = ROOT / "configs" / "base.yaml"

SEEDS = (0, 1, 2)
N_STATES = 20_000
RANDOM_EPISODES = 100
FIGURE_MAX_BYTES = 1_000_000

# C3 thresholds, set from a measurement before coding (spec Section 7): mean -0.002 / -0.004,
# median abs 0.034 / 0.022 per step.
RESIDUAL_MEAN_MAX = 0.02
RESIDUAL_ABS_MEDIAN_MAX = 0.1


def dirty_paths() -> list[str]:
    """Every uncommitted or untracked path. No exemptions: stale outputs or an edited
    report/data_audit.json would otherwise slip into a run that claims to be clean."""
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=ROOT
    )
    return [line[3:] for line in out.stdout.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_policy(dataset_id: str) -> dict[str, Any]:
    import minari

    env = minari.load_dataset(dataset_id).recover_environment()
    try:
        out = run_policy(env, lambda _obs: env.action_space.sample(), range(RANDOM_EPISODES))
    finally:
        env.close()
    r = out["returns"]
    return {
        "env_id": env.spec.id,
        "seeds": [0, RANDOM_EPISODES - 1],
        "return": eda.describe(r),
        "return_se": float(r.std(ddof=1) / np.sqrt(len(r))),
        "length": eda.describe(out["lengths"]),
        "fall_rate": float(out["fell"].mean()),
    }


def analyse(episodes: dict[str, list[Episode]], r_random: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every number in the summary (except provenance and checks) plus the arrays for figures."""
    datasets = {
        s: {
            "episodes": eda.episode_stats(episodes[s]),
            "state_stats_descriptive_only": eda.state_stats(episodes[s]),
            "velocity_clip_fraction": eda.velocity_clipping(episodes[s]),
            "actions": eda.action_stats(episodes[s]),
            "reward_decomposition": eda.reward_decomposition(episodes[s]),
        }
        for s in SOURCES
    }
    r_expert = datasets[EXPERT]["episodes"]["return"]["mean"]
    r_simple = datasets[SIMPLE]["episodes"]["return"]["mean"]

    pca, sep, arrays = {}, {}, {}
    for seed in SEEDS:
        pca[str(seed)], proj = eda.coverage_pca(episodes[EXPERT], episodes[SIMPLE], N_STATES, seed)
        sep[str(seed)], dists = eda.separability(episodes[EXPERT], episodes[SIMPLE], N_STATES, seed)
        if seed == SEEDS[0]:
            arrays["pca"], arrays["knn"] = proj, dists

    def across_seeds(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}

    sep_keys = [k for k in sep[str(SEEDS[0])] if not k.startswith("n_test")]
    sep_summary = {k: across_seeds([sep[str(s)][k] for s in SEEDS]) for k in sep_keys}
    pca_summary = {
        f"pc{i + 1}_explained_variance": across_seeds(
            [pca[str(s)]["explained_variance_ratio"][i] for s in SEEDS]
        )
        for i in range(2)
    }
    mixtures = eda.mixture_reference(episodes, eda.P_GRID, SEEDS, 900_000, r_random, r_expert)

    results = {
        "datasets": datasets,
        "score": {
            "formula": "100 * (R - r_random) / (r_expert - r_random)",
            "r_random": r_random,
            "r_expert": r_expert,
            "r_simple": r_simple,
            "simple_dataset_score": eda.normalized_score(r_simple, r_random, r_expert),
        },
        "sanity_bands": {s: eda.sanity_bands(datasets[s]["episodes"], r_random, r_expert) for s in SOURCES},
        "pca": {"per_seed": pca, "summary": pca_summary},
        "separability": {"k": 10, "n_states_per_source": N_STATES, "per_seed": sep, "summary": sep_summary},
        "mixture_reference": mixtures,
    }
    return results, arrays


def run_checks(results: dict, rerun: dict, audit: dict, ids: dict[str, str]) -> list[Check]:
    checks = []
    for s in SOURCES:
        ds_audit = audit["datasets"][ids[s]]
        ep = results["datasets"][s]["episodes"]
        ours = {"total_steps": ep["n_transitions"], "total_episodes": ep["n_episodes"]}
        checks.append(
            Check(
                f"C1_counts_match_audit[{s}]",
                ours == ds_audit["metadata"],
                f"ours={ours}; audit={ds_audit['metadata']}",
            )
        )

        rehashed = file_hashes(ids[s])
        checks.append(
            Check(
                f"C2_rehash_matches_audit[{s}]",
                rehashed == ds_audit["file_sha256"],
                f"{len(rehashed)} files re-hashed now vs. audit",
            )
        )

        rd = results["datasets"][s]["reward_decomposition"]
        ok = (
            abs(rd["residual_mean"]) < RESIDUAL_MEAN_MAX
            and rd["residual_abs_median"] < RESIDUAL_ABS_MEDIAN_MAX
        )
        checks.append(
            Check(
                f"C3_reward_decomposition[{s}]",
                ok,
                f"residual mean {rd['residual_mean']:.4f} (< {RESIDUAL_MEAN_MAX}), "
                f"median abs {rd['residual_abs_median']:.4f} (< {RESIDUAL_ABS_MEDIAN_MAX})",
            )
        )

        f, nf = ep["fell"], ep["no_fall"]
        means = [g["return"]["mean"] * g["n_episodes"] for g in (f, nf) if g["return"] is not None]
        ok = (
            f["n_episodes"] + nf["n_episodes"] == ep["n_episodes"]
            and f["n_transitions"] + nf["n_transitions"] == ep["n_transitions"]
            and np.isclose(sum(means) / ep["n_episodes"], ep["return"]["mean"], rtol=1e-12, atol=0)
        )
        checks.append(
            Check(
                f"C4_fall_split_adds_up[{s}]",
                bool(ok),
                f"fell {f['n_episodes']} + no_fall {nf['n_episodes']} = {ep['n_episodes']}",
            )
        )

    sc = results["score"]
    checks.append(
        Check(
            "C5_reference_order",
            sc["r_expert"] > sc["r_simple"] > sc["r_random"],
            f"r_expert {sc['r_expert']:.1f} > r_simple {sc['r_simple']:.1f} > r_random {sc['r_random']:.1f}",
        )
    )

    by_p = {row["p"]: row for row in results["mixture_reference"]}
    gaps = {
        0.0: abs(by_p[0.0]["score_mean"] - sc["simple_dataset_score"]),
        1.0: abs(by_p[1.0]["score_mean"] - 100.0),
    }
    checks.append(
        Check(
            "C6_mixture_endpoints_match_datasets",
            all(g < 1.0 for g in gaps.values()),
            f"|score gap| at p=0: {gaps[0.0]:.3f}, at p=1: {gaps[1.0]:.3f} (< 1 point)",
        )
    )

    diffs = eda.compare_numeric(results, rerun, rtol=1e-9)
    checks.append(Check("C7_deterministic_rerun", not diffs, f"{len(diffs)} differing values {diffs[:3]}"))
    return checks


def write_score_config(path: Path, values: dict[str, float]) -> None:
    """Replace the score lines in place, keeping the file's comments (yaml.dump would drop them)."""
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text, n = re.subn(rf"^(  {key}:).*$", rf"\g<1> {value!r}", text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"expected exactly one '  {key}:' line in {path}, found {n}")
    path.write_text(text, encoding="utf-8", newline="\n")


def write_figures(results: dict, arrays: dict, episodes: dict[str, list[Episode]]) -> dict[str, int]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    lengths = {s: np.array([e.length for e in episodes[s]]) for s in SOURCES}
    returns = {s: np.array([e.ret for e in episodes[s]]) for s in SOURCES}
    fell = {s: eda.fell_mask(episodes[s]) for s in SOURCES}
    ds = results["datasets"]
    seed0 = str(SEEDS[0])
    figures = {
        "lengths.png": plots.hist_by_source(lengths, "episode length (steps)", "Episode lengths"),
        "returns.png": plots.hist_by_source(returns, "episode return", "Episode returns"),
        "returns_by_fall.png": plots.returns_by_fall(returns, fell),
        "state_std.png": plots.state_std({s: ds[s]["state_stats_descriptive_only"] for s in SOURCES}),
        "reward_terms.png": plots.reward_terms({s: ds[s]["reward_decomposition"] for s in SOURCES}),
        "pca_coverage.png": plots.pca_coverage(
            arrays["pca"], results["pca"]["per_seed"][seed0]["explained_variance_ratio"]
        ),
        "knn_actions.png": plots.knn_actions(
            arrays["knn"]["d_mix"], arrays["knn"]["d_same"], results["separability"]["per_seed"][seed0]["auc"]
        ),
        "mixture_reference.png": plots.mixture_reference(
            results["mixture_reference"], results["score"]["simple_dataset_score"]
        ),
    }
    sizes = {}
    for name, fig in figures.items():
        plots.save(fig, FIGURES / name)
        sizes[name] = (FIGURES / name).stat().st_size
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write-config", action="store_true", help="write score values to configs/base.yaml")
    parser.add_argument("--allow-dirty", action="store_true", help="development only; marks the run dirty")
    args = parser.parse_args()

    dirty = dirty_paths()
    if dirty and not args.allow_dirty:
        print(
            f"REFUSING: uncommitted changes: {dirty}\n"
            "Commit the code first; delete or restore stale outputs (git restore report/) (C8)."
        )
        return 1

    cfg = load_config(BASE_CONFIG)
    ids = {EXPERT: cfg["data"]["expert_id"], SIMPLE: cfg["data"]["simple_id"]}
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    run = collect_metadata()
    episodes = {s: load_episodes(ids[s]) for s in SOURCES}

    rand = random_policy(ids[EXPERT])
    rand_again = random_policy(ids[EXPERT])
    results, arrays = analyse(episodes, rand["return"]["mean"])
    rerun, _ = analyse(episodes, rand_again["return"]["mean"])
    results["random_policy"], rerun["random_policy"] = rand, rand_again

    checks = run_checks(results, rerun, audit, ids)
    checks.append(
        Check(
            "C8_clean_tree",
            not dirty and run["git_dirty"] is False,
            f"git_dirty={run['git_dirty']} at commit {run['git_commit']}"
            + (f"; --allow-dirty with changes: {dirty}" if dirty else ""),
        )
    )
    sizes = write_figures(results, arrays, episodes)
    too_big = {k: v for k, v in sizes.items() if v > FIGURE_MAX_BYTES}
    checks.append(Check("figures_under_1MB", not too_big, f"sizes {sizes}"))
    all_passed = all(c.passed for c in checks)

    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run,
        "audit": {
            "path": AUDIT.relative_to(ROOT).as_posix(),
            "sha256": sha256(AUDIT),
            "git_commit": audit["run"]["git_commit"],
        },
        "dataset_ids": ids,
        **results,
        "figures": sizes,
        "checks": [c.as_dict() for c in checks],
        "all_passed": all_passed,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    sc = results["score"]
    print(f"\nr_random {sc['r_random']:.2f} | r_expert {sc['r_expert']:.2f} | r_simple {sc['r_simple']:.2f}")

    if args.write_config:
        if all_passed:
            write_score_config(BASE_CONFIG, {k: sc[k] for k in ("r_random", "r_expert", "r_simple")})
            print(f"score values written to {BASE_CONFIG.relative_to(ROOT).as_posix()}")
        else:
            print("NOT writing config: some checks failed")
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}; summary written to {SUMMARY}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
