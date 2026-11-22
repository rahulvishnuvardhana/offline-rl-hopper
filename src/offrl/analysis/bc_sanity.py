"""Phase 4 sanity analysis (spec Section 8): pure functions, tested on synthetic data.

Held-out rows are dataset episodes **not picked** into the mixture, so nothing is removed from
training (no validation split, Phase 3 decision) and the rows were never trained on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from offrl.analysis.eda import normalized_score
from offrl.data.load import Episode
from offrl.data.mixture import EXPERT, SIMPLE
from offrl.data.normalize import StateNormalizer
from offrl.nets import GaussianPolicy

R2_MIN_EXPERT = 0.8  # D3, pre-registered before any sanity run
# No-fall return match (spec Section 8, fixed 2026-11-16 after the 25k dry run, before the
# official runs): the policy's no-fall episodes must return within 10% of the dataset's no-fall
# mean, judged on at least 10 no-fall episodes; fewer is a FAIL (cannot be evaluated).
NO_FALL_REL_TOL = 0.10
NO_FALL_MIN_EPISODES = 10


def heldout_rows(episodes: Sequence[Episode], picks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    """``obs`` and ``act`` of every episode of one source that the mixture did not pick at all
    (tails of trimmed episodes are excluded: they continue a training trajectory)."""
    picked = {int(i) for i, _ in picks}
    unused = [ep for k, ep in enumerate(episodes) if k not in picked]
    if not unused:
        raise ValueError("every episode is in the mixture; no held-out rows")
    return {
        "obs": np.concatenate([ep.obs[:-1] for ep in unused]),
        "act": np.concatenate([ep.act for ep in unused]),
        "n_episodes": np.array(len(unused)),
    }


@torch.no_grad()
def action_fit(
    policy: GaussianPolicy, normalizer: StateNormalizer, obs: np.ndarray, act: np.ndarray
) -> dict[str, Any]:
    """How well the mean action predicts dataset actions, and the per-dimension residual std
    (compared with the learned std to judge whether the data actions were noisy)."""
    x = torch.as_tensor(normalizer.transform(obs))
    a = np.asarray(act, dtype=np.float64)
    mean = policy.act(x).double().numpy()
    resid = a - mean
    sse, sst = (resid**2).sum(0), ((a - a.mean(0)) ** 2).sum(0)
    nll = float(-policy.log_prob(x, torch.as_tensor(a, dtype=torch.float32)).double().mean())
    return {
        "n_rows": len(a),
        "mse": float((resid**2).mean()),
        "r2": float(1.0 - sse.sum() / sst.sum()),
        "r2_per_dim": (1.0 - sse / sst).tolist(),
        "nll": nll,
        "residual_std": resid.std(0).tolist(),
        "learned_std": policy.effective_log_std().exp().tolist(),
    }


def bands(eda_summary: Mapping[str, Any]) -> dict[str, list[float]]:
    """D4: lower = Phase 2's fell-mean score, upper = the dataset's best episode."""
    s = eda_summary["score"]
    out = {}
    for src, (lo, _) in eda_summary["sanity_bands"].items():
        best = eda_summary["datasets"][src]["episodes"]["return"]["max"]
        out[src] = [lo, normalized_score(best, s["r_random"], s["r_expert"])]
    return out


def no_fall_match(no_fall: Mapping[str, Any], dataset_mean: float) -> tuple[bool, str]:
    """Plumbing check: a broken normalizer or action path changes the gait, not just the falls."""
    n, mean = no_fall["n_no_fall"], no_fall["mean"]
    if n < NO_FALL_MIN_EPISODES:
        return False, f"only {n} no-fall episodes (< {NO_FALL_MIN_EPISODES}); cannot be evaluated"
    rel = (mean - dataset_mean) / dataset_mean
    ok = abs(rel) <= NO_FALL_REL_TOL
    return (
        ok,
        f"{mean:.0f} vs dataset {dataset_mean:.0f} ({rel:+.1%}, tolerance {NO_FALL_REL_TOL:.0%}, n={n})",
    )


def no_fall_return(rows: Sequence[Mapping[str, str]], steps: Sequence[int]) -> dict[str, Any]:
    """Mean return of the deterministic policy's no-fall episodes at the final evaluations."""
    keep = [r for r in rows if int(r["step"]) in set(steps)]
    ok = [float(r["return"]) for r in keep if r["fell"] == "False"]
    return {"n_episodes": len(keep), "n_no_fall": len(ok), "mean": float(np.mean(ok)) if ok else None}


def _clean_at(record: Mapping[str, Any], commit: str) -> bool:
    return record["git_commit"] == commit and record["git_dirty"] is False


def assemble_checks(
    results: Sequence[Mapping[str, Any]],
    resume: Mapping[str, bool],
    demo_meta: Mapping[str, Any],
    commit: str,
    no_fall_ref: Mapping[str, float],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The DoD checks and the reported p = 1 vs. p = 0 scores.

    ``results`` holds one entry per run in training order; a 300k rerun (D4) follows its miss,
    so ``runs[-1]`` of a cell is the rerun when there is one. Provenance covers the start of
    every run **and** every resume record, so a resume on another commit or a dirty tree fails.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    by_cell: dict[tuple[float, int], list[Mapping[str, Any]]] = {}
    for r in results:
        by_cell.setdefault((r["p"], r["seed"]), []).append(r)
    for (p, s), runs in by_cell.items():
        detail = "; ".join(
            f"{r['run']}: {r['final']['score']:.1f} in {[round(b, 1) for b in r['band']]}"
            + ("" if r["in_band"] else " MISS (kept, D4)")
            for r in runs
        )
        check(f"band[p{p:g}_s{s}]", any(r["in_band"] for r in runs), detail)
        prov = runs[0]["provenance"]
        check(
            f"R5_provenance[p{p:g}_s{s}]",
            all(
                _clean_at(r["provenance"], commit)
                and all(_clean_at(rec, commit) for rec in r["provenance"]["resume_records"])
                and r["provenance"]["content_sha256"] == r["provenance"]["manifest_content_sha256"]
                for r in runs
            ),
            f"commit {prov['git_commit'][:7]}, dirty {prov['git_dirty']}, "
            f"resumes {sum(len(r['provenance']['resume_records']) for r in runs)}, "
            f"mixture {prov['content_sha256'][:12]}",
        )
        src = EXPERT if p == 1.0 else SIMPLE
        ok, detail = no_fall_match(runs[-1]["no_fall_return"], no_fall_ref[src])
        check(f"no_fall_match[p{p:g}_s{s}]", ok, f"{runs[-1]['run']}: {detail}")
        if p == 1.0:
            r2 = runs[-1]["heldout_fit"]["r2"]
            check(f"heldout_r2[p1_s{s}]", r2 >= R2_MIN_EXPERT, f"R^2 {r2:.3f} >= {R2_MIN_EXPERT} (D3)")
    # Reported, not a gate (spec Section 8.1): "expert data makes BC better" is a hypothesis.
    p1_vs_p0 = {
        f"s{s}": {
            "p1": by_cell[(1.0, s)][-1]["final"]["score"],
            "p0": by_cell[(0.0, s)][-1]["final"]["score"],
        }
        for s in seeds
    }
    records = demo_meta["resumes"]
    check(
        "resume_bitwise",
        all(resume.values())
        and len(records) == 1  # must have resumed, not restarted
        and _clean_at(demo_meta, commit)  # a demo left by an earlier job is not reused
        and all(_clean_at(rec, commit) for rec in records),
        f"{dict(resume)}; resumed from {[r['from_step'] for r in records]} "
        f"at commit {str(demo_meta['git_commit'])[:7]}",
    )
    return checks, p1_vs_p0
