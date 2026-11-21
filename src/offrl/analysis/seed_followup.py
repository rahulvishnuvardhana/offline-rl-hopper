"""Phase 5 V1 follow-up rule (spec Section 8, user decisions D5 to D7): our IQL vs. CORL's at
``p = 1`` on development seeds 102 to 104. Pure functions; ``scripts/seed_followup.py`` reads the
runs. All comparisons use unrounded scores; rounding is for display only (D6)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

BC_P1_TWO_SEED_MEAN = 105.08619608725604  # job 11981887, report/phase5/iql_validation.json
V1_BAR = BC_P1_TWO_SEED_MEAN - 10.0  # "falls short" = below this (D6)
GAP = 5.0  # "below CORL" = CORL - ours > GAP on the same seed (D6); exactly 5 is not below
SEEDS = (102, 103, 104)
ALGOS = ("iql", "corl_iql")
GPU = "Tesla T4"  # all six runs on T4 (D6), resumes included; the name torch reports

Key = tuple[str, int]  # (algo, seed)


def failure(run: Mapping[str, Any] | None, commit: str) -> str | None:
    """Why a run counts as failed (D7), else None. ``run`` is None when it has no ``final.json``.
    A run that timed out, resumed and finished is complete if every resume passes too."""
    if run is None:
        return "no final score"
    if not math.isfinite(run["final"]["score"]):
        return "non-finite final score"
    prov = run["provenance"]
    for rec in (prov, *prov["resume_records"]):
        if rec["git_commit"] != commit:
            return f"trained at {str(rec['git_commit'])[:7]}, not {commit[:7]}"
        if rec["git_dirty"] is not False:
            return "dirty tree"
        if rec["gpu"] != GPU:
            return f"GPU {rec['gpu']}, not a {GPU}"
    if prov["config_matches"] is not True:
        return "config is not the validation config"
    if prov["content_sha256"] != prov["manifest_content_sha256"]:
        return "mixture hash differs from its committed manifest"
    return None


def decide(
    first: Mapping[Key, Mapping[str, Any] | None],
    rerun: Mapping[Key, Mapping[str, Any] | None],
    commit: str,
) -> dict[str, Any]:
    """Apply the rule. ``first``: every (algo, seed); ``rerun``: the reruns made so far (a failed
    run is rerun once, from scratch, at the same commit). Action: ``"rerun"`` (list in
    ``to_rerun``), ``"investigate"`` or ``"proceed"``."""
    used: dict[Key, Mapping[str, Any]] = {}
    failed_twice, to_rerun, failures = [], [], {}
    for key in ((a, s) for s in SEEDS for a in ALGOS):
        why = failure(first.get(key), commit)
        if why is None:
            used[key] = first[key]
            continue
        failures[f"{key[0]}_s{key[1]}"] = {"first": why}
        if key not in rerun:
            to_rerun.append(key)
            continue
        why_again = failure(rerun[key], commit)
        failures[f"{key[0]}_s{key[1]}"]["rerun"] = why_again
        if why_again is None:
            used[key] = rerun[key]
        else:
            failed_twice.append(key)

    out: dict[str, Any] = {"v1_bar": V1_BAR, "gap": GAP, "commit": commit, "failures": failures}
    # Every finite score that exists, for reporting (D7); the rule reads only ``per_seed``.
    out["scores_reported"] = {
        f"{a}_s{s}_{attempt}": r["final"]["score"]
        for attempt, records in (("first", first), ("rerun", rerun))
        for (a, s), r in records.items()
        if r is not None and math.isfinite(r["final"]["score"])
    }
    if failed_twice:  # D6: a run that still fails after its rerun means investigate
        return {**out, "action": "investigate", "reason": f"failed after rerun: {failed_twice}"}
    if to_rerun:  # the rule is applied only when all six final scores exist
        return {**out, "action": "rerun", "to_rerun": to_rerun, "reason": "runs failed once"}

    per_seed = {}
    for s in SEEDS:
        ours, corl = used[("iql", s)]["final"]["score"], used[("corl_iql", s)]["final"]["score"]
        per_seed[f"s{s}"] = {
            "ours": ours,
            "corl": corl,
            "corl_minus_ours": corl - ours,
            "corl_falls_short": corl < V1_BAR,
            "ours_falls_short": ours < V1_BAR,  # reported, not part of the rule
            "ours_below_corl": corl - ours > GAP,
            "from_rerun": any(used[(a, s)] is rerun.get((a, s)) for a in ALGOS),
        }
    a = any(v["corl_falls_short"] for v in per_seed.values())
    b = sum(v["ours_below_corl"] for v in per_seed.values()) >= 2
    action = "investigate" if b else "proceed"  # D6: B alone decides; A is recorded
    reason = (
        "ours below CORL on at least 2 of 3 seeds: implementation difference, investigate before Phase 6"
        if b
        else "V1's failure is recorded as seed variance; Phase 5 proceeds"
    )
    return {**out, "per_seed": per_seed, "A": a, "B": b, "action": action, "reason": reason}
