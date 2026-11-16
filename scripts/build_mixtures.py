"""Phase 3: build p-mixtures, check them (M1 to M9), write manifests, a summary and an audit.

Spec: docs/specs/phase-3-mixtures.md. Usage:
    python scripts/build_mixtures.py                    # final grid: p in P_FINAL, seeds 0 1 2
    python scripts/build_mixtures.py --seeds 100        # development mixtures
    python scripts/build_mixtures.py --out-dir /scratch/$USER/mixtures

Outputs: {out_dir}/mix_p{p}_s{seed}.npz, report/mixtures/mix_p{p}_s{seed}.json (manifests),
report/mixtures/summary_{seeds}.json and audit_{seeds}.json.
Provenance order (as Phase 2): commit the code, run on the clean tree, commit report/mixtures/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from offrl.analysis.eda import PREVIEW_SEEDS, normalized_score
from offrl.config import load_config
from offrl.data import mixture as mx
from offrl.data.leakage import COLLECTION_SEED_OFFSET
from offrl.data.load import Episode, load_episodes
from offrl.data.mixture import EXPERT, SIMPLE, SOURCES
from offrl.data.mixture_audit import SourceIndex, build_mixture, check_mixture, saco_cells
from offrl.data.normalize import beta_eff
from offrl.data.validate import Check
from offrl.provenance import committed_bytes, dirty_paths
from offrl.runmeta import collect_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "report" / "mixtures"
AUDIT = ROOT / "report" / "data_audit.json"
EDA_SUMMARY = ROOT / "report" / "eda_summary.json"
SACO_BINS = (0.5, 1.0)
REPLAY_EPISODES = 3
REPLAY_STEPS = 100


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def replay_check(ids: dict[str, str], episodes: dict[str, list[Episode]]) -> dict[str, Any]:
    """M9 (informational, never blocks): replay episodes from their collection seed."""
    try:
        import minari
        import mujoco

        env = minari.load_dataset(ids[EXPERT]).recover_environment()
        worst = {}
        try:
            for s in SOURCES:
                err = 0.0
                for i in range(REPLAY_EPISODES):
                    ep = episodes[s][i]
                    obs, _ = env.reset(seed=COLLECTION_SEED_OFFSET + i)
                    err = max(err, float(np.abs(obs - ep.obs[0]).max()))
                    for t in range(min(REPLAY_STEPS, ep.length)):
                        obs, r, *_ = env.step(ep.act[t])
                        err = max(
                            err, float(np.abs(obs - ep.obs[t + 1]).max()), abs(float(r) - float(ep.rew[t]))
                        )
                worst[s] = err
        finally:
            env.close()
        return {
            "max_abs_error": worst,
            "steps": REPLAY_STEPS,
            "episodes_per_source": REPLAY_EPISODES,
            "mujoco": mujoco.__version__,
            "within_1e-9": all(v <= 1e-9 for v in worst.values()),
        }
    except Exception as err:  # informational: record, never block the build
        return {"error": f"{type(err).__name__}: {err}"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--p", type=float, nargs="+", default=list(mx.P_FINAL))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=None, help="default: data.mixture_dir from base.yaml")
    parser.add_argument("--allow-dirty", action="store_true", help="development only; marks the run dirty")
    args = parser.parse_args()

    dirty = dirty_paths()
    if dirty and not args.allow_dirty:
        print(f"REFUSING: uncommitted changes: {dirty}\nCommit the code first (M8).")
        return 1

    cfg = load_config(ROOT / "configs" / "base.yaml")
    beta = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "algo" / "iql.yaml")["algo"]["beta"]
    eps = cfg["data"]["state_std_eps"]
    out_dir = args.out_dir or ROOT / cfg["data"]["mixture_dir"]
    ids = {EXPERT: cfg["data"]["expert_id"], SIMPLE: cfg["data"]["simple_id"]}
    audit_blob, eda_blob = committed_bytes(AUDIT), committed_bytes(EDA_SUMMARY)
    audit, eda = json.loads(audit_blob), json.loads(eda_blob)
    phase2 = {row["p"]: row["per_seed"] for row in eda["mixture_reference"]}
    if any(len(v) != len(PREVIEW_SEEDS) for v in phase2.values()):
        raise ValueError("eda_summary per_seed lists do not match PREVIEW_SEEDS")
    run = collect_metadata()
    episodes = {s: load_episodes(ids[s]) for s in SOURCES}
    index = {s: SourceIndex(episodes[s]) for s in SOURCES}
    score_ref = cfg["score"]

    checks: list[Check] = []
    summaries: dict[str, dict[str, Any]] = {}
    hashes: dict[tuple[float, int], str] = {}
    for seed in args.seeds:
        manifests: dict[float, dict[str, Any]] = {}
        saco_inputs: dict[float, dict[str, np.ndarray]] = {}
        for p in args.p:
            name = mx.mixture_name(p, seed)
            arrays, picks = build_mixture(episodes, p, seed, mx.N_TRANSITIONS, eps)
            file_sha = mx.save_mixture(out_dir / f"{name}.npz", arrays)
            content_sha = mx.content_sha256(arrays)

            ref = phase2[p][PREVIEW_SEEDS.index(seed)] if p in phase2 and seed in PREVIEW_SEEDS else None
            checks += check_mixture(name, p, arrays, picks, index, episodes, mx.N_TRANSITIONS, eps, ref)

            again, _ = build_mixture(episodes, p, seed, mx.N_TRANSITIONS, eps)  # M7: rebuild from scratch
            same = mx.content_sha256(again) == content_sha
            same &= hashlib.sha256(mx.npz_bytes(again)).hexdigest() == file_sha
            checks.append(Check(f"M7_rebuild_identical[{name}]", same, f"content {content_sha[:12]}"))
            hashes[(p, seed)] = content_sha

            summ = mx.episode_summary(episodes, picks)
            scale = float(arrays["reward_scale"])
            whole_scores = [
                normalized_score(r, score_ref["r_random"], score_ref["r_expert"])
                for r in summ.returns[summ.whole]
            ]
            summary = {
                "p_target": p,
                "p_actual": int((arrays["source"] == 1).sum()) / mx.N_TRANSITIONS,
                "n_transitions": mx.N_TRANSITIONS,
                "seed": seed,
                "n_episodes": int(len(summ.returns)),
                "n_trimmed": int((~summ.whole).sum()),
                "mean_return": float(summ.returns.mean()),
                "score": normalized_score(
                    float(summ.returns.mean()), score_ref["r_random"], score_ref["r_expert"]
                ),
                "fall_rate_untrimmed": float(summ.fell[summ.whole].mean()),
                "tq": float(np.mean(whole_scores)),
                "reward_scale": scale,
                "beta_eff": beta_eff(beta, scale),
            }
            manifests[p] = {
                **summary,
                "picks": {s: [list(pk) for pk in picks[s]] for s in SOURCES},
                "source_versions": ids,
                "source_sha256": {s: audit["datasets"][ids[s]]["file_sha256"] for s in SOURCES},
                "file_sha256": file_sha,
                "content_sha256": content_sha,
                "data_sha256": mx.data_sha256(arrays),
                "state_mean": arrays["state_mean"].tolist(),
                "state_std": arrays["state_std"].tolist(),
                "state_std_eps": eps,
                "git_commit": run["git_commit"],
                "packages": run["packages"],
            }
            saco_inputs[p] = {"obs": arrays["obs"], "act": arrays["act"]}
            print(
                f"  built {name}: content {content_sha[:12]}, scale {scale:.4f}, score {summary['score']:.1f}"
            )

        if 0.0 in saco_inputs:  # SACo proxy against the seed's p = 0 reference normalizer
            ref_x = np.concatenate([saco_inputs[0.0]["obs"], saco_inputs[0.0]["act"]], axis=1).astype(
                np.float64
            )
            ref_mean, ref_std = ref_x.mean(0), ref_x.std(0)
            base = {w: saco_cells(saco_inputs[0.0], ref_mean, ref_std, w) for w in SACO_BINS}
            for p, x in saco_inputs.items():
                manifests[p]["saco"] = {
                    f"bin_{w:g}": saco_cells(x, ref_mean, ref_std, w) / base[w] for w in SACO_BINS
                }
        for p, manifest in manifests.items():
            name = mx.mixture_name(p, seed)
            write_json(OUT_REPORT / f"{name}.json", manifest)
            summaries[name] = {k: manifest[k] for k in (*summary, "saco") if k in manifest}

        seed_hashes = [hashes[(p, seed)] for p in args.p]
        checks.append(
            Check(
                f"M7_distinct_within_seed[s{seed}]",
                len(set(seed_hashes)) == len(seed_hashes),
                f"{len(seed_hashes)} mixtures (guard)",
            )
        )
    if len(args.seeds) > 1:
        for p in args.p:
            across = [hashes[(p, s)] for s in args.seeds]
            checks.append(
                Check(
                    f"M7_distinct_across_seeds[p{p:g}]",
                    len(set(across)) == len(across),
                    f"{len(across)} seeds (guard)",
                )
            )

    checks.append(
        Check(
            "M8_clean_tree",
            not dirty and run["git_dirty"] is False,
            f"git_dirty={run['git_dirty']} at commit {run['git_commit']}"
            + (f"; --allow-dirty with changes: {dirty}" if dirty else ""),
        )
    )
    same = AUDIT.read_bytes() == audit_blob and EDA_SUMMARY.read_bytes() == eda_blob
    checks.append(
        Check(
            "M8b_inputs_on_disk_equal_commit", same, "byte-identical to HEAD" if same else "differ from HEAD"
        )
    )
    replay = replay_check(ids, episodes)
    all_passed = all(c.passed for c in checks)

    tag = "_".join(f"s{s}" for s in args.seeds)
    write_json(OUT_REPORT / f"summary_{tag}.json", summaries)
    write_json(
        OUT_REPORT / f"audit_{tag}.json",
        {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "run": run,
            "inputs_sha256_of_committed_blob": {
                "report/data_audit.json": hashlib.sha256(audit_blob).hexdigest(),
                "report/eda_summary.json": hashlib.sha256(eda_blob).hexdigest(),
            },
            "p": args.p,
            "seeds": args.seeds,
            "content_sha256": {mx.mixture_name(p, s): h for (p, s), h in hashes.items()},
            "M9_replay_informational": replay,
            "checks": [c.as_dict() for c in checks],
            "all_passed": all_passed,
        },
    )
    for c in checks:
        if not c.passed or c.name.startswith(("M7_distinct", "M8")):
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    n_fail = sum(not c.passed for c in checks)
    print(f"  {len(checks) - n_fail}/{len(checks)} checks passed; M9 (informational): {replay}")
    verdict = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"\n{verdict}; audit written to {OUT_REPORT / f'audit_{tag}.json'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
