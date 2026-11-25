"""Export evaluation-episode trajectories of every grid run for the project's story page (visual only).

    python scripts/export_trajectories.py --runs <archive>/runs --norms <grid_norms.npz> --out-dir <dir>
        [--seeds 10000 10001 10002 10003 10004]

For each of the 70 runs, its final policy (``ckpt/latest.pt``) is replayed on the stock Hopper-v5
model for each evaluation seed, exactly as in evaluation, and every episode must reproduce the run's
logged final evaluation for that seed (return, length and fall, ``eval_episodes.csv`` at the final
step); any mismatch stops the export. Each step stores, for the torso, thigh, leg and foot capsules,
the world x and z of the capsule centre and its pitch in the x-z plane (atan2, so it wraps at plus
or minus pi), plus the running return. One JSON file per run and an ``index.json`` (settings, run
score and per-episode outcome) are written; all 70 runs must be present and finished. Run it on the
CPU type the grid evaluated on (Explorer, cascadelake); each run's torch thread count is taken from
its ``meta.json``. Floating-point rounding differs between CPUs and thread counts, and a 1,000-step
hopping episode amplifies it.

``--norms`` holds each grid mixture's state_mean, state_std and state_std_eps (keys
``p<p>_s<seed>__<name>``), copied unchanged from the archived mixtures on Explorer; a wrong normalizer
would fail the replay check."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offrl.data.normalize import StateNormalizer  # noqa: E402
from offrl.evaluate import deterministic_policy, make_env  # noqa: E402
from offrl.train import TRAINERS, load_checkpoint  # noqa: E402
from record_demo import logged_episode, same, score  # noqa: E402

GEOMS = ("torso_geom", "thigh_geom", "leg_geom", "foot_geom")


def eval_threads(run_dir: Path) -> int:
    """The torch thread count the run actually trained and evaluated with (``meta.json``)."""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    counts = {meta["torch_threads"], *(r["torch_threads"] for r in meta["resumes"])}
    if len(counts) != 1:
        raise SystemExit(f"{run_dir.name}: thread count changed across resumes {sorted(counts)}")
    return counts.pop()


def episode(env, policy, seed: int, gids: list[int]):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    data = env.unwrapped.data
    total, steps, terminated, truncated, frames = 0.0, 0, False, False, []

    def snap():
        row = []
        for g in gids:
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
            row += [round(float(data.geom_xpos[g][0]), 3), round(float(data.geom_xpos[g][2]), 3)]
            row.append(round(math.atan2(float(axis[0]), float(axis[2])), 3))
        row.append(round(total, 1))
        frames.append(row)

    snap()
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += float(reward)
        steps += 1
        snap()
    return (total, steps, bool(terminated)), frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002, 10003, 10004])
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds has duplicates")
    norms = np.load(args.norms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "seeds": args.seeds,
        "geoms": list(GEOMS),
        "columns": "per geom: x, z, pitch; then return",
        "runs": {},
    }
    env = None
    for grid in ("main", "ablation"):
        for d in sorted(p for p in (args.runs / grid).iterdir() if p.is_dir()):
            if not (d / "final.json").exists():
                raise SystemExit(f"{d}: no final.json (unfinished or partly copied run)")
            cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
            key = f"p{cfg['data']['p_expert']:g}_s{cfg['seed']}"
            norm = StateNormalizer(
                norms[f"{key}__state_mean"].astype(np.float64),
                norms[f"{key}__state_std"].astype(np.float64),
                float(norms[f"{key}__state_std_eps"]),
            )
            state = load_checkpoint(d / "ckpt" / "latest.pt")
            if int(state["step"]) != cfg["train"]["steps"]:
                raise SystemExit(f"{d.name}: checkpoint at step {state['step']}")
            # as in evaluation: matmul rounding depends on the thread count
            torch.set_num_threads(eval_threads(d))
            trainer = TRAINERS[cfg["algo"]["name"]](cfg, 11, 3, torch.device("cpu"))
            trainer.load_state_dict(state["trainer"])
            policy = deterministic_policy(trainer, norm, torch.device("cpu"))
            if env is None:
                env = make_env(cfg["data"]["expert_id"])
                gids = [env.unwrapped.model.geom(n).id for n in GEOMS]
            episodes, outcomes = {}, []
            for seed in args.seeds:
                got, frames = episode(env, policy, seed, gids)
                want = logged_episode(d, cfg["train"]["steps"], seed)
                if not same(got, want):
                    raise SystemExit(f"{d.name} seed {seed}: replay {got}, logged {want}")
                episodes[str(seed)] = frames
                outcomes.append(
                    {"seed": seed, "score": round(score(got[0], cfg), 1), "steps": got[1], "fell": got[2]}
                )
            final = json.loads((d / "final.json").read_text(encoding="utf-8"))
            if isinstance(final["score"], str) or isinstance(final["fall_rate"], str):
                raise SystemExit(f"{d.name}: non-numeric final score or fall rate")
            run_score = final["score"]
            index["runs"][d.name] = {
                "algo": cfg["algo"]["name"],
                "tau": cfg["algo"].get("expectile"),
                "p": cfg["data"]["p_expert"],
                "seed": cfg["seed"],
                "run_score": round(run_score, 1),
                "fall_rate": round(final["fall_rate"], 3),
                "episodes": outcomes,
            }
            (args.out_dir / f"{d.name}.json").write_text(
                json.dumps(episodes, separators=(",", ":")), encoding="utf-8"
            )
            print(f"{d.name}: {len(args.seeds)} episodes match the log", flush=True)
    (args.out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    if len(index["runs"]) != 70:
        raise SystemExit(f"expected 70 runs (50 main, 20 ablation), found {len(index['runs'])}")
    print(f"wrote {len(index['runs'])} runs to {args.out_dir}")


if __name__ == "__main__":
    main()
