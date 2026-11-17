"""Training entry point (spec Section 6): pre-flight audit, loop, evaluation, checkpoint, resume.

    python -m offrl.train configs/algo/bc.yaml data.p_expert=1.0 seed=100
    python -m offrl.train configs/algo/bc.yaml configs/final.yaml seed=0   # final grid only

``run(cfg)`` is the importable form (Phase 6's sweep). A finished run (``final.json``) returns
immediately; an interrupted one resumes from ``ckpt/latest.pt`` only if its config and git
commit are unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from offrl.algos.bc import BCTrainer
from offrl.config import Config, load_checked, save_config
from offrl.data import mixture as mx
from offrl.data.buffer import ReplayBuffer
from offrl.data.leakage import COLLECTION_SEEDS
from offrl.data.normalize import beta_eff
from offrl.evaluate import deterministic_policy, evaluate, make_env
from offrl.rollout import FINAL_EVAL_RESERVED, FINAL_EVAL_SEED_OFFSET, eval_seeds
from offrl.runlog import CsvLog, TbLog, read_csv, truncate_csv
from offrl.runmeta import REPO_ROOT, collect_metadata
from offrl.seeding import derive_seeds, seed_everything

BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
MANIFEST_DIR = REPO_ROOT / "report" / "mixtures"
DEV_SEED_MIN = 100  # mixture seeds 0 to 99 are final-grid seeds; 100+ are development
FINAL_EVAL_SEEDS = range(FINAL_EVAL_SEED_OFFSET, FINAL_EVAL_SEED_OFFSET + 50)  # Phase 4 D1
TRAINERS = {"bc": BCTrainer}
EVAL_FIELDS = ["step", "n_episodes", "return_mean", "return_std", "score_mean", "fall_rate", "length_mean"]
EVAL_FIELDS += [f"sub_{k}" for k in EVAL_FIELDS[1:]]
EPISODE_FIELDS = ["step", "seed", "return", "length", "fell"]


class PreflightError(RuntimeError):
    pass


def run_name(cfg: Config) -> str:
    return f"{cfg['algo']['name']}_p{cfg['data']['p_expert']:g}_s{cfg['seed']}{cfg['train']['run_suffix']}"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def final_eval_steps(cfg: Config) -> list[int]:
    t, every, n = cfg["train"]["steps"], cfg["train"]["eval_every"], cfg["eval"]["final_score_last_n"]
    return [t - i * every for i in range(n - 1, -1, -1)]


def preflight(cfg: Config, arrays: dict[str, np.ndarray], manifest_dir: Path) -> dict[str, Any]:
    """Refuse to train on the wrong data, a dead config key, or the wrong evaluation seeds."""
    tr, ev, data, seed = cfg["train"], cfg["eval"], cfg["data"], cfg["seed"]
    name = mx.mixture_name(data["p_expert"], seed)
    errors: list[str] = []
    manifest_path = manifest_dir / f"{name}.json"
    content = mx.content_sha256(arrays)
    if not manifest_path.exists():
        errors.append(f"no committed manifest {manifest_path.name}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["content_sha256"] != content:
            errors.append(f"content_sha256 {content[:12]} != manifest {manifest['content_sha256'][:12]}")
        if manifest["p_target"] != data["p_expert"] or manifest["seed"] != seed:
            errors.append(f"manifest is p={manifest['p_target']} s={manifest['seed']}, config {name}")
    if not data["normalize_states"]:
        errors.append("data.normalize_states must be true (the buffer always normalizes)")
    eps = float(arrays["state_std_eps"])
    if eps != data["state_std_eps"]:
        errors.append(f"data.state_std_eps {data['state_std_eps']} != mixture {eps}")
    used = set(eval_seeds(ev)) | set(eval_seeds(ev, ev["final_episodes"]))
    if any(s in r for s in used for r in COLLECTION_SEEDS.values()):
        errors.append("evaluation seeds overlap the data collection seeds")
    if seed >= DEV_SEED_MIN and used & set(FINAL_EVAL_RESERVED):
        errors.append("development mixture with evaluation seeds reserved for the final grid")
    final_ok = used == set(FINAL_EVAL_SEEDS)
    if seed < DEV_SEED_MIN and not final_ok:
        errors.append("final-grid mixture without the final evaluation seeds (load configs/final.yaml)")
    if tr["steps"] % tr["eval_every"] or tr["steps"] % tr["checkpoint_every"]:
        errors.append("train.steps must be a multiple of train.eval_every and train.checkpoint_every")
    if tr["checkpoint_every"] % tr["log_every"] or tr["eval_every"] % tr["log_every"]:
        errors.append("checkpoint_every and eval_every must be multiples of log_every")
    if tr["steps"] < ev["final_score_last_n"] * tr["eval_every"]:
        errors.append("too few steps for the final-score evaluations")
    if errors:
        raise PreflightError("; ".join(errors))
    return {"mixture": name, "content_sha256": content, "passed": True}


# ---------- checkpoint ----------


def _rng_state() -> dict[str, Any]:
    """All global RNG states as tensors, lists and ints, so ``weights_only=True`` loads them."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()  # noqa: NPY002
    version, internal, gauss = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [kind, keys.tolist(), int(pos), int(has_gauss), float(cached)],
        "python": [version, list(internal), gauss],
    }


def _set_rng_state(s: dict[str, Any]) -> None:
    torch.set_rng_state(s["torch"])
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])
    kind, keys, pos, has_gauss, cached = s["numpy"]
    np.random.set_state((kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))  # noqa: NPY002
    version, internal, gauss = s["python"]
    random.setstate((version, tuple(internal), gauss))


def save_checkpoint(path: Path, step: int, trainer, buffer: ReplayBuffer) -> None:
    """Atomic: a kill mid-write leaves the previous checkpoint intact."""
    tmp = path.with_suffix(".pt.tmp")
    state = {
        "step": step,
        "trainer": trainer.state_dict(),
        "buffer": buffer.generator_state(),
        "rng": _rng_state(),
    }
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Loaded on the CPU: RNG and generator states must stay CPU tensors. ``load_state_dict``
    copies network and optimizer tensors onto their parameters' device."""
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------- run ----------


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def final_result(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Final score (pre-registered): mean of the last ``final_score_last_n`` evaluations, each over
    ``final_episodes`` episodes. Built from ``eval.csv`` so a crash before this point loses nothing."""
    steps = final_eval_steps(cfg)
    rows = {int(r["step"]): r for r in read_csv(run_dir / "eval.csv")}
    missing = [s for s in steps if s not in rows]
    if missing:
        raise RuntimeError(f"final evaluations missing at steps {missing}")
    sel = [rows[s] for s in steps]
    if any(int(r["n_episodes"]) != cfg["eval"]["final_episodes"] for r in sel):
        raise RuntimeError("a final evaluation does not have eval.final_episodes episodes")

    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in sel]))

    return {
        "run": run_name(cfg),
        "score": mean("score_mean"),
        "return_mean": mean("return_mean"),
        "fall_rate": mean("fall_rate"),
        "sub_score": mean("sub_score_mean"),
        "steps": steps,
        "per_eval": [{k: float(r[k]) for k in ("score_mean", "return_mean", "fall_rate")} for r in sel],
        "episodes_per_eval": cfg["eval"]["final_episodes"],
        "note": "mean of correlated checkpoints on the same starting states, not independent samples",
    }


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(cfg: Config, *, stop_at: int | None = None, env=None, manifest_dir: Path = MANIFEST_DIR) -> Path:
    """Train one run to ``train.steps`` (or resume it). ``stop_at`` returns after that step, as a
    kill would: anything logged after the last checkpoint is discarded on resume. ``env``
    replaces the recovered env (tests)."""
    tr, ev = cfg["train"], cfg["eval"]
    run_dir = resolve_path(tr["out_dir"]) / run_name(cfg)
    if (run_dir / "final.json").exists():
        return run_dir

    name = mx.mixture_name(cfg["data"]["p_expert"], cfg["seed"])
    path = resolve_path(cfg["data"]["mixture_dir"]) / f"{name}.npz"
    if not path.exists():
        raise PreflightError(f"mixture file {path} not found (scripts/build_mixtures.py)")
    arrays = mx.load_mixture(path)
    audit = preflight(cfg, arrays, manifest_dir)

    meta_now = collect_metadata()
    ckpt_path = run_dir / "ckpt" / "latest.pt"
    resuming = ckpt_path.exists()
    if (run_dir / "config.yaml").exists():
        import yaml

        saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if saved != cfg:
            raise PreflightError(f"{run_dir} holds a run with a different config; use a new run_suffix")
    if resuming:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["git_commit"] != meta_now["git_commit"]:
            raise PreflightError(f"{run_dir} was started at commit {meta['git_commit']}; code has changed")

    if tr["num_threads"]:
        torch.set_num_threads(int(tr["num_threads"]))
    device = pick_device(tr["device"])
    if resuming:  # another device or thread count breaks the bitwise continuation
        now = {"device": str(device), "torch_threads": torch.get_num_threads()}
        was = {k: meta[k] for k in now}
        if was != now:
            raise PreflightError(f"{run_dir} was started with {was}, now {now}; resume on the same setup")
    init_seed, buffer_seed, _diag_seed = derive_seeds(cfg["seed"])
    seed_everything(init_seed)
    buffer = ReplayBuffer.from_mixture(arrays, cfg["data"]["scale_rewards"], device, buffer_seed)
    obs_dim, act_dim = buffer.obs.shape[1], buffer.act.shape[1]
    trainer = TRAINERS[cfg["algo"]["name"]](cfg, obs_dim, act_dim, device)

    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    start = 0
    if resuming:
        state = load_checkpoint(ckpt_path)
        start = int(state["step"])
        trainer.load_state_dict(state["trainer"])
        buffer.set_generator_state(state["buffer"])
        _set_rng_state(state["rng"])
        for name in ("metrics.csv", "eval.csv", "eval_episodes.csv", "timing.csv"):
            truncate_csv(run_dir / name, start)
        meta["resumes"].append({"from_step": start, **meta_now})
    else:
        for name in ("metrics.csv", "eval.csv", "eval_episodes.csv", "timing.csv"):
            (run_dir / name).unlink(missing_ok=True)  # a run killed before its first checkpoint
        shutil.rmtree(run_dir / "tb", ignore_errors=True)
        save_config(cfg, run_dir / "config.yaml")
        scale = float(arrays["reward_scale"])
        meta = {
            **meta_now,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
            "seeds": {"run": cfg["seed"], "init": init_seed, "buffer": buffer_seed},
            "preflight": audit,
            "reward_scale": scale,
            "beta_eff": beta_eff(float(cfg["algo"].get("beta", 0.0)), scale),
            "resumes": [],
        }
    write_json(run_dir / "meta.json", meta)

    if start == tr["steps"]:  # killed after the last checkpoint, before final.json
        write_json(run_dir / "final.json", final_result(cfg, run_dir))
        return run_dir

    log_keys = list(trainer.LOG_KEYS)
    metrics = CsvLog(run_dir / "metrics.csv", ["step", *log_keys])
    evals = CsvLog(run_dir / "eval.csv", EVAL_FIELDS)
    episodes = CsvLog(run_dir / "eval_episodes.csv", EPISODE_FIELDS)
    timing = CsvLog(run_dir / "timing.csv", ["step", "kind", "seconds"])
    tb = TbLog(run_dir / "tb", purge_step=start + 1 if resuming else None)
    if env is None:
        env = make_env(cfg["data"]["expert_id"])
    policy_fn = deterministic_policy(trainer, buffer.normalizer, device)
    final_steps = set(final_eval_steps(cfg))

    sums = {k: torch.zeros((), device=device) for k in log_keys}
    t0 = time.perf_counter()
    try:
        for step in range(start + 1, tr["steps"] + 1):
            out = trainer.update(buffer.sample(tr["batch_size"]))
            for k in log_keys:
                sums[k] += out[k].to(device)
            if step % tr["log_every"] == 0:
                row = {k: float(v.item()) / tr["log_every"] for k, v in sums.items()}
                metrics.write({"step": step, **row})
                tb.scalars("train", row, step)
                timing.write({"step": step, "kind": "train", "seconds": time.perf_counter() - t0})
                sums = {k: torch.zeros((), device=device) for k in log_keys}
                t0 = time.perf_counter()
            if step % tr["eval_every"] == 0:
                t_eval = time.perf_counter()
                n = ev["final_episodes"] if step in final_steps else ev["episodes"]
                summary, rows = evaluate(env, policy_fn, cfg, n)
                evals.write({"step": step, **summary})
                for r in rows:
                    episodes.write({"step": step, **r})
                tb.scalars("eval", summary, step)
                timing.write({"step": step, "kind": "eval", "seconds": time.perf_counter() - t_eval})
                t0 = time.perf_counter()
            if step % tr["checkpoint_every"] == 0:
                save_checkpoint(ckpt_path, step, trainer, buffer)
            if step == stop_at:
                return run_dir
    finally:
        for log in (metrics, evals, episodes, timing, tb):
            log.close()

    write_json(run_dir / "final.json", final_result(cfg, run_dir))
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("args", nargs="+", help="algo YAML file(s), then key.sub=value overrides")
    args = parser.parse_args(argv).args
    files = [a for a in args if "=" not in a]
    overrides = [a for a in args if "=" in a]
    cfg = load_checked(BASE_CONFIG, *files, overrides=overrides)
    run_dir = run(cfg)
    print(json.dumps(json.loads((run_dir / "final.json").read_text(encoding="utf-8")), indent=2))
    return run_dir


if __name__ == "__main__":
    main()
