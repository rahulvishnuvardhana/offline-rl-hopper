"""Phase 5, EXPLORATORY: same-weights CPU check, our IQL vs. CORL's (spec Section 8, investigation).

The seed follow-up rule said proceed (report/phase5/seed_followup.json); the user asked for this
check anyway, to tell an implementation difference from run-to-run sensitivity. It does not change
the decision. Two starts on the ``p = 1``, seed 103 mixture, float64 on one CPU thread:

- ``init``: our initial weights for seed 103 (drawn in float32 as in train.run, then cast to
  float64) copied into CORL, Adam empty, schedule at step 0;
- ``late``: CORL s103 at 200k (D7), weights, target, Adam state and schedule position loaded into
  both implementations; batches continue the checkpoint's buffer generator.

Each start runs three trainers on identical batches: ours, CORL's, and ours with every nonzero
weight moved once by a relative 2^-24 in a random direction (the size of one float32 rounding;
the late weights come from a float32 checkpoint, so rounding itself would change nothing; zero
weights, such as log_std at init, stay put). ours vs. CORL measures the implementation
difference (``bitwise_equal`` per row). ours vs. perturbed measures how fast a one-time
rounding-sized difference grows: a rough proxy for how GPU runs of one seed drift apart, since
those differ by fresh rounding at every step; ``first_step_at`` gives the first recorded step
(resolution ``--every``) at which it crosses 1e-6, 1e-3 and 1e-1. The two arms measure
different things and are not read as sizes against each other.
CORL's Q target multiplies by ``(1 - terminals.float())``, which rounds the discount to float32
before it meets float64 values; ours gets the same float32-rounded discount here, as it
effectively has in the float32 training runs, or the harness itself would create a gap.
Bitwise agreement in float64 on the CPU shows the same arithmetic, not that GPU float32
kernels round alike. Writes
report/phase5/same_weights_exploratory.json. On the laptop CPU (spec Section 8); no evaluation:
    python scripts/same_weights.py --ckpt <corl_iql_p1_s103 latest.pt at 200k> [--steps 5000]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from offrl import train
from offrl.algos.corl import CorlTrainer
from offrl.algos.iql import IQLTrainer
from offrl.config import Config, load_checked
from offrl.data import mixture as mx
from offrl.data.buffer import Batch, ReplayBuffer
from offrl.runmeta import collect_metadata
from offrl.seeding import derive_seeds, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "phase5" / "same_weights_exploratory.json"
SEED, P, LATE_STEP = 103, 1.0, 200_000
N_STATES = 2048  # fixed buffer states on which the policies' actions are compared
THRESHOLDS = (1e-6, 1e-3, 1e-1)


def config(algo: str) -> Config:
    return load_checked(
        train.BASE_CONFIG,
        ROOT / "configs" / "algo" / f"{algo}.yaml",
        overrides=[f"seed={SEED}", f"data.p_expert={P}", "train.device=cpu"],
    )


def corl_name(ours: str) -> str:
    """Our parameter name -> CORL's (as tests/test_corl.py): CORL's MLP wraps its layers in ``.net``."""
    head, _, rest = ours.partition(".")
    return ours if head == "log_std" else f"{head}.net.{rest}"


def pairs(t: IQLTrainer, c: CorlTrainer) -> list[tuple[str, torch.nn.Module, torch.nn.Module]]:
    i = c.iql
    return [
        ("policy", t.policy, i.actor),
        ("q", t.q, i.qf),
        ("q_target", t.q_target, i.q_target),
        ("v", t.v, i.vf),
    ]


def copy_corl_into_ours(t: IQLTrainer, c: CorlTrainer, with_optim: bool) -> None:
    """Weights (and, for the late start, Adam state and schedule) from CORL into ours, by name."""
    for _, ours, theirs in pairs(t, c):
        dst, src = dict(ours.named_parameters()), dict(theirs.named_parameters())
        assert {corl_name(k) for k in dst} == set(src), (sorted(dst), sorted(src))
        with torch.no_grad():
            for k, p in dst.items():
                p.copy_(src[corl_name(k)])
    if not with_optim:
        return
    i = c.iql
    for opt, theirs_opt, net, theirs_net in (
        (t.opt, i.actor_optimizer, t.policy, i.actor),
        (t.q_opt, i.q_optimizer, t.q, i.qf),
        (t.v_opt, i.v_optimizer, t.v, i.vf),
    ):
        src = dict(theirs_net.named_parameters())
        for k, p in net.named_parameters():
            state = theirs_opt.state[src[corl_name(k)]]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}, sorted(state)
            opt.state[p] = {name: v.clone() for name, v in state.items()}
        opt.param_groups[0]["lr"] = theirs_opt.param_groups[0]["lr"]
    t.sched.load_state_dict(i.actor_lr_schedule.state_dict())


def corl_from_ours(t: IQLTrainer, cfg: Config, obs_dim: int, act_dim: int) -> CorlTrainer:
    """The ``init`` start: CORL built by the adapter, then our weights copied in."""
    c = CorlTrainer(cfg, obs_dim, act_dim, torch.device("cpu"))
    for net in (c.iql.qf, c.iql.q_target, c.iql.vf, c.iql.actor):
        net.double()  # in place, so the optimizers still hold these parameters; log_std was float32
    for _, ours, theirs in pairs(t, c):
        src = dict(ours.named_parameters())
        dst = dict(theirs.named_parameters())
        assert {corl_name(k) for k in src} == set(dst)
        with torch.no_grad():
            for k, p in src.items():
                dst[corl_name(k)].copy_(p)
    return c


def perturbed(t: IQLTrainer) -> IQLTrainer:
    """A copy of ``t`` with every weight times ``1 +/- 2^-24`` (random sign, fixed seed); the
    optimizer state is untouched."""
    r = copy.deepcopy(t)
    gen = torch.Generator().manual_seed(1)
    for net in (r.policy, r.q, r.q_target, r.v):
        with torch.no_grad():
            for p in net.parameters():
                sign = torch.randint(0, 2, p.shape, generator=gen).to(p.dtype) * 2 - 1
                p.mul_(1 + sign * 2.0**-24)
    return r


def diff(a: torch.nn.Module, b: torch.nn.Module, map_names: bool) -> dict[str, float]:
    """Max absolute difference and relative L2 difference over all parameters of two networks."""
    other = dict(b.named_parameters())
    num = den = mx_abs = 0.0
    for k, p in a.named_parameters():
        q = other[corl_name(k) if map_names else k]
        d = (p - q).detach()
        if not torch.isfinite(d).all():
            raise FloatingPointError(f"non-finite difference in {k}")
        mx_abs = max(mx_abs, d.abs().max().item())
        num += d.pow(2).sum().item()
        den += q.detach().pow(2).sum().item()
    return {"max_abs": mx_abs, "rel_l2": (num / den) ** 0.5 if den else 0.0}


def as64(b: Batch) -> Batch:
    return Batch(*(x.double() for x in (b.obs, b.act, b.rew, b.next_obs, b.done)))


def run_start(
    name: str,
    t: IQLTrainer,
    c: CorlTrainer,
    buffer: ReplayBuffer,
    states: torch.Tensor,
    steps: int,
    every: int,
) -> dict[str, Any]:
    r = perturbed(t)
    rows, t0 = [], time.perf_counter()

    def record(step: int, logs: dict[str, dict[str, torch.Tensor]] | None) -> None:
        with torch.no_grad():
            a_t, a_c, a_r = t.act(states), c.act(states), r.act(states)
        row: dict[str, Any] = {"step": step}
        for label, mapped in (("ours_vs_corl", True), ("ours_vs_perturbed", False)):
            nets = (
                pairs(t, c)
                if mapped
                else [
                    (n, getattr(t, a), getattr(r, a))
                    for n, a in (("policy", "policy"), ("q", "q"), ("q_target", "q_target"), ("v", "v"))
                ]
            )
            row[label] = {n: diff(ours, theirs, mapped) for n, ours, theirs in nets}
            row[label]["action_max_abs"] = (a_t - (a_c if mapped else a_r)).abs().max().item()
            row[label]["bitwise_equal"] = row[label]["action_max_abs"] == 0 and all(
                row[label][n]["max_abs"] == 0 for n, _, _ in nets
            )
        if logs is not None:
            row["losses"] = {
                k: {who: float(v[k]) for who, v in logs.items()} for k in ("v_loss", "q_loss", "policy_loss")
            }
            row["actor_lr"] = {who: float(v["actor_lr"]) for who, v in logs.items()}
        rows.append(row)

    record(0, None)
    for step in range(1, steps + 1):
        b = as64(buffer.sample(256))
        logs = {"ours": t.update(b), "corl": c.update(b)}
        with warnings.catch_warnings():  # the deep copy drops the scheduler's step wrapper: noise
            warnings.filterwarnings("ignore", message="Seems like", category=UserWarning)
            logs["perturbed"] = r.update(b)
        if step % every == 0:
            record(step, logs)
            last = rows[-1]
            print(
                f"[{name}] step {step}: ours-CORL policy rel {last['ours_vs_corl']['policy']['rel_l2']:.2e} "
                f"act {last['ours_vs_corl']['action_max_abs']:.2e}; ours-perturbed policy rel "
                f"{last['ours_vs_perturbed']['policy']['rel_l2']:.2e} "
                f"act {last['ours_vs_perturbed']['action_max_abs']:.2e}; "
                f"{(time.perf_counter() - t0) / step * 1000:.1f} ms/step",
                flush=True,
            )
    first: dict[str, dict[str, dict[str, int | None]]] = {}
    for label in ("ours_vs_corl", "ours_vs_perturbed"):
        first[label] = {
            metric: {
                f"{th:g}": next((row["step"] for row in rows if get(row[label]) >= th), None)
                for th in THRESHOLDS
            }
            for metric, get in (
                ("policy_rel_l2", lambda d: d["policy"]["rel_l2"]),
                ("action_max_abs", lambda d: d["action_max_abs"]),
            )
        }
    return {
        "first_step_at": first,
        "resolution_steps": every,
        "bitwise_equal_throughout": all(row["ours_vs_corl"]["bitwise_equal"] for row in rows),
        "rows": rows,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="CORL s103 checkpoint at 200k (D7)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--starts", nargs="+", default=["init", "late"], choices=["init", "late"])
    args = parser.parse_args()

    torch.set_num_threads(1)
    ours_cfg, corl_cfg = config("iql"), config("corl_iql")
    path = train.resolve_path(ours_cfg["data"]["mixture_dir"]) / f"{mx.mixture_name(P, SEED)}.npz"
    arrays = mx.load_mixture(path)
    audit = train.preflight(ours_cfg, arrays, train.MANIFEST_DIR)  # refuses a mixture off its manifest
    init_seed, buffer_seed, _ = derive_seeds(SEED)
    buffer = ReplayBuffer.from_mixture(arrays, True, "cpu", buffer_seed)  # float32, as in training
    del arrays
    obs_dim, act_dim = buffer.obs.shape[1], buffer.act.shape[1]
    states = buffer.obs[
        torch.randperm(len(buffer), generator=torch.Generator().manual_seed(0))[:N_STATES]
    ].double()

    ckpt = train.load_checkpoint(Path(args.ckpt))
    assert int(ckpt["step"]) == LATE_STEP and ckpt["trainer"]["total_it"] == LATE_STEP, ckpt["step"]

    results: dict[str, Any] = {}
    for start in args.starts:
        torch.set_default_dtype(torch.float32)
        seed_everything(init_seed)  # as train.run: built right after seeding, drawn in float32
        t = IQLTrainer(ours_cfg, obs_dim, act_dim, torch.device("cpu"))
        for net in (t.policy, t.q, t.q_target, t.v):
            net.double()  # in place: the optimizers keep these parameters
        torch.set_default_dtype(torch.float64)
        t.discount = torch.tensor(t.discount, dtype=torch.float32).item()  # as CORL (docstring)
        if start == "init":
            buffer.set_generator_state(torch.Generator().manual_seed(buffer_seed).get_state())
            c = corl_from_ours(t, corl_cfg, obs_dim, act_dim)
        else:
            c = CorlTrainer(corl_cfg, obs_dim, act_dim, torch.device("cpu"))
            for net in (c.iql.qf, c.iql.q_target, c.iql.vf, c.iql.actor):
                net.double()
            c.load_state_dict(ckpt["trainer"])  # Adam state is cast to the (float64) parameters
            copy_corl_into_ours(t, c, with_optim=True)
            buffer.set_generator_state(ckpt["buffer"])
            assert t.sched.last_epoch == c.iql.actor_lr_schedule.last_epoch == LATE_STEP
        results[start] = run_start(start, t, c, buffer, states, args.steps, args.every)

    report = {
        "EXPLORATORY": "not part of the seed follow-up rule; does not change the Phase 5 decision",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": "docs/specs/phase-5-iql.md Section 8 (investigation step, run at the user's request)",
        "run": collect_metadata(),
        "setup": {
            "seed": SEED,
            "p": P,
            "mixture": audit,
            "dtype": "float64",
            "discount_ours": "rounded to float32, as CORL's",
            "threads": 1,
            "ckpt": Path(args.ckpt).name,
            "late_step": LATE_STEP,
            "steps": args.steps,
            "every": args.every,
            "n_states": N_STATES,
        },
        "starts": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"report written to {OUT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
