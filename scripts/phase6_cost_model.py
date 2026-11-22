"""Phase 6 spec Section 5: apply the pre-fixed choice rule to the benchmark logs (the D1 evidence).

Usage: python choice_rule.py <dir with offrl-bench-*.out logs> <dir with archived 12034721 iql runs>
Prints the per-configuration model inputs, the 12034721 validation, and the grid wall clock for
each candidate pair (T4, CPU; V100 ineligible: no kernels for sm_70 in torch 2.11+cu128).
"""

import math
import re
import sys
from pathlib import Path

from offrl.runlog import read_csv

N = {"bc": 25, "iql": 45}  # 70 runs: main 25 BC + 25 IQL, ablation 20 IQL
UPDATES, EVAL_STEPS = 300_000, 800 * 1_000

logs = Path(sys.argv[1])
cfg = {}
for f in sorted(logs.glob("offrl-bench-*.out")):
    text = f.read_text(errors="replace")
    m = re.search(r"model inputs \(slowest run\): u = ([\d.]+) ms, e = ([\d.]+) ms", text)
    c = re.search(r"config: algo=(\w+) k=(\d+) device=(\w+) cpus=(\d+) mem=(\d+)M", text)
    q = re.search(r"QUEUE_WAIT_S=(-?\d+)", text)
    model = re.search(r"model recorded by every run: (.+)", text)
    if not (m and c):
        print(f"{f.name}: no result (failed or incomplete)")
        continue
    algo, k, device, cpus, mem = c.group(1), int(c.group(2)), c.group(3), int(c.group(4)), int(c.group(5))
    u, e = float(m.group(1)) / 1000, float(m.group(2)) / 1000
    h = (UPDATES * u + EVAL_STEPS * e) / 3600
    name = f.stem.split("_")[0].replace("offrl-bench-", "")
    cfg[name] = dict(
        algo=algo,
        k=k,
        device=device,
        cpus=cpus,
        mem_gb=mem / 1024,
        u=u,
        e=e,
        h=h,
        queue_s=int(q.group(1)) if q else None,
        model=model.group(1) if model else "?",
    )
    print(
        f"{name:4s} {algo:4s} k={k} {device:4s} cpus={cpus:2d} u={1000 * u:6.2f} ms e={1000 * e:6.3f} ms "
        f"h={h:5.2f} h queue={cfg[name]['queue_s']} s model={cfg[name]['model']}"
    )

# Validation (spec 5): G2's u and e predict 12034721's "ours" runs (IQL, k = 2, T4) within 20% each.
arch = Path(sys.argv[2])
if "G2" in cfg:
    for run in sorted(arch.glob("iql_p1_s10[234]")):
        rows = read_csv(run / "timing.csv")
        train_s = sum(float(r["seconds"]) for r in rows if r["kind"] == "train")
        eval_s = sum(float(r["seconds"]) for r in rows if r["kind"] == "eval")
        steps = sum(int(r["length"]) for r in read_csv(run / "eval_episodes.csv"))
        pt, pe = UPDATES * cfg["G2"]["u"], steps * cfg["G2"]["e"]
        print(
            f"validate {run.name}: train {train_s:.0f} s vs predicted {pt:.0f} s "
            f"({100 * (pt / train_s - 1):+.0f}%); eval {eval_s:.0f} s vs predicted {pe:.0f} s "
            f"({100 * (pe / eval_s - 1):+.0f}%) over {steps} env steps"
        )

# J_eff per spec 5: min(running limit, throttle, free devices, node cores / cores per job, node mem / mem).
T4_NODE = dict(cores=28, mem_gb=187, gpus=4)
CPU_IDLE_CORES = 90  # lowest reading: the CPU benchmark jobs, 2026-11-21 23:34 (98 at 22:30)


def jeff_t4(c):
    return min(4, 4, T4_NODE["gpus"], T4_NODE["cores"] // c["cpus"], int(T4_NODE["mem_gb"] // c["mem_gb"]))


def jeff_cpu(c):
    return min(50, 25, CPU_IDLE_CORES // c["cpus"])


def wall(pair, jeff, max_submit=None, wave_latency_h=0.0, queue_h=0.0):
    total, tasks = 0.0, 0
    for algo, name in pair.items():
        c = cfg[name]
        j = jeff(c)
        n_tasks = math.ceil(N[algo] / c["k"])
        tasks += n_tasks
        total += math.ceil(n_tasks / j) * (c["h"] + queue_h)
    waves = math.ceil(tasks / max_submit) if max_submit else 1
    return total + (waves - 1) * wave_latency_h, tasks, waves


cands = []
for iql in ("G1", "G2", "G4"):
    if iql in cfg and "GB" in cfg:
        cands.append(("T4", {"bc": "GB", "iql": iql}, jeff_t4, 8))
if "C1" in cfg and "CB" in cfg:
    cands.append(("CPU cascadelake", {"bc": "CB", "iql": "C1"}, jeff_cpu, None))
for label, pair, jeff, ms in cands:
    limit_h = 8.0 if label == "T4" else 48.0  # gpu partition 8 h, short 2 days
    ok = all(cfg[n]["h"] * 1.5 <= limit_h for n in pair.values())
    qh = max((cfg[n]["queue_s"] or 0) for n in pair.values()) / 3600 if label.startswith("CPU") else 0.0
    for lat in (0.0, 1.0):
        w, tasks, waves = wall(pair, jeff, ms, lat, qh)
        gpu_h = (
            0.0
            if label.startswith("CPU")
            else sum(math.ceil(N[a] / cfg[n]["k"]) * cfg[n]["h"] for a, n in pair.items())
        )
        print(
            f"{label:16s} {pair} eligible={ok} J_eff={[jeff(cfg[n]) for n in pair.values()]} tasks={tasks} "
            f"waves={waves} wave_latency={lat:.0f} h queue/wave={qh:.2f} h "
            f"-> wall {w:.1f} h, GPU-hours {gpu_h:.1f}"
        )
