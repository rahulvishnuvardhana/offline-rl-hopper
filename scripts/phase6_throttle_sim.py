"""Makespan of the three Phase 6 arrays under per-array throttles and a cluster cap C.
Durations from the D1 benchmark (report/compute.md): IQL 1.00 h, BC 0.22 h per 300k-step run.
Slurm model: when a slot frees, the earliest-submitted array with pending tasks and running < its
throttle starts its next task (FIFO by job id); C = concurrent jobs the free cores allow."""

import heapq
import itertools

H = {"iql": 1.00, "bc": 0.22}
ARRAYS = [("main_iql", "iql", 25), ("abl_iql", "iql", 20), ("main_bc", "bc", 25)]


def makespan(caps, C, order):
    arrays = [ARRAYS[i] for i in order]
    pending = {name: n for name, _, n in arrays}
    running = {name: 0 for name, _, _ in arrays}
    done_at = {}
    events, t, free = [], 0.0, C
    while True:
        started = True
        while started and free:
            started = False
            for name, algo, _ in arrays:
                if pending[name] and running[name] < caps[name]:
                    pending[name] -= 1
                    running[name] += 1
                    free -= 1
                    heapq.heappush(events, (t + H[algo], name))
                    started = True
                    break
        if not events:
            return t, done_at
        t, name = heapq.heappop(events)
        running[name] -= 1
        free += 1
        if not pending[name] and not running[name]:
            done_at[name] = t


SPLITS = {
    "10/8/7 (current)": (10, 8, 7),
    "12/10/3 (work-proportional)": (12, 10, 3),
    "13/10/2": (13, 10, 2),
    "11/9/5": (11, 9, 5),
    "9/9/7": (9, 9, 7),
}
print("work bound = 50.5 run-hours / C")
for C in (25, 22, 20, 16):
    print(f"\nC = {C} concurrent jobs (lower bound {50.5 / C:.2f} h)")
    for label, (a, b, c) in SPLITS.items():
        caps = {"main_iql": a, "abl_iql": b, "main_bc": c}
        res = []
        for order in itertools.permutations(range(3)):
            m, d = makespan(caps, C, order)
            res.append(m)
        m_bc_first, d = makespan(caps, C, (2, 0, 1))
        print(
            f"  {label:28s} makespan {min(res):.2f} to {max(res):.2f} h over the 6 submission orders;"
            f" BC submitted first: {m_bc_first:.2f} h (BC done {d['main_bc']:.2f} h)"
        )

print("\n10/8/7 by submission order (0 main IQL, 1 ablation IQL, 2 BC):")
caps = {"main_iql": 10, "abl_iql": 8, "main_bc": 7}
for order in itertools.permutations(range(3)):
    print(" ", order, "  ".join(f"C={C}: {makespan(caps, C, order)[0]:.2f} h" for C in (25, 22, 20, 16)))
