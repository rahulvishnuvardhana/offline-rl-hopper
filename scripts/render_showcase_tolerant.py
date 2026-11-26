"""Run render_showcase.py with its log check relaxed, for episodes the laptop cannot replay exactly.
Visual only; it never trains, evaluates or writes a result.

    python scripts/render_showcase_tolerant.py <render_showcase.py arguments>

render_showcase.py requires the laptop replay of each policy to equal the run's logged final
evaluation (return within 1e-6 relative, same length, same fall). Long episodes break that: the grid
ran on Explorer CPUs, and floating-point differences between CPU types grow over hundreds of steps,
so on the laptop no IQL episode that the logs show falling between steps 550 and 750 replays exactly
(all 25 such episodes were tried). This wrapper accepts a replay with the same fall outcome and a
length within TOLERANCE steps of the log, prints the logged and replayed episodes, and leaves every
other check unchanged; the rendered episode must still equal the replay (that comparison is exact in
practice, since both run on the same machine). Any video made with it must say it is a re-render and
give both lengths.

README's second demo: ``--run <main>/bc_p0.05_s4 --norm norm_p0.05_s4.npz --name bc --run
<main>/iql_p0.05_s2 --norm norm_p0.05_s2.npz --name iql --seed 10000``, where ``<main>`` is the
archived grid's ``runs/main`` folder and the normalizers are copied from the archived mixtures as in
record_demo.py. Logged: BC 1,000 steps, no fall (exact match); IQL falls at step 613. Laptop
re-render: IQL falls at step 622 (return 2306.3 vs. 2273.0 logged).

The two clips were then joined with ffmpeg: each padded with a 72 px label bar (``pad=800:672:0:72``)
carrying its method, run and a step counter (``drawtext`` with ``%{eif:min(n,1000):d}`` for BC and
``min(n,622)`` for IQL), a "FELL AT STEP 622" badge from frame 622 and a "COMPLETES 1,000 STEPS"
badge from frame 1000, then ``hstack``, ``tpad=stop_mode=clone:stop_duration=1.5``, ``fps=62.5``,
``scale=1280:-2``, x264 at CRF 26. The README's animated WebP is that MP4 at ``fps=15,scale=640:-1``,
``libwebp_anim -q:v 60``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_showcase  # noqa: E402

TOLERANCE = 15  # steps; the README's second demo needs 9 (622 vs. 613)
exact = render_showcase.same


def tolerant(a: tuple[float, int, bool], b: tuple[float, int, bool]) -> bool:
    if exact(a, b):
        return True
    ok = a[2] == b[2] and abs(a[1] - b[1]) <= TOLERANCE
    print(f"  not exact: {a} vs. {b} (return, steps, fell); {'accepted' if ok else 'rejected'}")
    return ok


if __name__ == "__main__":
    render_showcase.same = tolerant
    render_showcase.main()
