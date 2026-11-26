"""Build the A0 landscape poster as HTML; its charts are drawn here from the pre-registered analysis.

    python scripts/build_poster.py            # writes report/poster/poster.html

Then print it to report/poster/poster.pdf with Chromium at 1189 x 841 mm, backgrounds on (the page sets
``@page`` to that size, so any Chromium "Save as PDF" or Playwright ``page.pdf(prefer_css_page_size=True)``
gives the same layout). Every chart is vector SVG built from report/phase7/prereg_analysis.json and
report/phase7/learning_curves.json; the text's numbers are the report's (scripts/reconcile_report.py checks
them). The page layout and text are report/poster/poster_template.html. The robot stills are frames of
the story page's median-run replays (runs/demo/showcase, evaluation seed 10000, logged episodes replayed
exactly); assets/qr_story.svg encodes the public story page URL."""

from __future__ import annotations

import json
import random
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/poster/poster.html"
P_GRID = ["0", "0.05", "0.2", "0.5", "1"]
P_LABEL = {"0": "0%", "0.05": "5%", "0.2": "20%", "0.5": "50%", "1": "100%"}
BC, IQL, IQL_LO, IQL_HI = "#2a78d6", "#eb6834", "#f29a72", "#a8441a"
INK, INK2, INK3, GRID, PANEL = "#171a17", "#4c524b", "#7a8078", "#e6e8e2", "#ffffff"
MONO = "'IBM Plex Mono', 'IBM Plex Sans', Consolas, monospace"


def f1(x: float) -> str:
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s.replace("-", "−")


class Svg:
    def __init__(self, w: int, h: int, label: str):
        self.w, self.h, self.parts = w, h, []
        self.label = label

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=26, anchor="middle", color=INK2, weight=400, family=MONO, halo=False):
        if halo:  # a panel-coloured box behind the label; stroked text would print as a Type 3 font
            w = 0.6 * size * len(s) + 12  # IBM Plex Mono advances 0.6 em per character
            x0 = {"start": x - 6, "middle": x - w / 2, "end": x - w + 6}[anchor]
            self.add(
                f'<rect x="{x0:.1f}" y="{y - 0.82 * size:.1f}" width="{w:.1f}" height="{1.04 * size:.1f}" '
                f'rx="4" fill="{PANEL}"/>'
            )
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}" font-family="{family}">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, color, width=2, dash="", cap="butt", opacity=1.0) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="{cap}" opacity="{opacity}"{d}/>'
        )

    def __str__(self) -> str:
        body = "".join(self.parts)
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" role="img" aria-label="{escape(self.label)}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
        )


def polyline(pts: str, color: str, width: float, opacity: float = 1.0) -> str:
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" opacity="{opacity}"/>'
    )


def y_axis(svg: Svg, y, x0, x1, ticks, size=26) -> None:
    for t in ticks:
        svg.line(x0, y(t), x1, y(t), GRID, 2)
        svg.text(x0 - 16, y(t) + size * 0.35, str(t), size=size, anchor="end", color=INK3)


def marker(svg: Svg, x, y, color, shape, r) -> None:
    if shape == "o":
        svg.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{PANEL}" stroke-width="4"/>'
        )
    elif shape == "s":
        svg.add(
            f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{color}" '
            f'stroke="{PANEL}" stroke-width="4"/>'
        )
    else:
        svg.add(
            f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{color}" '
            f'stroke="{PANEL}" stroke-width="4" transform="rotate(45 {x:.1f} {y:.1f})"/>'
        )


def cell_marks(svg: Svg, x, y, c, color, shape, rng, dot=9, bar=16, r=17) -> None:
    """Seed dots, t whisker, bootstrap bar, mean marker (as in the report's Figure 1)."""
    for s in c["per_seed"]:
        cx = x + rng.uniform(-22, 22)
        svg.add(f'<circle cx="{cx:.1f}" cy="{y(s):.1f}" r="{dot}" fill="{color}" opacity="0.32"/>')
    svg.line(x, y(c["t_95"][0]), x, y(c["t_95"][1]), color, 4)
    svg.line(x, y(c["bootstrap_95"][0]), x, y(c["bootstrap_95"][1]), color, bar)
    marker(svg, x, y(c["mean"]), color, shape, r)


def main_chart(cells: dict) -> str:
    w, h, ml, mr, mt, mb = 1660, 960, 100, 70, 30, 118
    svg = Svg(w, h, "Final normalized score versus expert share for BC and IQL")
    y = lambda v: mt + (h - mt - mb) * (1 - (v + 12) / 127)  # noqa: E731
    x = lambda i: ml + 100 + i * (w - ml - mr - 200) / 4  # noqa: E731
    y_axis(svg, y, ml, w - mr, [0, 20, 40, 60, 80, 100], size=30)
    svg.line(ml, y(90), w - mr, y(90), INK3, 3, dash="14 10")
    svg.text((x(0) + x(1)) / 2, y(90) - 14, "near-expert (90)", size=28, color=INK3, halo=True)
    rng = random.Random(0)
    for key, color, shape, off in (("bc", BC, "o", -28), ("iql tau=0.7", IQL, "s", 28)):
        cs = [cells[f"{key} p={p}"] for p in P_GRID]
        pts = " ".join(f"{x(i) + off:.1f},{y(c['mean']):.1f}" for i, c in enumerate(cs))
        svg.add(polyline(pts, color, 7))
        for i, c in enumerate(cs):
            cell_marks(svg, x(i) + off, y, c, color, shape, rng)
    for i, p in enumerate(P_GRID):  # mean labels: BC above its line, IQL beside its marker
        b, q = cells[f"bc p={p}"]["mean"], cells[f"iql tau=0.7 p={p}"]["mean"]
        svg.text(x(i) - 28, y(b) - 30, f1(b), size=31, color=BC, weight=600, halo=True)
        if p == "0":
            svg.text(x(i) + 28, y(q) + 58, f1(q), size=31, color=IQL, weight=600, halo=True)
        elif p == "1":
            svg.text(x(i) + 58, y(q) + 40, f1(q), size=31, color=IQL, weight=600, anchor="start", halo=True)
        else:
            svg.text(x(i) + 58, y(q) + 11, f1(q), size=31, color=IQL, weight=600, anchor="start", halo=True)
        svg.text(x(i), h - mb + 50, P_LABEL[p], size=34, color=INK, weight=500)
        svg.text(x(i), h - mb + 88, f"p = {p}", size=26, color=INK3)
    svg.text(x(2), y(cells["bc p=0.2"]["mean"]) + 66, "BC", size=42, color=BC, weight=600, halo=True)
    svg.text(x(1) + 34, y(64), "IQL (τ = 0.7)", size=42, color=IQL, weight=600, anchor="start", halo=True)
    return str(svg)


def tau_chart(cells: dict) -> str:
    w, h, ml, mr, mt, mb = 1000, 420, 86, 20, 60, 66
    svg = Svg(w, h, "IQL score by expectile tau at 5% and 50% expert share")
    y = lambda v: mt + (h - mt - mb) * (1 - (v + 12) / 112)  # noqa: E731
    y_axis(svg, y, ml, w - mr, [0, 50, 100], size=24)
    svg.line(ml, y(90), w - mr, y(90), INK3, 3, dash="12 9")
    rng = random.Random(1)
    groups = ["0.05", "0.5"]
    taus = (("0.5", IQL_LO), ("0.7", IQL), ("0.9", IQL_HI))
    for g, p in enumerate(groups):
        cx = ml + (w - ml - mr) * (g + 0.5) / 2
        svg.text(cx, h - mb + 46, f"{P_LABEL[p]} expert", size=26, color=INK, weight=500)
        for k, (tau, col) in enumerate(taus):
            xx = cx + (k - 1) * 120
            cell_marks(svg, xx, y, cells[f"iql tau={tau} p={p}"], col, "d", rng, dot=7, bar=12, r=13)
    for k, (tau, col) in enumerate(taus):
        lx = ml + 20 + k * 250
        marker(svg, lx, 26, col, "d", 11)
        svg.text(lx + 24, 35, f"τ = {tau}", size=26, color=INK2, anchor="start")
    return str(svg)


def curve_panels(runs: dict) -> str:
    w, h, ml, mr, mt, mb = 340, 300, 58, 10, 50, 46
    out = []
    for p in P_GRID:
        svg = Svg(w, h, f"Learning curves at {P_LABEL[p]} expert")
        y = lambda v: mt + (h - mt - mb) * (1 - (v + 5) / 120)  # noqa: E731
        x = lambda s: ml + (w - ml - mr) * s / 300000  # noqa: E731
        y_axis(svg, y, ml, w - mr, [0, 50, 100], size=22)
        svg.line(ml, y(90), w - mr, y(90), INK3, 2, dash="8 6")
        svg.text(ml, 30, f"{P_LABEL[p]} expert", size=26, color=INK, weight=600, anchor="start")
        for algo, color in (("bc", BC), ("iql", IQL)):
            sel = [r for r in runs.values() if r["grid"] == "main" and r["algo"] == algo]
            sel = [r for r in sel if f"{r['p']:g}" == p]
            steps = sel[0]["step"]
            for r in sel:
                pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in zip(steps, r["score"], strict=True))
                svg.add(polyline(pts, color, 1, opacity=0.28))
            mean = [sum(r["score"][i] for r in sel) / len(sel) for i in range(len(steps))]
            pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in zip(steps, mean, strict=True))
            svg.add(polyline(pts, color, 3.2))
        for s in (0, 150000, 300000):
            svg.text(x(s), h - mb + 32, f"{s // 1000}k", size=22, color=INK3, anchor="end" if s else "start")
        out.append(str(svg))
    return "".join(f'<div class="curve">{s}</div>' for s in out)


TEMPLATE = ROOT / "report/poster/poster_template.html"

DIAL = [("0", 0, 0), ("0.05", 5, 3), ("0.2", 20, 12), ("0.5", 50, 36), ("1", 100, 100)]


def on(width: int) -> str:
    """White label on a nearly full bar."""
    return " class=on" if width > 80 else ""


def dial_rows() -> str:
    rows = []
    for p, moves, starts in DIAL:
        rows.append(
            f'<span class="p">{P_LABEL[p]}</span>'
            f'<div class="bar"><i style="width:{moves}%"></i><em{on(moves)}>{moves}%</em></div>'
            f'<div class="bar starts"><i style="width:{starts}%"></i><em{on(starts)}>{starts}%</em></div>'
        )
    return "".join(rows)


def main() -> None:
    d = json.loads((ROOT / "report/phase7/prereg_analysis.json").read_text(encoding="utf-8"))
    runs = json.loads((ROOT / "report/phase7/learning_curves.json").read_text(encoding="utf-8"))["runs"]
    cells = d["cells"]
    qr = (ROOT / "report/poster/assets/qr_story.svg").read_text(encoding="utf-8")
    qr = qr.replace('width="330" height="330"', 'viewBox="0 0 33 33"').replace('transform="scale(10)" ', "")
    diff = d["iql_minus_bc_descriptive"]
    fall = sum(cells["iql tau=0.7 p=0.05"]["fall_rate_per_seed"]) / 5
    values = {
        "bc005": f1(cells["bc p=0.05"]["mean"]),
        "iql005": f1(cells["iql tau=0.7 p=0.05"]["mean"]),
        "iqlfall": f"{round(fall * 100)}",
        "d005": f1(diff["0.05"]["mean"]),
        "d02": f1(diff["0.2"]["mean"]),
        "d05": f1(diff["0.5"]["mean"]),
        "qr": qr,
        "dial": dial_rows(),
        "main_chart": main_chart(cells),
        "tau_chart": tau_chart(cells),
        "curves": curve_panels(runs),
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for k, v in values.items():
        html = html.replace("{" + k + "}", v)
    OUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
