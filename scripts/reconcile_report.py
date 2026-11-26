"""Check the report's numbers against the pre-registered analysis (read-only).

    python scripts/reconcile_report.py

Every Table 2 row, the H1 to H3 values and the other headline numbers in report/paper/main.tex are
recomputed from report/phase7/prereg_analysis.json and must appear in the paper as printed. Then every
decimal number in the README's Results section and in the poster must also appear in the paper. Two
things are left out of that cross-check on purpose: the README's other sections (development results,
cluster notes, software versions), which hold numbers that are not grid results, and the poster's layout
arguments (column and title widths such as ``\\column{0.50}`` or ``0.95\\linewidth``). Exits 1 on any
FAIL."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"bc": "BC", "iql tau=0.7": r"IQL $\tau{=}0.7$", "iql tau=0.5": r"IQL $\tau{=}0.5$",
         "iql tau=0.9": r"IQL $\tau{=}0.9$"}  # fmt: skip
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")


def f1(x: float) -> str:
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def num(x: float) -> str:
    s = f1(x)
    return f"$-{s[1:]}$" if s.startswith("-") else s


def readme_results(text: str) -> str:
    """The README's Results section, up to its reproduction instructions."""
    start = text.index("## Results")
    return text[start : text.index("### Reproduce the results", start)]


def poster_content(text: str) -> str:
    """The poster without layout arguments (column widths, title width)."""
    text = re.sub(r"\\column\{[\d.]+\}", "", text)
    text = re.sub(r"[\d.]+\\(?:linewidth|textwidth|columnwidth|paperwidth)", "", text)
    return text


def main() -> int:
    d = json.loads((ROOT / "report/phase7/prereg_analysis.json").read_text(encoding="utf-8"))
    paper = (ROOT / "report/paper/main.tex").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    poster = (ROOT / "report/poster/poster.tex").read_text(encoding="utf-8")
    rows: list[tuple[str, str, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        rows.append((name, "PASS" if cond else "FAIL", detail))

    for key, c in d["cells"].items():
        algo, p = re.match(r"(bc|iql tau=[\d.]+) p=([\d.]+)", key).groups()
        fall = statistics.mean(c["fall_rate_per_seed"])
        row = (
            f"{NAMES[algo]} & {p} & {f1(c['mean'])} & {f1(c['median'])} & "
            f"[{num(c['t_95'][0])}, {f1(c['t_95'][1])}] & {fall:.2f} & {len(c['high_fall_seeds'])} & "
            f"{len(c['collapsed_seeds'])} & {num(c['plateau'])} \\\\"
        )
        check(f"Table 2: {key}", row in paper, row)

    h1 = d["H1"]["comparison"]
    check("H1 mean +23.2", f"$+{f1(h1['mean'])}$" in paper)
    check("H1 per seed", ", ".join(f1(x) for x in h1["per_seed"]) in paper)
    check("H1 P 0.031", abs(h1["p_greater"] - 1 / 32) < 1e-9 and "0.031" in paper)
    check("H1 t-interval", f"[{f1(h1['t_95'][0])}, {f1(h1['t_95'][1])}]" in paper)
    check("H1 Spearman 0.46", f"{d['H1']['spearman_mean']:.2f}" == "0.46" and "0.46" in paper)
    h2 = d["H2"]["comparison"]
    check("H2 mean -5.7", f1(h2["mean"]) == "-5.7" and "$-5.7$" in paper)
    check(
        "H2 per seed", ", ".join(f1(x) for x in h2["per_seed"]).replace("-", "") == "4.4, 0.1, 2.2, 19.4, 2.3"
    )
    check(
        "H2 median gap 2.3",
        f1(abs(statistics.median(h2["per_seed"]))) == "2.3" and "median per-seed gap 2.3" in paper,
    )
    check(
        "H2 BC 105.4, IQL 99.7",
        f1(d["H2"]["bc_means"]["1.0"]) == "105.4" and f1(d["H2"]["iql_means"]["1.0"]) == "99.7",
    )
    h3 = d["H3"]["contrast"]
    check("H3 contrast -35.0", f1(h3["mean"]) == "-35.0" and "$-35.0$" in paper)
    check("H3 P 0.969 / 0.0625", f"{h3['p_greater']:.3f}" == "0.969" and h3["p_less"] == 0.0625)
    se = d["H3"]["simple_effects_tau0.9_minus_tau0.5"]
    check(
        "H3 simple effects -12.1 / +22.9",
        f1(se["0.05"]["mean"]) == "-12.1" and f1(se["0.5"]["mean"]) == "22.9",
    )
    for p, want in (("0.05", "-75.4"), ("0.2", "-54.3"), ("0.5", "-28.0")):
        check(f"IQL minus BC at p={p}", f1(d["iql_minus_bc_descriptive"][p]["mean"]) == want)
    check("IQL p=0 82.0", f1(d["cells"]["iql tau=0.7 p=0"]["mean"]) == "82.0" and "82.0 at $p=0$" in paper)
    check(
        "IQL p=0.05 fall 98%",
        round(statistics.mean(d["cells"]["iql tau=0.7 p=0.05"]["fall_rate_per_seed"]), 2) == 0.98,
    )
    check(
        "Plateau +5.4 / +6.7",
        f1(d["cells"]["iql tau=0.7 p=0.5"]["plateau"]) == "5.4"
        and f1(d["cells"]["iql tau=0.5 p=0.05"]["plateau"]) == "6.7"
        and "+5.4" in paper
        and "+6.7" in paper,
    )
    n_coll = sum(
        len(c["collapsed_seeds"]) for k, c in d["cells"].items() if "p=0.05" in k and k.startswith("iql")
    )
    check("11 of 15 collapsed at p=0.05", n_coll == 11 and "11 of 15" in paper, f"computed {n_coll}")

    for label, text in (("README Results", readme_results(readme)), ("poster", poster_content(poster))):
        missing = sorted(n for n in set(DECIMAL.findall(text)) if n not in paper)
        check(f"{label}: decimals all appear in the paper", not missing, ", ".join(missing))

    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{status}  {name:<{width}}  {detail if status == 'FAIL' else ''}".rstrip())
    fails = sum(r[1] == "FAIL" for r in rows)
    print(f"\n{len(rows) - fails} PASS, {fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
