"""Check mixture files on disk against manifests: content and data hashes, p and seed.

Used after a mixture build whose manifests were copied out of the job (Phase 5, job 12014512):
    python scripts/check_mixture_files.py --manifests logs/phase5_seedmix_<id> \\
        --mixture-dir /scratch/$USER/mixtures
Checks every ``mix_*.json`` in ``--manifests`` and the audit beside them, if any. Read-only.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from offrl.data import mixture as mx


def check(manifest_dir: Path, mixture_dir: Path) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    manifests = sorted(manifest_dir.glob("mix_*.json"))
    if not manifests:
        return [(False, f"no mix_*.json in {manifest_dir}")]
    for path in manifests:
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
            arrays = mx.load_mixture(mixture_dir / f"{path.stem}.npz")
            content, data = mx.content_sha256(arrays), mx.data_sha256(arrays)
        except (OSError, ValueError, KeyError) as e:  # a missing file or column is a FAIL line
            results.append((False, f"{path.stem}: {type(e).__name__}: {e}"))
            continue
        ok = (
            content == m["content_sha256"]
            and data == m["data_sha256"]
            and path.stem == mx.mixture_name(m["p_target"], m["seed"])
        )
        results.append((ok, f"{path.stem}: file {content[:12]} vs manifest {m['content_sha256'][:12]}"))
    for path in sorted(manifest_dir.glob("audit_*.json")):
        a = json.loads(path.read_text(encoding="utf-8"))
        n = sum(c["passed"] for c in a["checks"])
        detail = f"{path.name}: {n}/{len(a['checks'])} checks, dirty {a['run']['git_dirty']}"
        results.append((a["all_passed"] is True and a["run"]["git_dirty"] is False, detail))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--mixture-dir", type=Path, required=True)
    args = parser.parse_args()
    results = check(args.manifests, args.mixture_dir)
    for ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {detail}")
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
