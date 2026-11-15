"""Validate both Hopper datasets and write the audit trail to report/data_audit.json.

Usage:
    python scripts/validate_data.py            # exit code 1 if any check fails
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from offrl.data.load import dataset_metadata, file_hashes, load_episodes
from offrl.data.validate import EXPECTED, run_all
from offrl.runmeta import collect_metadata

OUT = Path(__file__).resolve().parents[1] / "report" / "data_audit.json"


def main() -> int:
    audit = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": collect_metadata(),
        "datasets": {},
    }
    all_passed = True
    for dataset_id, documented in EXPECTED.items():
        episodes = load_episodes(dataset_id, download=True)
        metadata = dataset_metadata(dataset_id)
        checks = run_all(episodes, documented, metadata)
        all_passed &= all(c.passed for c in checks)

        print(f"\n{dataset_id}")
        for c in checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")

        audit["datasets"][dataset_id] = {
            "documented": documented,
            "metadata": metadata,
            "file_sha256": file_hashes(dataset_id),
            "checks": [c.as_dict() for c in checks],
        }

    audit["all_passed"] = all_passed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}; audit written to {OUT}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
