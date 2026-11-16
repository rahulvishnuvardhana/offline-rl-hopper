"""configs/base.yaml score values must equal report/eda_summary.json (spec C5 config check).

Kept as a test, not a script check, so the EDA script never verifies its own output.
"""

import json
from pathlib import Path

import pytest

from offrl.config import load_config

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("r_random", "r_expert", "r_simple")


def test_config_score_equals_eda_summary():
    score = load_config(ROOT / "configs" / "base.yaml")["score"]
    summary_path = ROOT / "report" / "eda_summary.json"
    if not summary_path.exists():
        # Only the pre-EDA state may skip: no summary AND no values written yet.
        assert all(score[k] is None for k in KEYS), "score values set but report/eda_summary.json missing"
        pytest.skip("EDA not run yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["all_passed"] is True
    for key in KEYS:
        assert score[key] == pytest.approx(summary["score"][key], rel=1e-9), key
