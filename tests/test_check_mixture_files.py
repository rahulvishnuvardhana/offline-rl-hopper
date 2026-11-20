"""scripts/check_mixture_files.py: mixture files on disk against copied manifests and audits."""

import importlib.util
import json

import numpy as np

from offrl import train
from offrl.data import mixture as mx

SCRIPT = train.REPO_ROOT / "scripts" / "check_mixture_files.py"


def load_script():
    spec = importlib.util.spec_from_file_location("check_mixture_files", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hashes_names_and_audit(make_mixture, tmp_path):
    script = load_script()
    mix_dir, man_dir = tmp_path / "mix", tmp_path / "man"
    mix_dir.mkdir()
    man_dir.mkdir()

    def mixture(seed):  # the synthetic mixture plus the analysis columns a built one has
        a = make_mixture(seed=seed)
        n = len(a["rew"])
        a["source"], a["episode_id"] = np.zeros(n, dtype=np.int8), np.arange(n, dtype=np.int32) // 100
        return a

    arrays = mixture(1)
    name = mx.mixture_name(1.0, 102)
    mx.save_mixture(mix_dir / f"{name}.npz", arrays)
    manifest = {
        "p_target": 1.0,
        "seed": 102,
        "content_sha256": mx.content_sha256(arrays),
        "data_sha256": mx.data_sha256(arrays),
    }
    (man_dir / f"{name}.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = {"all_passed": True, "run": {"git_dirty": False}, "checks": [{"passed": True}]}
    (man_dir / "audit_s102.json").write_text(json.dumps(audit), encoding="utf-8")
    assert all(ok for ok, _ in script.check(man_dir, mix_dir))

    # Another seed's data under this name, a dirty audit, and an empty folder each fail.
    mx.save_mixture(mix_dir / f"{name}.npz", mixture(2))
    assert [ok for ok, _ in script.check(man_dir, mix_dir)] == [False, True]
    audit["run"]["git_dirty"] = True
    (man_dir / "audit_s102.json").write_text(json.dumps(audit), encoding="utf-8")
    assert [ok for ok, _ in script.check(man_dir, mix_dir)] == [False, False]
    assert not script.check(tmp_path / "missing", mix_dir)[0][0]
