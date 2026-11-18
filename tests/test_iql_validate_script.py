"""scripts/iql_validate.py on a real tiny IQL run: analysis and figure run end to end."""

import importlib.util
import json

import gymnasium as gym
import pytest

from offrl import train
from offrl.data import mixture as mx

SCRIPT = train.REPO_ROOT / "scripts" / "iql_validate.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("iql_validate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_and_figure_on_a_tiny_run(small_run, script, tmp_path):
    make, manifests = small_run
    cfg = make(algo="iql")
    env = gym.make("Hopper-v5")
    run_dir = train.run(cfg, env=env, manifest_dir=manifests)
    env.close()
    paths = cfg["data"]["mixture_dir"], cfg["train"]["out_dir"]
    script.EXTRA.extend([f"data.mixture_dir={paths[0]}", f"train.out_dir={paths[1]}"])
    name = mx.mixture_name(1.0, 100)
    arrays = mx.load_mixture(train.resolve_path(cfg["data"]["mixture_dir"]) / f"{name}.npz")
    manifest = json.loads((manifests / f"{name}.json").read_text(encoding="utf-8"))
    mix = {"scaled_reward_min": 0.0, "scaled_reward_max": 1.0}

    out = script.analyze("iql", 1.0, 100, mix, arrays, manifest)

    assert out["run"] == run_dir.name and out["provenance"]["content_sha256"] == manifest["content_sha256"]
    assert set(out["v3"]) == {"bounded", "drift", "v_s0"}
    assert out["v3"]["drift"][0] is False  # a 600-step run has no 200k and 300k rows
    assert "V(s0)" in out["v3"]["v_s0"][1] and out["no_fall_return"]["n_episodes"] == 6
    json.dumps(out, allow_nan=False)  # the report must serialize

    results = [{**out, "algo": a, "p": p} for a in ("iql", "corl_iql") for p in (1.0, 0.2)]
    script.save(script.figure(results), tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0
