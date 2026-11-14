from pathlib import Path

import pytest

from offrl.config import apply_override, deep_merge, load_config, save_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_deep_merge_recurses_and_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    update = {"a": {"y": 3}, "c": 4}
    merged = deep_merge(base, update)
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_override_parses_yaml_types():
    cfg = {"train": {"steps": 1, "lr": 0.1, "device": "auto", "flag": False}}
    for override in ["train.steps=2000", "train.lr=3e-4", "train.device=cuda", "train.flag=true"]:
        apply_override(cfg, override)
    assert cfg["train"] == {"steps": 2000, "lr": 3e-4, "device": "cuda", "flag": True}
    assert isinstance(cfg["train"]["lr"], float)  # PyYAML alone would return the string "3e-4"


@pytest.mark.parametrize("override", ["train.stpes=5", "trian.steps=5", "train.steps"])
def test_bad_override_fails_loudly(override):
    with pytest.raises((KeyError, ValueError)):
        apply_override({"train": {"steps": 1}}, override)


def test_iql_config_resolves_to_paper_setting():
    cfg = load_config(CONFIGS / "base.yaml", CONFIGS / "algo" / "iql.yaml")
    assert cfg["algo"]["name"] == "iql"
    assert cfg["algo"]["expectile"] == 0.7
    assert cfg["algo"]["beta"] == 3.0
    assert cfg["data"]["scale_rewards"] is True
    assert cfg["train"]["batch_size"] == 256


def test_bc_and_iql_share_everything_but_algo_and_reward_scaling():
    bc = load_config(CONFIGS / "base.yaml", CONFIGS / "algo" / "bc.yaml")
    iql = load_config(CONFIGS / "base.yaml", CONFIGS / "algo" / "iql.yaml")
    for section in ("net", "train", "eval", "score"):
        assert bc[section] == iql[section], section
    assert bc["algo"]["beta"] == 0.0


def test_save_and_reload_roundtrip(tmp_path):
    cfg = load_config(CONFIGS / "base.yaml", overrides=["data.p_expert=0.05"])
    save_config(cfg, tmp_path / "run" / "config.yaml")
    assert load_config(tmp_path / "run" / "config.yaml") == cfg
