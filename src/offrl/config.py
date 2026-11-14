"""YAML config loading: base <- algo file(s) <- CLI overrides.

Overrides use dotted keys, e.g. ``train.steps=2000`` or ``data.p_expert=0.05``.
An override for a key that does not exist raises, so typos fail loudly instead
of silently running the default.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Config:
    """Return a new dict: ``update`` merged into ``base``, recursing into nested dicts."""
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: Config, override: str) -> None:
    """Apply one ``a.b.c=value`` override in place. The key path must already exist."""
    if "=" not in override:
        raise ValueError(f"override must look like key=value, got {override!r}")
    dotted, raw = override.split("=", 1)
    *parents, leaf = dotted.strip().split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"unknown config section {part!r} in override {override!r}")
        node = node[part]
    if leaf not in node:
        raise KeyError(f"unknown config key {dotted!r}")
    node[leaf] = _parse_value(raw)


def _parse_value(raw: str) -> Any:
    """YAML-parse a CLI value. PyYAML (YAML 1.1) reads ``3e-4`` as a string, so retry as float."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(*paths: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Merge YAML files left to right, then apply CLI overrides."""
    cfg: Config = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for override in overrides:
        apply_override(cfg, override)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the fully resolved config next to a run's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
