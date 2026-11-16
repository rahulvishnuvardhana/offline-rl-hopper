"""p-mixtures of the expert and simple datasets (spec: docs/specs/phase-3-mixtures.md).

A mixture is fully determined by ``(p, seed)``: the picks (which episodes, how many steps of
each) come from ``sample_mixture_episodes``; everything else is derived from the picks.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offrl.data.load import Episode

EXPERT, SIMPLE = "expert", "simple"
SOURCES = (EXPERT, SIMPLE)  # fixed order: the mixture RNG stream depends on it
SOURCE_CODE = {EXPERT: 1, SIMPLE: 0}  # value of the `source` column
N_TRANSITIONS = 900_000
P_FINAL = (0.0, 0.05, 0.2, 0.5, 1.0)  # F2

MODEL_KEYS = ("obs", "act", "rew", "next_obs", "done")  # float32, raw (unnormalized)
COLUMN_KEYS = ("source", "episode_id", "t")  # analysis columns

Picks = dict[str, list[tuple[int, int]]]  # source -> [(episode index, steps taken), ...]


def sample_mixture_episodes(
    lengths: Mapping[str, np.ndarray], p: float, n_transitions: int, seed: int
) -> Picks:
    """Per source, shuffle episodes with the seed and take whole episodes until its budget is
    met; the last one is trimmed. Returns (index, steps taken).

    One generator draws the expert permutation first, then the simple one, independently of
    ``p``: mixtures with the same seed are nested across ``p``.
    """
    rng = np.random.default_rng(seed)
    n_expert = int(round(p * n_transitions))
    budgets = {EXPERT: n_expert, SIMPLE: n_transitions - n_expert}
    out: Picks = {}
    for source in SOURCES:
        order = rng.permutation(len(lengths[source]))
        taken, remaining = [], budgets[source]
        for i in order:
            if remaining == 0:
                break
            n = min(int(lengths[source][i]), remaining)
            taken.append((int(i), n))
            remaining -= n
        if remaining:
            raise ValueError(f"{source}: budget {budgets[source]} exceeds the dataset")
        out[source] = taken
    return out


def mixture_name(p: float, seed: int) -> str:
    return f"mix_p{p:g}_s{seed}"


def flatten(episodes: Mapping[str, Sequence[Episode]], picks: Picks) -> dict[str, np.ndarray]:
    """Transitions inside each picked episode only, expert picks first, steps ascending.

    ``next_obs`` is the same episode's ``obs[t + 1]``; ``done`` is the true termination, so a
    trimmed episode's last row has ``done = 0`` and a real ``next_obs``.
    """
    parts: dict[str, list[np.ndarray]] = {k: [] for k in MODEL_KEYS + COLUMN_KEYS}
    for source in SOURCES:
        for i, n in picks[source]:
            ep = episodes[source][i]
            parts["obs"].append(ep.obs[:n])
            parts["act"].append(ep.act[:n])
            parts["rew"].append(ep.rew[:n])
            parts["next_obs"].append(ep.obs[1 : n + 1])
            parts["done"].append(ep.term[:n])
            parts["source"].append(np.full(n, SOURCE_CODE[source], dtype=np.int8))
            parts["episode_id"].append(np.full(n, ep.id, dtype=np.int32))
            parts["t"].append(np.arange(n, dtype=np.int32))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for k in MODEL_KEYS:
        arrays[k] = arrays[k].astype(np.float32)
    return arrays


def content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash of the array contents, independent of the file container: for each key in sorted
    order, its name, dtype, shape and C-order little-endian bytes. Compared across machines."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        digest.update(f"{key}|{a.dtype.str}|{a.shape}|".encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def data_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash of the rows and columns only (no float64 normalizer). If two machines disagree on
    ``content_sha256`` but agree here, the difference is in float summation, not in the data."""
    return content_sha256({k: arrays[k] for k in MODEL_KEYS + COLUMN_KEYS})


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def save_mixture(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write the .npz and return its file SHA-256 (meaningful on the same machine only)."""
    data = npz_bytes(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_mixture(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def picks_from_columns(source: np.ndarray, episode_id: np.ndarray, t: np.ndarray) -> Picks:
    """Re-derive the picks from a mixture's analysis columns (used to reconcile the manifest).

    Raises if an episode's rows are not a contiguous run starting at t = 0.
    """
    out: Picks = {s: [] for s in SOURCES}
    code_to_source = {v: k for k, v in SOURCE_CODE.items()}
    starts = np.flatnonzero(np.r_[True, (episode_id[1:] != episode_id[:-1]) | (source[1:] != source[:-1])])
    ends = np.r_[starts[1:], len(t)]
    for s, e in zip(starts, ends, strict=True):
        if not np.array_equal(t[s:e], np.arange(e - s)):
            raise ValueError(f"rows {s}:{e} are not steps 0..{e - s - 1} of one episode")
        out[code_to_source[int(source[s])]].append((int(episode_id[s]), int(e - s)))
    return out


@dataclass(frozen=True)
class EpisodeSummary:
    """Per-pick facts from the float64 sources (no float32 rounding)."""

    returns: np.ndarray  # return of each pick (trimmed picks at their trimmed return)
    whole: np.ndarray  # True where the pick is the full episode
    fell: np.ndarray  # True where the pick is whole and ended by termination


def episode_summary(episodes: Mapping[str, Sequence[Episode]], picks: Picks) -> EpisodeSummary:
    taken = [(episodes[s][i], n) for s in SOURCES for i, n in picks[s]]
    returns = np.array([float(ep.rew[:n].sum()) for ep, n in taken])
    whole = np.array([n == ep.length for ep, n in taken])
    fell = np.array([n == ep.length and bool(ep.term[-1]) for ep, n in taken])
    return EpisodeSummary(returns, whole, fell)
