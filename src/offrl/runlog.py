"""Run logs: CSV files (the source of truth) and TensorBoard. Named ``runlog`` so it never
shadows the standard library's ``logging``."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CsvLog:
    """Append-only CSV with a fixed header, flushed after every row so a kill loses at most
    the row being written. ``truncate`` drops rows logged after a checkpoint."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path, self.fields = Path(path), list(fields)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", encoding="utf-8", newline="")  # noqa: SIM115 (closed in close())
        self._w = csv.DictWriter(self._f, fieldnames=self.fields, lineterminator="\n")
        if new:
            self._w.writeheader()
            self._f.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        self._w.writerow({k: _fmt(row[k]) for k in self.fields})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _fmt(v: Any) -> Any:
    return repr(v) if isinstance(v, float) else v  # repr: shortest exact round-trip


def truncate_csv(path: Path, max_step: int) -> int:
    """Keep the header and complete rows with ``step <= max_step``; return the rows kept.

    A half-written last line (a kill mid-write) has the wrong field count and is dropped.
    """
    path = Path(path)
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    header = lines[0]
    n_fields = len(header.split(","))
    step_col = header.split(",").index("step")
    kept = [
        line
        for line in lines[1:]
        if len(line.split(",")) == n_fields
        and line.split(",")[step_col].isdigit()
        and int(line.split(",")[step_col]) <= max_step
    ]
    path.write_text("\n".join([header, *kept]) + "\n", encoding="utf-8", newline="\n")
    return len(kept)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TbLog:
    """TensorBoard mirror of the CSV values. ``purge_step`` on resume removes events logged
    after the checkpoint, so curves never show a step twice."""

    def __init__(self, log_dir: Path, purge_step: int | None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._w = SummaryWriter(str(log_dir), purge_step=purge_step)

    def scalars(self, prefix: str, values: Mapping[str, float], step: int) -> None:
        for k, v in values.items():
            self._w.add_scalar(f"{prefix}/{k}", v, step)
        self._w.flush()

    def close(self) -> None:
        self._w.close()
