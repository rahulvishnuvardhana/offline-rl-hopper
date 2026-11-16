"""The Trainer interface (ROADMAP 3.4). The training loop never knows which algorithm it runs."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from offrl.data.buffer import Batch


class Trainer(Protocol):
    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        """One gradient step. Returns scalar device tensors; the loop calls ``.item()`` only when
        it logs, so a GPU is not synchronized every step."""
        ...

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for **normalized** observations (the evaluator normalizes)."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, sd: dict[str, Any]) -> None: ...
