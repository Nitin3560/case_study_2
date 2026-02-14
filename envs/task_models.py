from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    tid: int
    pos_xy: np.ndarray  # (2,)
    appears_step: int
    active: bool = False
    completed: bool = False
    delivered: bool = False
    dwell_left: int = 0

    # AoI stream semantics: once active, generates periodic updates
    last_generated_step: int | None = None
    last_received_step: int | None = None  # at base

    def maybe_activate(self, step: int) -> None:
        if (not self.active) and (step >= self.appears_step):
            self.active = True

    def within_visit_radius(self, agent_xy: np.ndarray, r: float) -> bool:
        return float(np.linalg.norm(agent_xy - self.pos_xy)) <= float(r)
