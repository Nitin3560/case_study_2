from __future__ import annotations
import numpy as np


def in_range(p_xy: np.ndarray, q_xy: np.ndarray, rng: float) -> bool:
    return float(np.linalg.norm(p_xy - q_xy)) <= float(rng)


def pick_best_neighbor(
    agent_idx: int,
    agent_pos: np.ndarray,
    neighbors: list[int],
    base_xy: np.ndarray,
) -> int | None:
    """
    Simple heuristic: choose neighbor that is closest to base.
    """
    if not neighbors:
        return None
    best = None
    best_dist = 1e18
    for j in neighbors:
        d = float(np.linalg.norm(agent_pos[j] - base_xy))
        if d < best_dist:
            best_dist = d
            best = j
    return best