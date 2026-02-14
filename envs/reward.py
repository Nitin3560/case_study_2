from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RewardWeights:
    r_task: float = 100.0
    r_bytes: float = 0.02  # per byte delivered
    r_sense: float = 2.0   # per dwell step progress
    r_gen: float = 0.01    # per byte generated (scan/sense)
    w_aoi: float = 0.005   # AoI penalty weight (per step average AoI)
    w_energy: float = 0.0002
    k_task_prog: float = 0.1
    k_base_prog: float = 0.1
    shaping_clip: float = 0.2
    r_fail: float = -20.0


def team_reward(
    tasks_completed_now: int,
    bytes_delivered_now: int,
    mean_aoi_now: float,
    energy_used_now_j: float,
    shaping_now: float,
    bytes_generated_now: int,
    w: RewardWeights,
) -> float:
    r = 0.0
    r += w.r_task * float(tasks_completed_now)
    r += w.r_bytes * float(bytes_delivered_now)
    r += w.r_gen * float(bytes_generated_now)
    r -= w.w_aoi * float(mean_aoi_now)
    r -= w.w_energy * float(energy_used_now_j)
    r += float(shaping_now)
    return float(r)
