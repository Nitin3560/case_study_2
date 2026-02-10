from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RewardWeights:
    r_task: float = 50.0
    r_bytes: float = 0.002  # per byte delivered
    w_aoi: float = 0.01     # AoI penalty weight (per step average AoI)
    w_energy: float = 0.0005
    r_fail: float = -20.0


def team_reward(
    tasks_completed_now: int,
    bytes_delivered_now: int,
    mean_aoi_now: float,
    energy_used_now_j: float,
    w: RewardWeights,
) -> float:
    r = 0.0
    r += w.r_task * float(tasks_completed_now)
    r += w.r_bytes * float(bytes_delivered_now)
    r -= w.w_aoi * float(mean_aoi_now)
    r -= w.w_energy * float(energy_used_now_j)
    return float(r)