from __future__ import annotations
import numpy as np


def move_energy_j(move_cost_j_per_m: float, delta_xy: np.ndarray) -> float:
    return float(move_cost_j_per_m) * float(np.linalg.norm(delta_xy))


def sensing_energy_j(sense_cost_j_per_step: float, sensing_steps: int) -> float:
    return float(sense_cost_j_per_step) * float(sensing_steps)


def tx_energy_j(tx_cost_j_per_byte: float, bytes_sent: int) -> float:
    return float(tx_cost_j_per_byte) * float(bytes_sent)