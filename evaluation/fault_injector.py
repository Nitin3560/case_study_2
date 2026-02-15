from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FaultConfig:
    num_agents: int
    horizon_steps: int
    fault_type: str
    t0: int
    duration: int
    severity: float = 0.5
    seed: int = 1


class FaultInjector:
    """Runtime fault schedule for Case Study 2 evaluation only.

    Fault types:
    - ``dropout``: subset of agents are marked offline during fault window.
    - ``sensor``: subset of agents have corrupted observations during fault window.
    - ``comm``: communication ranges/capacity scaled down during fault window.
    """

    def __init__(self, cfg: FaultConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._severity = float(np.clip(cfg.severity, 0.0, 1.0))

        self._dropout_mask = np.zeros((cfg.num_agents,), dtype=bool)
        self._sensor_corrupt_mask = np.zeros((cfg.num_agents,), dtype=bool)
        self._comm_scale = 1.0

        n_hit = max(1, int(round(self._severity * cfg.num_agents))) if cfg.num_agents > 0 else 0
        n_hit = min(n_hit, cfg.num_agents)
        if n_hit > 0:
            idx = self._rng.choice(cfg.num_agents, size=n_hit, replace=False)
        else:
            idx = np.array([], dtype=int)

        if cfg.fault_type == "dropout":
            self._dropout_mask[idx] = True
        elif cfg.fault_type == "sensor":
            self._sensor_corrupt_mask[idx] = True
        elif cfg.fault_type == "comm":
            self._comm_scale = max(0.05, 1.0 - self._severity)
        else:
            raise ValueError(f"Unsupported fault_type: {cfg.fault_type}")

    @property
    def fault_type(self) -> str:
        return self.cfg.fault_type

    @property
    def t0(self) -> int:
        return int(self.cfg.t0)

    @property
    def duration(self) -> int:
        return int(self.cfg.duration)

    def _active(self, t: int) -> bool:
        start = int(self.cfg.t0)
        end = start + int(self.cfg.duration)
        return start <= int(t) < end

    def sensor_corrupt_mask(self, t: int) -> np.ndarray:
        if self._active(t):
            return self._sensor_corrupt_mask.copy()
        return np.zeros_like(self._sensor_corrupt_mask)

    def dropout_mask(self, t: int) -> np.ndarray:
        if self._active(t):
            return self._dropout_mask.copy()
        return np.zeros_like(self._dropout_mask)

    def comm_scale(self, t: int) -> float:
        if self._active(t):
            return float(self._comm_scale)
        return 1.0

    def sample_obs_noise(self, obs_dim: int) -> np.ndarray:
        # Keep perturbations small and bounded so observation semantics remain recognizable.
        sigma = 0.08 + 0.22 * self._severity
        return self._rng.normal(0.0, sigma, size=(obs_dim,)).astype(np.float32)

    def sample_obs_bias(self, obs_dim: int) -> np.ndarray:
        # Small persistent shift for corrupted sensors.
        mag = 0.03 + 0.10 * self._severity
        return self._rng.uniform(-mag, mag, size=(obs_dim,)).astype(np.float32)
