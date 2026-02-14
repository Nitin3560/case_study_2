from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CriticConfig:
    state_dim: int
    hidden_dim: int = 256


class CentralCritic(nn.Module):
    """
    Centralized critic for CTDE MAPPO.
    """
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
