from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PolicyConfig:
    obs_dim: int
    task_dim: int
    move_dim: int
    comm_dim: int
    mode_dim: int
    hidden_dim: int = 128


class MAPPOPolicy(nn.Module):
    """
    Shared-parameter decentralized actor for MAPPO.
    Outputs separate logits for each discrete action head.
    """
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.head_task = nn.Linear(cfg.hidden_dim, cfg.task_dim)
        self.head_move = nn.Linear(cfg.hidden_dim, cfg.move_dim)
        self.head_comm = nn.Linear(cfg.hidden_dim, cfg.comm_dim)
        self.head_mode = nn.Linear(cfg.hidden_dim, cfg.mode_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        return self.head_task(h), self.head_move(h), self.head_comm(h), self.head_mode(h)

    def act(self, obs: torch.Tensor, deterministic: bool = False, head_mask: tuple[bool, bool, bool] = (True, True, True)):
        logits_task, logits_move, logits_comm, logits_mode = self.forward(obs)
        dist_task = Categorical(logits=logits_task)
        dist_move = Categorical(logits=logits_move)
        dist_comm = Categorical(logits=logits_comm)
        dist_mode = Categorical(logits=logits_mode)

        if deterministic:
            a_task = torch.argmax(logits_task, dim=-1)
            a_move = torch.argmax(logits_move, dim=-1)
            a_comm = torch.argmax(logits_comm, dim=-1)
            a_mode = torch.argmax(logits_mode, dim=-1)
        else:
            a_task = dist_task.sample()
            a_move = dist_move.sample()
            a_comm = dist_comm.sample()
            a_mode = dist_mode.sample()

        logp = 0.0
        ent = 0.0
        if head_mask[0]:
            logp = logp + dist_task.log_prob(a_task)
            ent = ent + dist_task.entropy()
        if head_mask[1]:
            logp = logp + dist_move.log_prob(a_move)
            ent = ent + dist_move.entropy()
        if head_mask[2]:
            logp = logp + dist_comm.log_prob(a_comm)
            ent = ent + dist_comm.entropy()
        # mode always included in action
        logp = logp + dist_mode.log_prob(a_mode)
        ent = ent + dist_mode.entropy()
        actions = torch.stack([a_task, a_move, a_comm, a_mode], dim=-1)
        return actions, logp, ent

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, head_mask: tuple[bool, bool, bool] = (True, True, True)):
        logits_task, logits_move, logits_comm, logits_mode = self.forward(obs)
        dist_task = Categorical(logits=logits_task)
        dist_move = Categorical(logits=logits_move)
        dist_comm = Categorical(logits=logits_comm)
        dist_mode = Categorical(logits=logits_mode)
        a_task, a_move, a_comm, a_mode = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
        logp = 0.0
        ent = 0.0
        if head_mask[0]:
            logp = logp + dist_task.log_prob(a_task)
            ent = ent + dist_task.entropy()
        if head_mask[1]:
            logp = logp + dist_move.log_prob(a_move)
            ent = ent + dist_move.entropy()
        if head_mask[2]:
            logp = logp + dist_comm.log_prob(a_comm)
            ent = ent + dist_comm.entropy()
        logp = logp + dist_mode.log_prob(a_mode)
        ent = ent + dist_mode.entropy()
        return logp, ent
