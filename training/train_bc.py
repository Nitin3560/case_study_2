from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.agents.marl_policy import MAPPOPolicy, PolicyConfig
from case_study_2.training.train_mappo import debug_deliver_policy, global_state


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def collect_dataset(env: LawnTaskCommParallelEnv, steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    obs, infos = env.reset(seed=seed)
    obs_list = []
    act_list = []
    for t in range(steps):
        actions = debug_deliver_policy(env)
        for i, a in enumerate(env.agents):
            obs_list.append(obs[a])
            act_list.append(actions[a])
        obs, rewards, terms, truncs, infos = env.step(actions)
        if not env.agents:
            obs, infos = env.reset(seed=seed + t + 1)
    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def train_bc(args):
    env_yaml = load_yaml(args.env_config)
    cfg = build_env_cfg(env_yaml, seed=args.seed)
    env = LawnTaskCommParallelEnv(cfg)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    policy = MAPPOPolicy(PolicyConfig(
        obs_dim=obs_dim,
        task_dim=cfg.num_tasks_total + 1,
        move_dim=8,
        comm_dim=cfg.num_agents + 2,
        mode_dim=2,
        hidden_dim=args.hidden_dim,
    ))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy.to(device)

    obs_np, act_np = collect_dataset(env, args.dataset_steps, args.seed)
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    act_t = torch.tensor(act_np, dtype=torch.int64, device=device)

    optim = Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    n = obs_t.shape[0]
    for epoch in range(args.epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            b = idx[start:start + args.batch_size]
            if b.numel() == 0:
                continue
            logits_task, logits_move, logits_comm, logits_mode = policy.forward(obs_t[b])
            loss = 0.0
            loss = loss + loss_fn(logits_task, act_t[b, 0])
            loss = loss + loss_fn(logits_move, act_t[b, 1])
            loss = loss + loss_fn(logits_comm, act_t[b, 2])
            loss = loss + loss_fn(logits_mode, act_t[b, 3])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

    out_dir = os.path.join(_repo_root(), "outputs", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "policy": policy.state_dict(),
        "env_cfg": asdict(cfg),
        "obs_dim": obs_dim,
    }
    path = os.path.join(out_dir, args.ckpt_name)
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env.yaml"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset-steps", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--ckpt-name", type=str, default="mappo.pt")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
