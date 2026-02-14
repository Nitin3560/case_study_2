from __future__ import annotations

import argparse
import os
from dataclasses import asdict
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.agents.marl_policy import MAPPOPolicy, PolicyConfig
from case_study_2.agents.critic import CentralCritic, CriticConfig
from case_study_2.controllers.motion_controller import move_towards


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def global_state(env: LawnTaskCommParallelEnv) -> np.ndarray:
    """
    Centralized state: agents (pos, energy, buffer, task),
    tasks (pos, active, completed), base pos.
    """
    cfg = env.cfg
    w, h = cfg.map_xy_m
    parts = []
    for i in range(env.n):
        pos = env.agent_pos[i] / np.array([w, h], dtype=np.float32)
        energy = np.array([env.agent_energy[i] / cfg.e0_j], dtype=np.float32)
        buf = np.array([env._buffer_total(i) / cfg.buffer_bytes_max], dtype=np.float32)
        tid = env.agent_task[i]
        if tid < 0:
            tid_norm = -1.0
        else:
            denom = max(cfg.num_tasks_total - 1, 1)
            tid_norm = (float(tid) / float(denom)) * 2.0 - 1.0
        parts.append(np.array([pos[0], pos[1], energy[0], buf[0], tid_norm], dtype=np.float32))

    for t in env.tasks:
        p = t.pos_xy / np.array([w, h], dtype=np.float32)
        parts.append(np.array([p[0], p[1], 1.0 if t.active else 0.0, 1.0 if t.completed else 0.0], dtype=np.float32))

    for uid in range(env.cfg.num_users_total):
        p = env.user_pos[uid] / np.array([w, h], dtype=np.float32)
        parts.append(np.array([p[0], p[1], 1.0 if env.user_active[uid] else 0.0], dtype=np.float32))

    base = np.array([env.base_xy[0] / w, env.base_xy[1] / h], dtype=np.float32)
    parts.append(base)
    return np.concatenate(parts, axis=0)


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logp = []
        self.rew = []
        self.val = []
        self.done = []
        self.state = []

    def add(self, obs, actions, logp, rew, val, done, state):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.done.append(done)
        self.state.append(state)

    def stack(self, device):
        n_agents = int(self.obs[0].shape[0]) if self.obs else 0
        obs = torch.tensor(np.concatenate(self.obs, axis=0), dtype=torch.float32, device=device)
        actions = torch.tensor(np.concatenate(self.actions, axis=0), dtype=torch.int64, device=device)
        logp = torch.tensor(np.concatenate(self.logp, axis=0), dtype=torch.float32, device=device)
        rew = torch.tensor(np.array(self.rew, dtype=np.float32), device=device)
        val = torch.tensor(np.array(self.val, dtype=np.float32), device=device)
        done = torch.tensor(np.array(self.done, dtype=np.float32), device=device)
        state = torch.tensor(np.array(self.state, dtype=np.float32), device=device)
        return obs, actions, logp, rew, val, done, state, n_agents


def compute_gae(rews, vals, dones, gamma, lam):
    adv = np.zeros_like(rews, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rews))):
        mask = 1.0 - dones[t]
        delta = rews[t] + gamma * vals[t + 1] * mask - vals[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv


def fixed_assignment(num_agents: int, num_tasks: int):
    assign = {i: [] for i in range(num_agents)}
    for tid in range(num_tasks):
        assign[tid % num_agents].append(tid)
    return assign


def pick_assigned_task(env, assign, ptr, i: int):
    lst = assign[i]
    if not lst:
        return None
    for _ in range(len(lst)):
        tid = lst[ptr[i] % len(lst)]
        ptr[i] = (ptr[i] + 1) % len(lst)
        t = env.tasks[tid]
        if t.active and (not t.completed):
            return tid
    return None


def routing_heuristic(env, i: int) -> int:
    # send to base if in range, else neighbor closest to base
    if float(np.linalg.norm(env.agent_pos[i] - env.base_xy)) <= float(env.cfg.base_range_m):
        return 1
    best = None
    best_d = 1e18
    for k in range(env.n):
        if k == i:
            continue
        d_ik = float(np.linalg.norm(env.agent_pos[i] - env.agent_pos[k]))
        if d_ik > float(env.cfg.comm_range_m):
            continue
        d_kb = float(np.linalg.norm(env.agent_pos[k] - env.base_xy))
        if d_kb < best_d:
            best_d = d_kb
            best = k
    if best is None:
        return 0
    return 2 + int(best)


def _nearest_active_user(env, i: int):
    best = None
    best_d = 1e18
    for uid in range(env.cfg.num_users_total):
        if not env.user_active[uid] or env.user_delivered[uid]:
            continue
        d = float(np.linalg.norm(env.agent_pos[i] - env.user_pos[uid]))
        if d < best_d:
            best_d = d
            best = uid
    return best


def _assign_users(env) -> list[int | None]:
    n = env.n
    active_users = [u for u in range(env.cfg.num_users_total) if env.user_active[u] and not env.user_delivered[u]]
    assignment = [None for _ in range(n)]
    if not active_users:
        return assignment
    # greedy matching by distance
    remaining_agents = set(range(n))
    remaining_users = set(active_users)
    while remaining_agents and remaining_users:
        best = None
        best_d = 1e18
        for i in remaining_agents:
            for u in remaining_users:
                d = float(np.linalg.norm(env.agent_pos[i] - env.user_pos[u]))
                if d < best_d:
                    best_d = d
                    best = (i, u)
        if best is None:
            break
        i, u = best
        assignment[i] = u
        remaining_agents.remove(i)
        remaining_users.remove(u)
    return assignment


def debug_deliver_policy(env) -> dict:
    user_assign = _assign_users(env)
    actions = {}
    for i, a in enumerate(env.possible_agents):
        # prioritize undelivered active users, then tasks
        user = user_assign[i] if user_assign[i] is not None else _nearest_active_user(env, i)
        # pick nearest active uncompleted task
        best = None
        best_d = 1e18
        for t in env.tasks:
            if not t.active or t.completed:
                continue
            d = float(np.linalg.norm(env.agent_pos[i] - t.pos_xy))
            if d < best_d:
                best_d = d
                best = t.tid
        task_choice = 0 if best is None else (best + 1)

        relay_threshold = 1000
        if env._buffer_total(i) >= relay_threshold:
            move_dir = move_towards(env.agent_pos[i], env.base_xy)
            comm = 1  # send_to_base
            mode = 1  # relay
        else:
            if user is not None:
                move_dir = move_towards(env.agent_pos[i], env.user_pos[user])
            elif best is not None:
                move_dir = move_towards(env.agent_pos[i], env.tasks[best].pos_xy)
            else:
                move_dir = 0
            comm = 0
            mode = 0  # scan
        actions[a] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
    return actions


def train(args):
    env_yaml = load_yaml(args.env_config)
    cfg = build_env_cfg(env_yaml, seed=args.seed)
    env = LawnTaskCommParallelEnv(cfg)
    obs, infos = env.reset(seed=args.seed)

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    state_dim = len(global_state(env))

    policy = MAPPOPolicy(PolicyConfig(
        obs_dim=obs_dim,
        task_dim=cfg.num_tasks_total + 1,
        move_dim=8,
        comm_dim=cfg.num_agents + 2,
        mode_dim=2,
        hidden_dim=args.hidden_dim,
    ))
    critic = CentralCritic(CriticConfig(state_dim=state_dim, hidden_dim=args.hidden_dim * 2))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy.to(device)
    critic.to(device)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])

    optim_pi = Adam(policy.parameters(), lr=args.lr)
    optim_v = Adam(critic.parameters(), lr=args.lr)

    buffer = RolloutBuffer()
    steps = 0
    episode = 0
    t0 = time.time()

    # Imitation warm-start (optional)
    if args.imitation_steps > 0:
        im_optim = Adam(policy.parameters(), lr=args.imitation_lr)
        obs, infos = env.reset(seed=args.seed)
        im_steps = 0
        while im_steps < args.imitation_steps:
            exp_actions = debug_deliver_policy(env)
            obs_list = [obs[a] for a in env.agents]
            obs_batch = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)
            act_np = np.stack([exp_actions[a] for a in env.agents], axis=0)
            act_t = torch.tensor(act_np, dtype=torch.int64, device=device)
            logp, ent = policy.evaluate_actions(obs_batch, act_t)
            loss = -logp.mean()
            im_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            im_optim.step()

            obs, rewards, terms, truncs, infos = env.step(exp_actions)
            im_steps += 1
            if not env.agents:
                obs, infos = env.reset(seed=args.seed + im_steps)

    assign = fixed_assignment(cfg.num_agents, cfg.num_tasks_total)
    ptr = {i: 0 for i in range(cfg.num_agents)}

    while steps < args.total_steps:
        obs, infos = env.reset(seed=args.seed + episode)
        done = False
        ep_steps = 0

        while not done and steps < args.total_steps:
            # build batched obs for all agents
            obs_list = [obs[a] for a in env.agents]
            obs_batch = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)

            head_mask = (True, True, True)
            if args.mode == "task_only":
                head_mask = (True, True, False)
            elif args.mode == "routing_only":
                head_mask = (False, False, True)
            with torch.no_grad():
                actions, logp, ent = policy.act(obs_batch, deterministic=False, head_mask=head_mask)
                state = torch.tensor(global_state(env), dtype=torch.float32, device=device)
                value = critic(state).item()

            # env expects dict actions per agent
            act_np = actions.cpu().numpy()
            if args.mode == "full":
                act_dict = {a: act_np[i] for i, a in enumerate(env.agents)}
            elif args.mode == "task_only":
                # policy controls task + move, routing fixed
                act_dict = {}
                for i, a in enumerate(env.agents):
                    task = int(act_np[i, 0])
                    move = int(act_np[i, 1])
                    comm = routing_heuristic(env, i)
                    mode = int(act_np[i, 3])
                    act_dict[a] = np.array([task, move, comm, mode], dtype=np.int32)
            elif args.mode == "routing_only":
                # fixed task assignment + motion, policy controls routing
                act_dict = {}
                for i, a in enumerate(env.agents):
                    tid = pick_assigned_task(env, assign, ptr, i)
                    task_choice = 0 if tid is None else (tid + 1)
                    if tid is None:
                        move_dir = 0
                    else:
                        move_dir = move_towards(env.agent_pos[i], env.tasks[tid].pos_xy)
                    comm = int(act_np[i, 2])
                    mode = int(act_np[i, 3])
                    act_dict[a] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            next_obs, rewards, terms, truncs, infos = env.step(act_dict)

            # team reward: identical for all agents
            any_r = next(iter(rewards.values())) if rewards else 0.0
            done = (not env.agents)

            buffer.add(
                obs=np.stack(obs_list, axis=0),
                actions=act_np,
                logp=logp.cpu().numpy(),
                rew=any_r,
                val=value,
                done=1.0 if done else 0.0,
                state=global_state(env),
            )

            obs = next_obs
            steps += 1
            ep_steps += 1

            if len(buffer.rew) >= args.rollout_steps:
                # bootstrap value
                with torch.no_grad():
                    next_val = critic(torch.tensor(global_state(env), dtype=torch.float32, device=device)).item()
                vals = np.array(buffer.val + [next_val], dtype=np.float32)
                rews = np.array(buffer.rew, dtype=np.float32)
                dones = np.array(buffer.done, dtype=np.float32)
                adv = compute_gae(rews, vals, dones, args.gamma, args.gae_lambda)
                ret = adv + vals[:-1]

                obs_t, actions_t, logp_t, rew_t, val_t, done_t, state_t, n_agents = buffer.stack(device)
                adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
                ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

                # normalize advantage
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                adv_rep = adv_t.repeat_interleave(n_agents)

                # PPO update
                for _ in range(args.epochs):
                    idx = torch.randperm(len(adv_rep), device=device)
                    for start in range(0, len(adv_rep), args.batch_size):
                        b = idx[start:start + args.batch_size]
                        if b.numel() == 0:
                            continue
                        logp_new, ent = policy.evaluate_actions(obs_t[b], actions_t[b], head_mask=head_mask)
                        ratio = torch.exp(logp_new - logp_t[b])
                        surr1 = ratio * adv_rep[b]
                        surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_rep[b]
                        loss_pi = -torch.min(surr1, surr2).mean() - args.entropy_coef * ent.mean()

                        # critic is updated on per-step states
                        idx_v = torch.randperm(len(ret_t), device=device)[: max(1, len(ret_t) // 2)]
                        v_pred = critic(state_t[idx_v])
                        loss_v = nn.functional.mse_loss(v_pred, ret_t[idx_v])

                        optim_pi.zero_grad()
                        loss_pi.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                        optim_pi.step()

                        optim_v.zero_grad()
                        loss_v.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                        optim_v.step()

                buffer = RolloutBuffer()

        episode += 1

        if episode % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"episode={episode} steps={steps} elapsed_s={elapsed:.1f}")

    # save checkpoint
    out_dir = os.path.join(_repo_root(), "outputs", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "env_cfg": asdict(cfg),
        "obs_dim": obs_dim,
        "state_dim": state_dim,
    }
    path = os.path.join(out_dir, args.ckpt_name)
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env.yaml"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--rollout-steps", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-name", type=str, default="mappo.pt")
    ap.add_argument("--load-ckpt", type=str, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mode", choices=["full", "task_only", "routing_only"], default="full")
    ap.add_argument("--imitation-steps", type=int, default=0)
    ap.add_argument("--imitation-lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
