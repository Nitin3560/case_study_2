from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
import numpy as np

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.baselines.static_assignment import StaticBaseline
from case_study_2.baselines.greedy_auction import GreedyBaseline
from case_study_2.baselines.fixed_relay_tree import FixedRelayTreeBaseline
from case_study_2.metrics import task_completion_rate, task_delivery_rate, user_coverage_rate, throughput_bps, throughput_bps_per_agent, mean_aoi, p95_aoi, remaining_energy_stats
from case_study_2.agents.marl_policy import MAPPOPolicy, PolicyConfig
from case_study_2.controllers.motion_controller import move_towards
import torch
from case_study_2.utils.logger import JsonlLogger


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


class AgenticHeuristic:
    def __init__(self, cooldown_steps: int = 1):
        self.cooldown_steps = int(max(0, cooldown_steps))
        self.cooldown = None

    def act(self, env) -> dict:
        if self.cooldown is None:
            self.cooldown = [0 for _ in range(env.n)]
        user_assign = _assign_users(env)
        actions = {}
        for i, a in enumerate(env.possible_agents):
            if self.cooldown[i] > 0:
                self.cooldown[i] -= 1
                actions[a] = np.array([0, 0, 0, 0], dtype=np.int32)
                continue

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

            if env._buffer_total(i) > 0:
                move_dir = move_towards(env.agent_pos[i], env.base_xy)
                comm = 1
                mode = 1
                base_in = np.linalg.norm(env.agent_pos[i] - env.base_xy) <= env.cfg.base_range_m
                if base_in:
                    self.cooldown[i] = self.cooldown_steps
            else:
                if user is not None:
                    move_dir = move_towards(env.agent_pos[i], env.user_pos[user])
                elif best is not None:
                    move_dir = move_towards(env.agent_pos[i], env.tasks[best].pos_xy)
                else:
                    move_dir = 0
                comm = 0
                mode = 0

            actions[a] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
        return actions


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


def run_one(
    seed: int,
    method: str,
    cfg_path_env: str | None = None,
    ckpt_path: str | None = None,
    log_timeseries: bool = False,
    timeseries_dir: str | None = None,
    stochastic: bool = False,
    use_heuristic: bool = False,
):
    if cfg_path_env is None:
        cfg_path_env = os.path.join(_repo_root(), "configs", "env.yaml")
    env_yaml = load_yaml(cfg_path_env)
    cfg = build_env_cfg(env_yaml, seed=seed)

    env = LawnTaskCommParallelEnv(cfg)
    obs, infos = env.reset(seed=seed)

    assign = None
    ptr = None
    heuristic_policy = None
    if method == "static":
        policy = StaticBaseline(num_agents=cfg.num_agents, num_tasks=cfg.num_tasks_total)
    elif method == "greedy":
        policy = GreedyBaseline(relay_neighbors_max=3)
    elif method == "fixed_relay":
        policy = FixedRelayTreeBaseline(num_agents=cfg.num_agents, num_tasks=cfg.num_tasks_total)
    elif method == "debug_deliver":
        policy = None
    elif method == "random":
        policy = None
    elif method in ("mappo", "mappo_full", "mappo_task_only", "mappo_routing_only"):
        if use_heuristic and method in ("mappo", "mappo_full"):
            policy = None
            heuristic_policy = AgenticHeuristic(cooldown_steps=1)
        else:
            if ckpt_path is None:
                name = "mappo.pt" if method in ("mappo", "mappo_full") else f"{method}.pt"
                ckpt_path = os.path.join(_repo_root(), "outputs", "checkpoints", name)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
            policy = MAPPOPolicy(PolicyConfig(
                obs_dim=obs_dim,
                task_dim=cfg.num_tasks_total + 1,
                move_dim=8,
                comm_dim=cfg.num_agents + 2,
                mode_dim=2,
                hidden_dim=128,
            ))
            policy.load_state_dict(ckpt["policy"])
            policy.eval()
            if method == "mappo_routing_only":
                assign = fixed_assignment(cfg.num_agents, cfg.num_tasks_total)
                ptr = {i: 0 for i in range(cfg.num_agents)}
    else:
        raise ValueError(f"Unknown method: {method}")

    total_bytes = 0
    bytes_per_agent = [0 for _ in range(cfg.num_agents)]
    sum_mean_aoi = 0.0
    sum_p95_aoi = 0.0
    steps_count = 0
    steps_buffer_nonempty = [0 for _ in range(cfg.num_agents)]
    steps_base_in_range = [0 for _ in range(cfg.num_agents)]
    steps_comm_to_base = [0 for _ in range(cfg.num_agents)]
    steps_comm_send_to_base = [0 for _ in range(cfg.num_agents)]
    steps_move_toward_base_when_buffer = [0 for _ in range(cfg.num_agents)]
    bytes_generated_total = 0
    bytes_dropped_total = 0
    bytes_relayed_total = 0
    mean_base_dist_sum = 0.0
    total_buffer_bytes_sum = 0
    r_task_total = 0.0
    r_bytes_total = 0.0
    r_aoi_total = 0.0
    r_energy_total = 0.0
    r_shaping_total = 0.0
    r_sense_total = 0.0
    r_idle_total = 0.0
    r_gen_total = 0.0
    r_mode_total = 0.0
    r_base_total = 0.0
    r_buffer_total = 0.0
    ts_f = None
    if log_timeseries:
        if timeseries_dir is None:
            timeseries_dir = os.path.join(_repo_root(), "outputs", "logs")
        os.makedirs(timeseries_dir, exist_ok=True)
        ts_path = os.path.join(timeseries_dir, f"timeseries_{method}_seed{seed}.jsonl")
        ts_f = open(ts_path, "w", encoding="utf-8")

    for _ in range(cfg.horizon_steps):
        if method in ("mappo", "mappo_full", "mappo_task_only", "mappo_routing_only") and policy is not None:
            obs_batch = torch.tensor(np.stack([obs[a] for a in env.agents], axis=0), dtype=torch.float32)
            with torch.no_grad():
                acts, _, _ = policy.act(obs_batch, deterministic=not stochastic)
            act_np = acts.cpu().numpy()
            if method in ("mappo", "mappo_full"):
                actions = {a: act_np[i] for i, a in enumerate(env.agents)}
            elif method == "mappo_task_only":
                actions = {}
                for i, a in enumerate(env.agents):
                    task = int(act_np[i, 0])
                    move = int(act_np[i, 1])
                    comm = routing_heuristic(env, i)
                    mode = int(act_np[i, 3])
                    actions[a] = np.array([task, move, comm, mode], dtype=np.int32)
            elif method == "mappo_routing_only":
                actions = {}
                for i, a in enumerate(env.agents):
                    tid = pick_assigned_task(env, assign, ptr, i)
                    task_choice = 0 if tid is None else (tid + 1)
                    move_dir = 0 if tid is None else move_towards(env.agent_pos[i], env.tasks[tid].pos_xy)
                    comm = int(act_np[i, 2])
                    mode = int(act_np[i, 3])
                    actions[a] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
        elif method in ("mappo", "mappo_full") and use_heuristic:
            actions = heuristic_policy.act(env)
        elif method == "debug_deliver":
            actions = debug_deliver_policy(env)
        elif method == "random":
            actions = {}
            for a in env.agents:
                # sample random action in valid ranges
                task = np.random.randint(0, cfg.num_tasks_total + 1)
                move = np.random.randint(0, 8)
                comm = np.random.randint(0, cfg.num_agents + 2)
                mode = np.random.randint(0, 2)
                actions[a] = np.array([task, move, comm, mode], dtype=np.int32)
        else:
            actions = policy.act(env)
        # track movement intent toward base when buffer is non-empty
        for i, a in enumerate(env.agents):
            act = actions.get(a)
            if act is None:
                continue
            move_dir = int(act[1])
            if env._buffer_total(i) > 0:
                desired = move_towards(env.agent_pos[i], env.base_xy)
                if move_dir == desired:
                    steps_move_toward_base_when_buffer[i] += 1

        obs, rewards, terms, truncs, infos = env.step(actions)

        # bytes delivered per-step is stored in infos (identical for all agents)
        if infos:
            step_total = 0
            for a, info in infos.items():
                b = int(info.get("bytes_delivered_now_agent", 0))
                step_total += b
                idx = env.possible_agents.index(a)
                bytes_per_agent[idx] += b
                if info.get("buffer_nonempty", False):
                    steps_buffer_nonempty[idx] += 1
                if info.get("base_in_range", False):
                    steps_base_in_range[idx] += 1
                if info.get("comm_to_base", False):
                    steps_comm_to_base[idx] += 1
                if info.get("comm_send_to_base_attempt", False):
                    steps_comm_send_to_base[idx] += 1
            total_bytes += step_total
            # time-average AoI metrics
            sum_mean_aoi += float(
                mean_aoi(
                    env.tasks,
                    env.step_i,
                    env.user_active if env.cfg.num_users_total > 0 else None,
                    env.user_last_received if env.cfg.num_users_total > 0 else None,
                    env.user_appears if env.cfg.num_users_total > 0 else None,
                    env.user_delivered if env.cfg.num_users_total > 0 else None,
                )
            )
            sum_p95_aoi += float(
                p95_aoi(
                    env.tasks,
                    env.step_i,
                    env.user_active if env.cfg.num_users_total > 0 else None,
                    env.user_last_received if env.cfg.num_users_total > 0 else None,
                    env.user_appears if env.cfg.num_users_total > 0 else None,
                    env.user_delivered if env.cfg.num_users_total > 0 else None,
                )
            )
            steps_count += 1
            any_info = next(iter(infos.values()))
            bytes_generated_total += int(any_info.get("bytes_generated_now", 0))
            bytes_dropped_total += int(any_info.get("bytes_dropped_now", 0))
            bytes_relayed_total += int(any_info.get("bytes_relayed_now", 0))
            mean_base_dist_sum += float(any_info.get("mean_base_dist_m", 0.0))
            total_buffer_bytes_sum += int(sum(env._buffer_total(i) for i in range(env.n)))
            r_task_total += float(any_info.get("r_task", 0.0))
            r_bytes_total += float(any_info.get("r_bytes", 0.0))
            r_gen_total += float(any_info.get("r_gen", 0.0))
            r_aoi_total += float(any_info.get("r_aoi", 0.0))
            r_energy_total += float(any_info.get("r_energy", 0.0))
            r_shaping_total += float(any_info.get("r_shaping", 0.0))
            r_sense_total += float(any_info.get("r_sense", 0.0))
            r_idle_total += float(any_info.get("r_idle", 0.0))
            r_mode_total += float(any_info.get("r_mode", 0.0))
            r_base_total += float(any_info.get("r_base", 0.0))
            r_buffer_total += float(any_info.get("r_buffer", 0.0))
            if ts_f is not None:
                any_info = next(iter(infos.values()))
                ts_f.write(json.dumps({
                    "seed": seed,
                    "method": method,
                    "step": env.step_i,
                    "tasks_completed_now": int(any_info.get("tasks_completed_now", 0)),
                    "bytes_delivered_now": int(any_info.get("bytes_delivered_now", 0)),
                    "mean_aoi_steps": float(mean_aoi(env.tasks, env.step_i)),
                    "p95_aoi_steps": float(p95_aoi(env.tasks, env.step_i)),
                }) + "\n")

        if not env.agents:
            break

    comp_sensed = task_completion_rate(env.tasks)
    comp_delivered = task_delivery_rate(env.tasks)
    user_cov = user_coverage_rate(env.user_delivered) if env.cfg.num_users_total > 0 else 0.0
    thr = throughput_bps(total_bytes, cfg.dt_s, env.step_i)
    aoi = mean_aoi(
        env.tasks,
        env.step_i,
        env.user_active if env.cfg.num_users_total > 0 else None,
        env.user_last_received if env.cfg.num_users_total > 0 else None,
        env.user_appears if env.cfg.num_users_total > 0 else None,
        env.user_delivered if env.cfg.num_users_total > 0 else None,
    )
    aoi_p95 = p95_aoi(
        env.tasks,
        env.step_i,
        env.user_active if env.cfg.num_users_total > 0 else None,
        env.user_last_received if env.cfg.num_users_total > 0 else None,
        env.user_appears if env.cfg.num_users_total > 0 else None,
        env.user_delivered if env.cfg.num_users_total > 0 else None,
    )
    aoi_time_avg = (sum_mean_aoi / steps_count) if steps_count > 0 else 0.0
    aoi_p95_time_avg = (sum_p95_aoi / steps_count) if steps_count > 0 else 0.0
    e_stats = remaining_energy_stats(env.agent_energy)
    thr_per_agent = throughput_bps_per_agent(bytes_per_agent, cfg.dt_s, env.step_i)

    if ts_f is not None:
        ts_f.close()

    return {
        "seed": seed,
        "method": method,
        "steps": env.step_i,
        # Publication metric: delivered task completion only.
        "task_completion_pct": comp_delivered,
        "task_delivery_pct": comp_delivered,
        "task_sensed_pct": comp_sensed,
        "user_coverage_pct": user_cov,
        "throughput_bps": thr,
        "mean_aoi_steps": aoi,
        "p95_aoi_steps": aoi_p95,
        "mean_aoi_time_avg": aoi_time_avg,
        "p95_aoi_time_avg": aoi_p95_time_avg,
        "total_bytes_delivered": total_bytes,
        "throughput_bps_per_agent": thr_per_agent,
        "total_bytes_per_agent": bytes_per_agent,
        "bytes_generated_total": bytes_generated_total,
        "bytes_dropped_total": bytes_dropped_total,
        "bytes_relayed_total": bytes_relayed_total,
        "steps_buffer_nonempty_frac": [s / steps_count if steps_count > 0 else 0.0 for s in steps_buffer_nonempty],
        "steps_base_in_range_frac": [s / steps_count if steps_count > 0 else 0.0 for s in steps_base_in_range],
        "steps_comm_to_base_frac": [s / steps_count if steps_count > 0 else 0.0 for s in steps_comm_to_base],
        "steps_comm_send_to_base_frac": [s / steps_count if steps_count > 0 else 0.0 for s in steps_comm_send_to_base],
        "steps_move_toward_base_when_buffer_frac": [s / steps_count if steps_count > 0 else 0.0 for s in steps_move_toward_base_when_buffer],
        "steps_count": steps_count,
        "mean_base_dist_time_avg": (mean_base_dist_sum / steps_count) if steps_count > 0 else 0.0,
        "total_buffer_bytes_time_avg": (total_buffer_bytes_sum / steps_count) if steps_count > 0 else 0.0,
        "r_task_total": r_task_total,
        "r_delivery_total": r_bytes_total,
        "r_gen_total": r_gen_total,
        "r_aoi_total": r_aoi_total,
        "r_energy_total": r_energy_total,
        "r_shaping_total": r_shaping_total,
        "r_sense_total": r_sense_total,
        "r_idle_total": r_idle_total,
        "r_mode_total": r_mode_total,
        "r_base_total": r_base_total,
        "r_buffer_total": r_buffer_total,
        **e_stats,
        "env_cfg": asdict(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["static", "greedy", "fixed_relay", "debug_deliver", "random", "mappo", "mappo_full", "mappo_task_only", "mappo_routing_only"], default="static")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--use-heuristic", action="store_true")
    ap.add_argument("--log-timeseries", action="store_true")
    ap.add_argument("--timeseries-dir", type=str, default=None)
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument("--env-config", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logs_dir = os.path.join(_repo_root(), "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(logs_dir, "eval_results.jsonl"))

    results = []
    for seed in seeds:
        r = run_one(
            seed=seed,
            method=args.method,
            cfg_path_env=args.env_config,
            ckpt_path=args.ckpt,
            log_timeseries=args.log_timeseries,
            timeseries_dir=args.timeseries_dir,
            stochastic=args.stochastic,
            use_heuristic=args.use_heuristic,
        )
        results.append(r)
        logger.write(r)
        print(json.dumps({k: r[k] for k in ["seed","method","task_completion_pct","throughput_bps","mean_aoi_steps","p95_aoi_steps"]}, indent=2))

    logger.close()


if __name__ == "__main__":
    main()
