from __future__ import annotations

import argparse
import json
import os

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.metrics import (
    task_completion_rate,
    task_delivery_rate,
    throughput_bps,
    throughput_bps_per_agent,
    mean_aoi,
    p95_aoi,
    remaining_energy_stats,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_env(env_cfg_path: str, seed: int):
    env_yaml = load_yaml(env_cfg_path)
    cfg = build_env_cfg(env_yaml, seed=seed)
    return LawnTaskCommParallelEnv(cfg)


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    algo = Algorithm.from_checkpoint(args.checkpoint)
    env = make_env(args.env_config, args.seed)
    obs, infos = env.reset(seed=args.seed)

    total_bytes = 0
    bytes_per_agent = [0 for _ in range(env.n)]
    sum_mean_aoi = 0.0
    sum_p95_aoi = 0.0
    steps_count = 0

    while env.agents:
        actions = {}
        for i, a in enumerate(env.agents):
            action = algo.compute_single_action(obs[a], policy_id="shared")
            actions[a] = action
        obs, rewards, terms, truncs, infos = env.step(actions)

        if infos:
            step_total = 0
            for a, info in infos.items():
                b = int(info.get("bytes_delivered_now_agent", 0))
                step_total += b
                idx = env.possible_agents.index(a)
                bytes_per_agent[idx] += b
            total_bytes += step_total
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

    comp_sensed = task_completion_rate(env.tasks)
    comp_delivered = task_delivery_rate(env.tasks)
    thr = throughput_bps(total_bytes, env.cfg.dt_s, env.step_i)
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
    thr_per_agent = throughput_bps_per_agent(bytes_per_agent, env.cfg.dt_s, env.step_i)

    result = {
        "seed": args.seed,
        "method": "rllib_mappo",
        "steps": env.step_i,
        "task_completion_pct": comp_delivered,
        "task_sensed_pct": comp_sensed,
        "throughput_bps": thr,
        "mean_aoi_steps": aoi,
        "p95_aoi_steps": aoi_p95,
        "mean_aoi_time_avg": aoi_time_avg,
        "p95_aoi_time_avg": aoi_p95_time_avg,
        "total_bytes_delivered": total_bytes,
        "throughput_bps_per_agent": thr_per_agent,
        "total_bytes_per_agent": bytes_per_agent,
        **e_stats,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
