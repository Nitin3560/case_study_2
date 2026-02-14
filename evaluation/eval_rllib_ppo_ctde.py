from __future__ import annotations

import argparse
import json
import os

from ray.rllib.algorithms.algorithm import Algorithm
import ray.rllib.algorithms.algorithm as rllib_algo
from ray.tune.registry import register_env
from ray.tune import result as tune_result
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.utils.ctde import CTDEWrapper, CentralizedCriticModel
from case_study_2.metrics import (
    task_completion_rate,
    task_delivery_rate,
    user_coverage_rate,
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
    base = LawnTaskCommParallelEnv(cfg)
    return CTDEWrapper(base)


def env_creator(env_config):
    path = env_config.get("env_config_path")
    seed = int(env_config.get("seed", 1))
    return ParallelPettingZooEnv(make_env(path, seed))


def main():
    root = _repo_root()
    # Keep Ray artifacts within workspace to avoid permission issues.
    local_ray_dir = os.path.join(root, "outputs", "ray_tmp")
    os.makedirs(local_ray_dir, exist_ok=True)
    os.environ.setdefault("RAY_AIR_LOCAL_CACHE_DIR", local_ray_dir)
    tune_result.DEFAULT_STORAGE_PATH = local_ray_dir
    rllib_algo.DEFAULT_STORAGE_PATH = local_ray_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)
    register_env("ctde_env", env_creator)
    register_env("ctde_env_curriculum", env_creator)
    algo = Algorithm.from_checkpoint(args.checkpoint)
    env = make_env(args.env_config, args.seed)
    obs, infos = env.reset(seed=args.seed)

    total_bytes = 0
    bytes_per_agent = [0 for _ in range(env.env.n)]
    sum_mean_aoi = 0.0
    sum_p95_aoi = 0.0
    steps_count = 0

    while env.agents:
        actions = {}
        for i, a in enumerate(env.agents):
            action = algo.compute_single_action(obs[a], policy_id="shared", explore=False)
            actions[a] = action
        obs, rewards, terms, truncs, infos = env.step(actions)

        if infos:
            step_total = 0
            for a, info in infos.items():
                b = int(info.get("bytes_delivered_now_agent", 0))
                step_total += b
                idx = env.env.possible_agents.index(a)
                bytes_per_agent[idx] += b
            total_bytes += step_total
            sum_mean_aoi += float(
                mean_aoi(
                    env.env.tasks,
                    env.env.step_i,
                    env.env.user_active if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_last_received if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_appears if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_delivered if env.env.cfg.num_users_total > 0 else None,
                )
            )
            sum_p95_aoi += float(
                p95_aoi(
                    env.env.tasks,
                    env.env.step_i,
                    env.env.user_active if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_last_received if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_appears if env.env.cfg.num_users_total > 0 else None,
                    env.env.user_delivered if env.env.cfg.num_users_total > 0 else None,
                )
            )
            steps_count += 1

    comp_sensed = task_completion_rate(env.env.tasks)
    comp_delivered = task_delivery_rate(env.env.tasks)
    user_cov = user_coverage_rate(env.env.user_delivered) if env.env.cfg.num_users_total > 0 else 0.0
    thr = throughput_bps(total_bytes, env.env.cfg.dt_s, env.env.step_i)
    aoi = mean_aoi(
        env.env.tasks,
        env.env.step_i,
        env.env.user_active if env.env.cfg.num_users_total > 0 else None,
        env.env.user_last_received if env.env.cfg.num_users_total > 0 else None,
        env.env.user_appears if env.env.cfg.num_users_total > 0 else None,
        env.env.user_delivered if env.env.cfg.num_users_total > 0 else None,
    )
    aoi_p95 = p95_aoi(
        env.env.tasks,
        env.env.step_i,
        env.env.user_active if env.env.cfg.num_users_total > 0 else None,
        env.env.user_last_received if env.env.cfg.num_users_total > 0 else None,
        env.env.user_appears if env.env.cfg.num_users_total > 0 else None,
        env.env.user_delivered if env.env.cfg.num_users_total > 0 else None,
    )
    aoi_time_avg = (sum_mean_aoi / steps_count) if steps_count > 0 else 0.0
    aoi_p95_time_avg = (sum_p95_aoi / steps_count) if steps_count > 0 else 0.0
    e_stats = remaining_energy_stats(env.env.agent_energy)
    thr_per_agent = throughput_bps_per_agent(bytes_per_agent, env.env.cfg.dt_s, env.env.step_i)

    result = {
        "seed": args.seed,
        "method": "rllib_ppo_ctde",
        "steps": env.env.step_i,
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
        **e_stats,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
