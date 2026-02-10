from __future__ import annotations
import argparse
import json
from dataclasses import asdict

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.baselines.static_assignment import StaticBaseline
from case_study_2.baselines.greedy_auction import GreedyBaseline
from case_study_2.metrics import task_completion_rate, throughput_bps, mean_aoi, remaining_energy_stats
from case_study_2.utils.logger import JsonlLogger


def run_one(seed: int, method: str, cfg_path_env: str = "case_study_2/configs/env.yaml"):
    env_yaml = load_yaml(cfg_path_env)
    cfg = build_env_cfg(env_yaml, seed=seed)

    env = LawnTaskCommParallelEnv(cfg)
    obs, infos = env.reset(seed=seed)

    if method == "static":
        policy = StaticBaseline(num_agents=cfg.num_agents, num_tasks=cfg.num_tasks_total)
    elif method == "greedy":
        policy = GreedyBaseline(relay_neighbors_max=3)
    else:
        raise ValueError(f"Unknown method: {method}")

    total_bytes = 0
    for _ in range(cfg.horizon_steps):
        actions = policy.act(env)
        obs, rewards, terms, truncs, infos = env.step(actions)

        # bytes delivered per-step is stored in infos (identical for all agents)
        if infos:
            any_info = next(iter(infos.values()))
            total_bytes += int(any_info.get("bytes_delivered_now", 0))

        if not env.agents:
            break

    comp = task_completion_rate(env.tasks)
    thr = throughput_bps(total_bytes, cfg.dt_s, env.step_i)
    aoi = mean_aoi(env.tasks, env.step_i)
    e_stats = remaining_energy_stats(env.agent_energy)

    return {
        "seed": seed,
        "method": method,
        "steps": env.step_i,
        "task_completion_pct": comp,
        "throughput_bps": thr,
        "mean_aoi_steps": aoi,
        "total_bytes_delivered": total_bytes,
        **e_stats,
        "env_cfg": asdict(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["static", "greedy"], default="static")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logger = JsonlLogger("case_study_2/outputs/logs/eval_results.jsonl")

    results = []
    for seed in seeds:
        r = run_one(seed=seed, method=args.method)
        results.append(r)
        logger.write(r)
        print(json.dumps({k: r[k] for k in ["seed","method","task_completion_pct","throughput_bps","mean_aoi_steps"]}, indent=2))

    logger.close()


if __name__ == "__main__":
    main()