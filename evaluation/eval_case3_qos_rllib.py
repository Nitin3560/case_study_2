from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "case3_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import ray.rllib.algorithms.algorithm as rllib_algo
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune import result as tune_result

from case_study_2.baselines.static_assignment import StaticBaseline
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.metrics import throughput_bps
from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.utils.ctde import CTDEWrapper, CentralizedCriticModel


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mean_std(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _collect_episode(env, action_fn, method: str, seed: int) -> tuple[dict, list[dict]]:
    base_env = env.env if hasattr(env, "env") else env
    obs, _ = env.reset(seed=seed)
    total_delivered = 0
    total_generated = 0
    latency_bytes_weighted_sum = 0.0
    latency_bytes_total = 0
    packet_lat_steps: list[float] = []
    ts_rows: list[dict] = []
    outstanding_bytes = 0

    while env.agents:
        actions = action_fn(env, obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        if not infos:
            continue
        any_info = next(iter(infos.values()))
        b_del = int(any_info.get("bytes_delivered_now", 0))
        b_gen = int(any_info.get("bytes_generated_now", 0))
        total_delivered += b_del
        total_generated += b_gen
        outstanding_bytes = max(0, outstanding_bytes + b_gen - b_del)
        latency_bytes_weighted_sum += float(any_info.get("delivered_latency_step_bytes_sum", 0.0))
        latency_bytes_total += int(any_info.get("delivered_latency_bytes_now", 0))
        packet_lat_steps.extend([float(x) for x in any_info.get("delivered_packet_latency_steps", [])])
        delivered_now = int(any_info.get("delivered_latency_bytes_now", 0))
        if delivered_now > 0:
            lat_now_ms = (
                float(any_info.get("delivered_latency_step_bytes_sum", 0.0))
                / float(delivered_now)
                * float(base_env.cfg.dt_s)
                * 1000.0
            )
        elif outstanding_bytes > 0:
            # Penalize non-delivery while traffic is outstanding to capture outages.
            lat_now_ms = float(base_env.cfg.horizon_steps * base_env.cfg.dt_s * 1000.0)
        else:
            lat_now_ms = 0.0
        ts_rows.append(
            {
                "step": int(base_env.step_i),
                "method": method,
                "seed": seed,
                "throughput_bps_now": float(b_del * 8.0 / float(base_env.cfg.dt_s)),
                "latency_ms_mean_now": float(lat_now_ms),
            }
        )

    mean_lat_steps = float(latency_bytes_weighted_sum / latency_bytes_total) if latency_bytes_total > 0 else 0.0
    p95_lat_steps = float(np.percentile(np.array(packet_lat_steps, dtype=np.float32), 95.0)) if packet_lat_steps else 0.0
    undelivered_bytes = max(0, total_generated - total_delivered)
    eff_lat_steps = (
        (latency_bytes_weighted_sum + float(undelivered_bytes * base_env.cfg.horizon_steps))
        / float(total_generated)
        if total_generated > 0 else 0.0
    )
    result = {
        "method": method,
        "seed": seed,
        "throughput_bps": throughput_bps(total_delivered, base_env.cfg.dt_s, base_env.step_i),
        "delivery_rate_pct": (float(total_delivered) / float(total_generated) * 100.0) if total_generated > 0 else 0.0,
        "packet_loss_pct": (100.0 - (float(total_delivered) / float(total_generated) * 100.0)) if total_generated > 0 else 100.0,
        "mean_latency_ms": float(mean_lat_steps * base_env.cfg.dt_s * 1000.0),
        "effective_latency_ms": float(eff_lat_steps * base_env.cfg.dt_s * 1000.0),
        "p95_latency_ms": float(p95_lat_steps * base_env.cfg.dt_s * 1000.0),
        "throughput_std_over_time": float(np.std([r["throughput_bps_now"] for r in ts_rows])) if ts_rows else 0.0,
        "throughput_cv_over_time": float(
            (np.std([r["throughput_bps_now"] for r in ts_rows]) / np.mean([r["throughput_bps_now"] for r in ts_rows]))
            if ts_rows and np.mean([r["throughput_bps_now"] for r in ts_rows]) > 1e-9 else 0.0
        ),
    }
    return result, ts_rows


def _plot_series(ts_by_method: dict[str, list[list[dict]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key, name, ylab in [
        ("latency_ms_mean_now", "latency_vs_time_case3.png", "Latency (ms)"),
        ("throughput_bps_now", "throughput_vs_time_case3.png", "Throughput (bps)"),
    ]:
        plt.figure(figsize=(9, 4.5))
        for method, runs in ts_by_method.items():
            step_vals = defaultdict(list)
            for run in runs:
                for r in run:
                    step_vals[int(r["step"])].append(float(r.get(key, 0.0)))
            steps = sorted(step_vals.keys())
            if not steps:
                continue
            ys = [float(np.mean(step_vals[s])) for s in steps]
            plt.plot(steps, ys, label=method)
        plt.xlabel("Step")
        plt.ylabel(ylab)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), dpi=180)
        plt.close()

    plt.figure(figsize=(7.5, 4.5))
    for method, runs in ts_by_method.items():
        vals = []
        for run in runs:
            vals.extend([float(r["latency_ms_mean_now"]) for r in run if float(r["latency_ms_mean_now"]) > 0])
        if not vals:
            continue
        xs = np.sort(np.array(vals, dtype=np.float32))
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        plt.plot(xs, ys, label=method)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_cdf_case3.png"), dpi=180)
    plt.close()


def make_ctde_env(path: str, seed: int):
    env_yaml = load_yaml(path)
    cfg = build_env_cfg(env_yaml, seed=seed)
    return CTDEWrapper(LawnTaskCommParallelEnv(cfg))


def env_creator(env_config):
    return ParallelPettingZooEnv(make_ctde_env(env_config["env_config_path"], int(env_config.get("seed", 1))))


def main() -> None:
    root = _repo_root()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "case3_mplconfig"))
    local_ray_dir = os.path.join(root, "outputs", "ray_tmp")
    os.makedirs(local_ray_dir, exist_ok=True)
    os.environ.setdefault("RAY_AIR_LOCAL_CACHE_DIR", local_ray_dir)
    tune_result.DEFAULT_STORAGE_PATH = local_ray_dir
    rllib_algo.DEFAULT_STORAGE_PATH = local_ray_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_easy.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs", "case3_qos_rllib"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)
    register_env("ctde_env", env_creator)
    register_env("ctde_env_curriculum", env_creator)

    algo = Algorithm.from_checkpoint(args.checkpoint)

    all_rows = {"static": [], "marl_ctde": []}
    ts_by_method = {"static": [], "marl_ctde": []}

    for seed in seeds:
        # static
        env_yaml = load_yaml(args.env_config)
        cfg = build_env_cfg(env_yaml, seed=seed)
        env_s = LawnTaskCommParallelEnv(cfg)
        static_policy = StaticBaseline(num_agents=cfg.num_agents, num_tasks=cfg.num_tasks_total)
        res_s, ts_s = _collect_episode(env_s, lambda env, obs: static_policy.act(env), "static", seed)
        all_rows["static"].append(res_s)
        ts_by_method["static"].append(ts_s)

        # RLlib CTDE
        env_r = make_ctde_env(args.env_config, seed)

        def _act_rllib(env, obs):
            acts = {}
            for a in env.agents:
                acts[a] = algo.compute_single_action(obs[a], policy_id="shared", explore=False)
            return acts

        res_r, ts_r = _collect_episode(env_r, _act_rllib, "marl_ctde", seed)
        all_rows["marl_ctde"].append(res_r)
        ts_by_method["marl_ctde"].append(ts_r)

    summary = {}
    for m, rows in all_rows.items():
        summary[m] = {
            "mean_latency_ms": _mean_std([r["mean_latency_ms"] for r in rows]),
            "effective_latency_ms": _mean_std([r["effective_latency_ms"] for r in rows]),
            "p95_latency_ms": _mean_std([r["p95_latency_ms"] for r in rows]),
            "throughput_bps": _mean_std([r["throughput_bps"] for r in rows]),
            "throughput_std_over_time": _mean_std([r["throughput_std_over_time"] for r in rows]),
            "throughput_cv_over_time": _mean_std([r["throughput_cv_over_time"] for r in rows]),
            "delivery_rate_pct": _mean_std([r["delivery_rate_pct"] for r in rows]),
            "packet_loss_pct": _mean_std([r["packet_loss_pct"] for r in rows]),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "case3_qos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "case3_qos_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Mean Latency (ms)", "Effective Latency (ms)", "P95 Latency (ms)", "Throughput (bps)", "Throughput std-over-time", "Throughput CV-over-time", "Delivery rate (%)", "Packet loss (%)"])
        for m in ["static", "marl_ctde"]:
            s = summary[m]
            w.writerow([
                m,
                f"{s['mean_latency_ms']['mean']:.2f} ± {s['mean_latency_ms']['std']:.2f}",
                f"{s['effective_latency_ms']['mean']:.2f} ± {s['effective_latency_ms']['std']:.2f}",
                f"{s['p95_latency_ms']['mean']:.2f} ± {s['p95_latency_ms']['std']:.2f}",
                f"{s['throughput_bps']['mean']:.2f} ± {s['throughput_bps']['std']:.2f}",
                f"{s['throughput_std_over_time']['mean']:.2f} ± {s['throughput_std_over_time']['std']:.2f}",
                f"{s['throughput_cv_over_time']['mean']:.3f} ± {s['throughput_cv_over_time']['std']:.3f}",
                f"{s['delivery_rate_pct']['mean']:.2f} ± {s['delivery_rate_pct']['std']:.2f}",
                f"{s['packet_loss_pct']['mean']:.2f} ± {s['packet_loss_pct']['std']:.2f}",
            ])

    _plot_series(ts_by_method, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
