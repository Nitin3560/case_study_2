from __future__ import annotations

import argparse
import json
import os
import statistics

from case_study_2.evaluation.eval_runner import run_one


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mean_std(rows, key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows]
    if len(vals) <= 1:
        return float(vals[0]) if vals else 0.0, 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_easy.yaml"))
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--methods", default="static,fixed_relay,greedy")
    ap.add_argument("--out", default=os.path.join(root, "outputs", "logs", "baselines_summary.json"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    results = {}
    for method in methods:
        rows = [run_one(seed=s, method=method, cfg_path_env=args.env_config) for s in seeds]
        results[method] = {
            "task_completion_pct": _mean_std(rows, "task_completion_pct"),
            "throughput_bps": _mean_std(rows, "throughput_bps"),
            "mean_aoi_steps": _mean_std(rows, "mean_aoi_steps"),
            "mean_aoi_time_avg": _mean_std(rows, "mean_aoi_time_avg"),
        }

    payload = {
        "seeds": seeds,
        "env_config": args.env_config,
        "methods": methods,
        "summary": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
