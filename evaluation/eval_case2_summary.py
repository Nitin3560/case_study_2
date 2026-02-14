from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess


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
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_case2_publication.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=os.path.join(root, "outputs", "logs", "case2_summary.json"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rllib_rows = []
    static_rows = []

    for seed in seeds:
        cmd_rl = [
            "python",
            os.path.join(root, "evaluation", "eval_rllib_ppo_ctde.py"),
            "--env-config",
            args.env_config,
            "--checkpoint",
            args.checkpoint,
            "--seed",
            str(seed),
        ]
        p_rl = subprocess.run(
            cmd_rl,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": "/Users/nitin/Desktop",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "OMP_NUM_THREADS": "1",
            },
            check=True,
        )
        rllib_rows.append(json.loads(p_rl.stdout[p_rl.stdout.rfind("{") :]))

        cmd_st = [
            "python",
            os.path.join(root, "evaluation", "eval_runner.py"),
            "--method",
            "static",
            "--env-config",
            args.env_config,
            "--seeds",
            str(seed),
        ]
        p_st = subprocess.run(
            cmd_st,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": "/Users/nitin/Desktop",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "OMP_NUM_THREADS": "1",
            },
            check=True,
        )
        # eval_runner prints short JSON; parse last object
        static_rows.append(json.loads(p_st.stdout[p_st.stdout.rfind("{") :]))

    summary = {
        "seeds": seeds,
        "env_config": args.env_config,
        "checkpoint": args.checkpoint,
        "rllib": {
            "task_completion_pct": _mean_std(rllib_rows, "task_completion_pct"),
            "throughput_bps": _mean_std(rllib_rows, "throughput_bps"),
            "mean_aoi_steps": _mean_std(rllib_rows, "mean_aoi_steps"),
        },
        "static": {
            "task_completion_pct": _mean_std(static_rows, "task_completion_pct"),
            "throughput_bps": _mean_std(static_rows, "throughput_bps"),
            "mean_aoi_steps": _mean_std(static_rows, "mean_aoi_steps"),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
