from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from typing import Any

from case_study_2.evaluation.eval_runner import run_one

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows]
    if len(vals) <= 1:
        return (float(vals[0]) if vals else 0.0, 0.0)
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def _paired_pvalue(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    key: str,
    alternative: str,
) -> float | None:
    if wilcoxon is None:
        return None
    a = [float(r[key]) for r in sorted(a_rows, key=lambda x: x["seed"])]
    b = [float(r[key]) for r in sorted(b_rows, key=lambda x: x["seed"])]
    if len(a) != len(b) or len(a) == 0:
        return None
    try:
        res = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return None


def _run_rllib_eval(root: str, env_cfg: str, checkpoint: str, seed: int) -> dict[str, Any]:
    cmd = [
        "python",
        os.path.join(root, "evaluation", "eval_rllib_ppo_ctde.py"),
        "--env-config",
        env_cfg,
        "--checkpoint",
        checkpoint,
        "--seed",
        str(seed),
    ]
    p = subprocess.run(
        cmd,
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
    return json.loads(p.stdout[p.stdout.rfind("{") :])


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_easy.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs", "camera_ready_pack"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    rllib_rows = []
    static_rows = []
    fixed_rows = []
    greedy_rows = []

    for seed in seeds:
        rllib_rows.append(_run_rllib_eval(root, args.env_config, args.checkpoint, seed))
        static_rows.append(run_one(seed=seed, method="static", cfg_path_env=args.env_config))
        fixed_rows.append(run_one(seed=seed, method="fixed_relay", cfg_path_env=args.env_config))
        greedy_rows.append(run_one(seed=seed, method="greedy", cfg_path_env=args.env_config))

    payload = {
        "env_config": args.env_config,
        "checkpoint": args.checkpoint,
        "seeds": seeds,
        "metrics": {
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
            "fixed_relay": {
                "task_completion_pct": _mean_std(fixed_rows, "task_completion_pct"),
                "throughput_bps": _mean_std(fixed_rows, "throughput_bps"),
                "mean_aoi_steps": _mean_std(fixed_rows, "mean_aoi_steps"),
            },
            "greedy": {
                "task_completion_pct": _mean_std(greedy_rows, "task_completion_pct"),
                "throughput_bps": _mean_std(greedy_rows, "throughput_bps"),
                "mean_aoi_steps": _mean_std(greedy_rows, "mean_aoi_steps"),
            },
        },
        "pvalues_wilcoxon": {
            "rllib_vs_static": {
                "completion_greater": _paired_pvalue(rllib_rows, static_rows, "task_completion_pct", "greater"),
                "throughput_greater": _paired_pvalue(rllib_rows, static_rows, "throughput_bps", "greater"),
                "aoi_less": _paired_pvalue(rllib_rows, static_rows, "mean_aoi_steps", "less"),
            },
            "rllib_vs_fixed_relay": {
                "completion_greater": _paired_pvalue(rllib_rows, fixed_rows, "task_completion_pct", "greater"),
                "throughput_greater": _paired_pvalue(rllib_rows, fixed_rows, "throughput_bps", "greater"),
                "aoi_less": _paired_pvalue(rllib_rows, fixed_rows, "mean_aoi_steps", "less"),
            },
            "rllib_vs_greedy": {
                "completion_greater": _paired_pvalue(rllib_rows, greedy_rows, "task_completion_pct", "greater"),
                "throughput_greater": _paired_pvalue(rllib_rows, greedy_rows, "throughput_bps", "greater"),
                "aoi_less": _paired_pvalue(rllib_rows, greedy_rows, "mean_aoi_steps", "less"),
            },
        },
        "per_seed": {
            "rllib": rllib_rows,
            "static": static_rows,
            "fixed_relay": fixed_rows,
            "greedy": greedy_rows,
        },
    }

    out_json = os.path.join(args.out_dir, "camera_ready_pack.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    def fmt(ms: tuple[float, float]) -> str:
        return f"{ms[0]:.2f} +- {ms[1]:.2f}"

    out_md = os.path.join(args.out_dir, "camera_ready_pack.md")
    lines = []
    lines.append("# Camera-Ready Statistical Pack")
    lines.append("")
    lines.append(f"- env_config: `{args.env_config}`")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append("")
    lines.append("| Method | Completion (%) | Throughput (bps) | Mean AoI (steps) |")
    lines.append("|---|---:|---:|---:|")
    for m in ["rllib", "static", "fixed_relay", "greedy"]:
        mm = payload["metrics"][m]
        lines.append(
            f"| {m} | {fmt(mm['task_completion_pct'])} | {fmt(mm['throughput_bps'])} | {fmt(mm['mean_aoi_steps'])} |"
        )
    lines.append("")
    lines.append("## Wilcoxon p-values (one-sided)")
    lines.append("- Alternative hypotheses:")
    lines.append("  - completion/throughput: RLlib > baseline")
    lines.append("  - AoI: RLlib < baseline")
    lines.append("")
    for k, v in payload["pvalues_wilcoxon"].items():
        lines.append(f"- {k}: {json.dumps(v)}")
    lines.append("")
    lines.append(f"Raw JSON: `{out_json}`")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"json": out_json, "md": out_md}, indent=2))


if __name__ == "__main__":
    main()
