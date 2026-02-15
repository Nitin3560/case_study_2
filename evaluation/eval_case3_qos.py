from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from case_study_2.evaluation.eval_runner import run_one


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mean_std(rows: list[dict], key: str) -> dict:
    vals = [float(r.get(key, 0.0)) for r in rows]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _read_ts(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _plot_time_series(ts_by_method: dict[str, list[list[dict]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key, ylabel, name in [
        ("latency_ms_mean_now", "Latency (ms)", "latency_vs_time_case3.png"),
        ("throughput_bps_now", "Throughput (bps)", "throughput_vs_time_case3.png"),
    ]:
        plt.figure(figsize=(9, 4.5))
        for method, runs in ts_by_method.items():
            step_to_vals: dict[int, list[float]] = defaultdict(list)
            for run in runs:
                for r in run:
                    step_to_vals[int(r["step"])].append(float(r.get(key, 0.0)))
            steps = sorted(step_to_vals.keys())
            if not steps:
                continue
            y = np.array([np.mean(step_to_vals[s]) for s in steps], dtype=np.float32)
            plt.plot(steps, y, label=method)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), dpi=180)
        plt.close()


def _plot_latency_cdf(ts_by_method: dict[str, list[list[dict]]], out_dir: str) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for method, runs in ts_by_method.items():
        vals = []
        for run in runs:
            vals.extend([float(r.get("latency_ms_mean_now", 0.0)) for r in run if float(r.get("latency_ms_mean_now", 0.0)) > 0])
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


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_case2_publication.yaml"))
    ap.add_argument("--methods", default="static,mappo_full")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs", "case3_qos"))
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    ts_dir = os.path.join(args.out_dir, "timeseries")
    os.makedirs(ts_dir, exist_ok=True)

    all_rows: dict[str, list[dict]] = {m: [] for m in methods}
    ts_by_method: dict[str, list[list[dict]]] = {m: [] for m in methods}

    for method in methods:
        for seed in seeds:
            r = run_one(
                seed=seed,
                method=method,
                cfg_path_env=args.env_config,
                ckpt_path=args.ckpt,
                log_timeseries=True,
                timeseries_dir=ts_dir,
            )
            all_rows[method].append(r)
            ts_path = os.path.join(ts_dir, f"timeseries_{method}_seed{seed}.jsonl")
            ts_by_method[method].append(_read_ts(ts_path))

    summary: dict[str, dict] = {}
    for method in methods:
        rows = all_rows[method]
        thr_step_std = []
        for run_ts in ts_by_method[method]:
            vals = [float(x.get("throughput_bps_now", 0.0)) for x in run_ts]
            thr_step_std.append(float(np.std(vals)) if vals else 0.0)

        loss_rows = [{"loss": 100.0 - float(r.get("delivery_rate_pct", 0.0))} for r in rows]
        summary[method] = {
            "mean_latency_ms": _mean_std(rows, "mean_delivery_latency_ms"),
            "p95_latency_ms": _mean_std(rows, "p95_delivery_latency_ms"),
            "mean_throughput_bps": _mean_std(rows, "throughput_bps"),
            "throughput_std_over_time": {"mean": float(np.mean(thr_step_std)) if thr_step_std else 0.0, "std": float(np.std(thr_step_std)) if thr_step_std else 0.0},
            "delivery_rate_pct": _mean_std(rows, "delivery_rate_pct"),
            "packet_loss_pct": _mean_std(loss_rows, "loss"),
        }

    with open(os.path.join(args.out_dir, "case3_qos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_path = os.path.join(args.out_dir, "case3_qos_table.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Latency Mean±Std (ms)", "Throughput Mean±Std (bps)", "Throughput Std-over-time Mean±Std", "Delivery Mean±Std (%)", "Packet Loss Mean±Std (%)"])
        for m in methods:
            s = summary[m]
            w.writerow([
                m,
                f"{s['mean_latency_ms']['mean']:.2f} ± {s['mean_latency_ms']['std']:.2f}",
                f"{s['mean_throughput_bps']['mean']:.2f} ± {s['mean_throughput_bps']['std']:.2f}",
                f"{s['throughput_std_over_time']['mean']:.2f} ± {s['throughput_std_over_time']['std']:.2f}",
                f"{s['delivery_rate_pct']['mean']:.2f} ± {s['delivery_rate_pct']['std']:.2f}",
                f"{s['packet_loss_pct']['mean']:.2f} ± {s['packet_loss_pct']['std']:.2f}",
            ])

    _plot_time_series(ts_by_method, args.out_dir)
    _plot_latency_cdf(ts_by_method, args.out_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
