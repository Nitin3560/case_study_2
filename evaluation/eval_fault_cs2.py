from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "case2_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from case_study_2.evaluation.eval_runner import run_one
from case_study_2.evaluation.fault_injector import FaultConfig, FaultInjector
from case_study_2.utils.config_loader import build_env_cfg, load_yaml


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _write_summary_csv(path: str, rows: list[dict]) -> None:
    fields = [
        "Method",
        "Pre Error",
        "Post Error",
        "Post vs Pre %",
        "Connectivity Pre",
        "Connectivity Post",
        "Connectivity %",
        "Utility Pre",
        "Utility Post",
        "Utility Δ",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_timeline(path: str, timeline_rows: list[dict], fault_type: str, t0: int, duration: int) -> None:
    by_method: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in timeline_rows:
        by_method[r["method"]][int(r["t"])].append((float(r["mean_tracking_error"]), float(r["connectivity"])))

    methods = sorted(by_method.keys())
    if not methods:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for m in methods:
        steps = sorted(by_method[m].keys())
        err = []
        conn = []
        for s in steps:
            vals = by_method[m][s]
            err.append(float(np.mean([v[0] for v in vals])))
            conn.append(float(np.mean([v[1] for v in vals])))
        axes[0].plot(steps, err, label=m)
        axes[1].plot(steps, conn, label=m)

    for ax in axes:
        ax.axvline(t0, color="black", linestyle="--", linewidth=1)
        ax.axvline(t0 + duration, color="black", linestyle=":", linewidth=1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Tracking Error (m)")
    axes[1].set_ylabel("Connectivity")
    axes[1].set_xlabel("Step")
    axes[0].set_title(f"Case Study 2 Fault Timeline ({fault_type})")
    axes[0].legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def run_fault_suite(
    env_config: str,
    methods: list[str],
    seeds: list[int],
    fault_type: str,
    t0: int,
    duration: int,
    severity: float,
    out_dir: str,
    ckpt: str | None,
    apply_wrapper_to_rl: bool,
    fault_window_steps: int,
) -> dict:
    env_yaml = load_yaml(env_config)
    timeline_rows: list[dict] = []
    per_method: dict[str, list[dict]] = defaultdict(list)

    for seed in seeds:
        cfg = build_env_cfg(env_yaml, seed=seed)
        fi = FaultInjector(FaultConfig(
            num_agents=cfg.num_agents,
            horizon_steps=cfg.horizon_steps,
            fault_type=fault_type,
            t0=t0,
            duration=duration,
            severity=severity,
            seed=seed,
        ))
        for method in methods:
            use_wrapper = bool(apply_wrapper_to_rl and method in {"mappo", "mappo_full"})
            r = run_one(
                seed=seed,
                method=method,
                cfg_path_env=env_config,
                ckpt_path=ckpt,
                fault_injector=fi,
                fault_action_wrapper=use_wrapper,
                fault_log_steps=True,
                fault_window_steps=fault_window_steps,
            )
            per_method[method].append(r)
            for row in r.get("fault_step_logs", []):
                timeline_rows.append(row)

    summary_rows: list[dict] = []
    for method in methods:
        rows = per_method.get(method, [])
        summary_rows.append({
            "Method": method,
            "Pre Error": _mean([float(r.get("pre_error", 0.0)) for r in rows]),
            "Post Error": _mean([float(r.get("post_error", 0.0)) for r in rows]),
            "Post vs Pre %": _mean([float(r.get("post_vs_pre_error_pct", 0.0)) for r in rows]),
            "Connectivity Pre": _mean([float(r.get("connectivity_pre", 0.0)) for r in rows]),
            "Connectivity Post": _mean([float(r.get("connectivity_post", 0.0)) for r in rows]),
            "Connectivity %": _mean([float(r.get("connectivity_pct", 0.0)) for r in rows]),
            "Utility Pre": _mean([float(r.get("utility_pre", 0.0)) for r in rows]),
            "Utility Post": _mean([float(r.get("utility_post", 0.0)) for r in rows]),
            "Utility Δ": _mean([float(r.get("utility_delta", 0.0)) for r in rows]),
        })

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"summary_fault_degradation_{fault_type}_cs2.csv")
    _write_summary_csv(csv_path, summary_rows)

    timeline_png = os.path.join(out_dir, f"fault_timeline_error_connectivity_{fault_type}_cs2.png")
    _plot_timeline(timeline_png, timeline_rows, fault_type=fault_type, t0=t0, duration=duration)

    raw_json = os.path.join(out_dir, f"fault_runs_{fault_type}_cs2.json")
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump({"fault_type": fault_type, "runs": per_method}, f, indent=2)

    return {
        "fault_type": fault_type,
        "summary_csv": csv_path,
        "timeline_png": timeline_png,
        "raw_json": raw_json,
    }


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env_case2_publication.yaml"))
    ap.add_argument("--methods", default="static,fixed_relay,greedy,mappo_full")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--fault-types", default="dropout,sensor,comm")
    ap.add_argument("--fault-t0", type=int, default=100)
    ap.add_argument("--fault-duration", type=int, default=60)
    ap.add_argument("--fault-severity", type=float, default=0.5)
    ap.add_argument("--fault-window-steps", type=int, default=50)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--apply-wrapper-to-rl", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs", "fault_cs2"))
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    fault_types = [f.strip() for f in args.fault_types.split(",") if f.strip()]

    outputs = []
    for ft in fault_types:
        outputs.append(
            run_fault_suite(
                env_config=args.env_config,
                methods=methods,
                seeds=seeds,
                fault_type=ft,
                t0=args.fault_t0,
                duration=args.fault_duration,
                severity=args.fault_severity,
                out_dir=args.out_dir,
                ckpt=args.ckpt,
                apply_wrapper_to_rl=args.apply_wrapper_to_rl,
                fault_window_steps=args.fault_window_steps,
            )
        )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
