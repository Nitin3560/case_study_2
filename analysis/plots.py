import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_method(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("method", "unknown"))].append(r)
    return groups


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _bar_with_error(ax, labels, means, stds, title, ylabel, ylim=None):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_all(log_path: str, out_dir: str) -> None:
    rows = _load_jsonl(log_path)
    if not rows:
        raise RuntimeError(f"No rows found in {log_path}")

    groups = _group_by_method(rows)
    methods = sorted(groups.keys())

    # 1) Task completion
    comp_means = []
    comp_stds = []
    for m in methods:
        vals = [r.get("task_completion_pct", 0.0) for r in groups[m]]
        mean, std = _mean_std(vals)
        comp_means.append(mean)
        comp_stds.append(std)
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_with_error(
        ax,
        methods,
        comp_means,
        comp_stds,
        "Task Completion (mean ± std)",
        "Completion (%)",
        ylim=(0, 100),
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task_completion_by_method.png"), dpi=150)
    plt.close(fig)

    # 2) Throughput
    thr_means = []
    thr_stds = []
    for m in methods:
        vals = [r.get("throughput_bps", 0.0) for r in groups[m]]
        mean, std = _mean_std(vals)
        thr_means.append(mean)
        thr_stds.append(std)
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_with_error(
        ax,
        methods,
        thr_means,
        thr_stds,
        "Throughput (mean ± std)",
        "Throughput (bps)",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_by_method.png"), dpi=150)
    plt.close(fig)

    # 3) AoI (mean + p95)
    aoi_means = []
    aoi_stds = []
    p95_means = []
    p95_stds = []
    for m in methods:
        vals = [r.get("mean_aoi_time_avg", r.get("mean_aoi_steps", 0.0)) for r in groups[m]]
        mean, std = _mean_std(vals)
        aoi_means.append(mean)
        aoi_stds.append(std)
        vals_p = [r.get("p95_aoi_time_avg", r.get("p95_aoi_steps", 0.0)) for r in groups[m]]
        mean_p, std_p = _mean_std(vals_p)
        p95_means.append(mean_p)
        p95_stds.append(std_p)
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - width / 2, aoi_means, width, yerr=aoi_stds, capsize=4, label="Mean AoI", color="#72b7b2")
    ax.bar(x + width / 2, p95_means, width, yerr=p95_stds, capsize=4, label="P95 AoI", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_title("AoI (mean and p95, mean ± std)")
    ax.set_ylabel("AoI (steps)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aoi_by_method.png"), dpi=150)
    plt.close(fig)

    # 4) AoI over time (if timeseries logs exist)
    ts_dir = os.path.join(os.path.dirname(log_path), "")
    ts_files = [f for f in os.listdir(ts_dir) if f.startswith("timeseries_") and f.endswith(".jsonl")]
    if ts_files:
        series = defaultdict(lambda: defaultdict(list))
        for fname in ts_files:
            with open(os.path.join(ts_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    m = r.get("method", "unknown")
                    step = int(r.get("step", 0))
                    series[m][step].append(float(r.get("mean_aoi_steps", 0.0)))
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for m in sorted(series.keys()):
            steps = sorted(series[m].keys())
            mean_vals = [np.mean(series[m][s]) for s in steps]
            ax.plot(steps, mean_vals, label=m)
        ax.set_title("AoI Over Time (mean over seeds)")
        ax.set_xlabel("Step")
        ax.set_ylabel("AoI (steps)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "aoi_over_time.png"), dpi=150)
        plt.close(fig)

    # 5) Per-agent throughput (mean over seeds)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m in methods:
        per_agent = [r.get("throughput_bps_per_agent", []) for r in groups[m]]
        if not per_agent:
            continue
        max_agents = max(len(x) for x in per_agent)
        # pad missing
        padded = [x + [0.0] * (max_agents - len(x)) for x in per_agent]
        arr = np.asarray(padded, dtype=float)
        mean_agent = np.mean(arr, axis=0)
        ax.plot(np.arange(len(mean_agent)), mean_agent, marker="o", label=m)
    ax.set_title("Per-Agent Throughput (mean over seeds)")
    ax.set_xlabel("Agent index")
    ax.set_ylabel("Throughput (bps)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_agent_throughput.png"), dpi=150)
    plt.close(fig)


def main():
    root = _repo_root()
    log_path = os.path.join(root, "outputs", "logs", "eval_results.jsonl")
    out_dir = os.path.join(root, "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_all(log_path, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
