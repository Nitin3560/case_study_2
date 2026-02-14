from __future__ import annotations

import os
import json
import copy
import tempfile
import yaml

from case_study_2.utils.config_loader import load_yaml
from case_study_2.evaluation.eval_runner import run_one, _repo_root


def _write_temp_cfg(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def _run_suite(name: str, cfg: dict, methods: list[str], seeds: list[int]) -> list[dict]:
    results = []
    cfg_path = _write_temp_cfg(cfg)
    for m in methods:
        for s in seeds:
            try:
                r = run_one(seed=s, method=m, cfg_path_env=cfg_path)
                r["suite"] = name
                results.append(r)
            except Exception as e:
                # Skip incompatible checkpoints (e.g., task count changes)
                results.append({
                    "suite": name,
                    "method": m,
                    "seed": s,
                    "error": str(e),
                })
    os.remove(cfg_path)
    return results


def main():
    root = _repo_root()
    base_cfg = load_yaml(os.path.join(root, "configs", "env.yaml"))

    methods = ["static", "fixed_relay", "greedy", "mappo_full", "mappo_task_only", "mappo_routing_only"]
    seeds = [1, 2, 3]

    all_results = []

    # 1) Energy stress: +/- 20%
    for scale in [0.8, 1.0, 1.2]:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["energy"]["e0_j"] = float(cfg["env"]["energy"]["e0_j"]) * scale
        name = f"energy_{int(scale*100)}pct"
        all_results.extend(_run_suite(name, cfg, methods, seeds))

    # 2) Bandwidth stress: half / base
    for scale in [0.5, 1.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["comm"]["link_capacity_bytes_per_step"] = int(cfg["env"]["comm"]["link_capacity_bytes_per_step"] * scale)
        name = f"bandwidth_{int(scale*100)}pct"
        all_results.extend(_run_suite(name, cfg, methods, seeds))

    # 3) Dynamic task count
    for extra in [0, 2, 4]:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["num_tasks_total"] = int(cfg["env"]["num_tasks_total"]) + extra
        name = f"tasks_{cfg['env']['num_tasks_total']}"
        # Skip MARL checkpoints when task count changes (action head size mismatch)
        meth = methods if extra == 0 else ["static", "fixed_relay", "greedy"]
        all_results.extend(_run_suite(name, cfg, meth, seeds))

    # 4) Map size (generalization)
    for size in [220.0, 260.0, 300.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["map_xy_m"] = [float(size), float(size)]
        cfg["env"]["comm"]["base_pos_xy"] = [float(size/2.0), float(size/2.0)]
        name = f"map_{int(size)}"
        all_results.extend(_run_suite(name, cfg, methods, seeds))

    # 5) Comm range (generalization)
    for rng in [30.0, 45.0, 60.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["comm"]["comm_range_m"] = float(rng)
        name = f"comm_{int(rng)}"
        all_results.extend(_run_suite(name, cfg, methods, seeds))

    out_dir = os.path.join(root, "outputs", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stress_tests.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote stress test results to {out_path}")


if __name__ == "__main__":
    main()
