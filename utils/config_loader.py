from __future__ import annotations
from dataclasses import dataclass
import yaml

from case_study_2.envs.lawn_task_comm_env import EnvCfg


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_cfg(env_yaml: dict, seed: int) -> EnvCfg:
    e = env_yaml["env"]

    spawn = e["task_spawn_schedule"]
    users = e["user_spawn_schedule"]
    sensing = e["sensing"]
    comm = e["comm"]
    motion = e["motion"]
    energy = e["energy"]
    user_cfg = e["users"]

    return EnvCfg(
        map_xy_m=tuple(e["map_xy_m"]),
        horizon_steps=int(e["horizon_steps"]),
        dt_s=float(e["dt_s"]),
        num_agents=int(e["num_agents"]),
        num_tasks_total=int(e["num_tasks_total"]),
        num_users_total=int(e["num_users_total"]),
        spawn_fracs=list(spawn["spawn_fracs"]),
        spawn_counts=list(spawn["spawn_counts"]),
        user_spawn_fracs=list(users["spawn_fracs"]),
        user_spawn_counts=list(users["spawn_counts"]),
        visit_radius_m=float(sensing["visit_radius_m"]),
        dwell_steps=int(sensing["dwell_steps"]),
        bytes_per_update=int(sensing["bytes_per_update"]),
        update_period_steps=int(sensing["update_period_steps"]),
        user_cov_radius_m=float(user_cfg["cov_radius_m"]),
        user_bytes_per_step=int(user_cfg["bytes_per_step"]),
        user_speed_mps=float(user_cfg["speed_mps"]),
        comm_range_m=float(comm["comm_range_m"]),
        base_xy=tuple(comm["base_pos_xy"]),
        base_range_m=float(comm["base_range_m"]),
        link_capacity_bytes_per_step=int(comm["link_capacity_bytes_per_step"]),
        buffer_bytes_max=int(comm["buffer_bytes_max"]),
        v_max_mps=float(motion["v_max_mps"]),
        e0_j=float(energy["e0_j"]),
        move_cost_j_per_m=float(energy["move_cost_j_per_m"]),
        sense_cost_j_per_step=float(energy["sense_cost_j_per_step"]),
        tx_cost_j_per_byte=float(energy["tx_cost_j_per_byte"]),
        low_energy_j=float(energy["low_energy_j"]),
        seed=int(seed),
    )
