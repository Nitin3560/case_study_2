from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from pettingzoo import ParallelEnv
except Exception as e:  # pragma: no cover
    ParallelEnv = object  # type: ignore

import gymnasium as gym
from gymnasium import spaces

from .task_models import Task
from .network_models import in_range
from .energy_models import move_energy_j, sensing_energy_j, tx_energy_j
from .reward import RewardWeights, team_reward


@dataclass
class EnvCfg:
    map_xy_m: tuple[float, float]
    horizon_steps: int
    dt_s: float

    num_agents: int
    num_tasks_total: int

    spawn_fracs: list[float]
    spawn_counts: list[int]

    visit_radius_m: float
    dwell_steps: int
    bytes_per_update: int
    update_period_steps: int

    comm_range_m: float
    base_xy: tuple[float, float]
    base_range_m: float
    link_capacity_bytes_per_step: int
    buffer_bytes_max: int

    v_max_mps: float

    e0_j: float
    move_cost_j_per_m: float
    sense_cost_j_per_step: float
    tx_cost_j_per_byte: float
    low_energy_j: float

    seed: int = 1


class LawnTaskCommParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv.
    - Agents move in 2D, complete tasks (visit + dwell), and relay periodic updates to base.
    - AoI is meaningful because active tasks generate periodic updates after activation.
    """

    metadata = {"name": "lawn_task_comm_v0"}

    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        self.n = cfg.num_agents
        self.possible_agents = [f"uav_{i}" for i in range(self.n)]
        self.agents = []

        self.rng = np.random.default_rng(cfg.seed)

        # state
        self.step_i = 0
        self.agent_pos = np.zeros((self.n, 2), dtype=np.float32)
        self.agent_energy = np.zeros((self.n,), dtype=np.float32)
        self.agent_buffer_bytes = np.zeros((self.n,), dtype=np.int32)
        self.agent_task = -np.ones((self.n,), dtype=np.int32)  # -1 none

        self.base_xy = np.array(cfg.base_xy, dtype=np.float32)

        self.tasks: list[Task] = []
        self.reward_w = RewardWeights()

        # spaces
        self._obs_space = self._make_obs_space()
        self._act_space = self._make_act_space()

    # --- spaces ---
    def _make_obs_space(self) -> gym.Space:
        # Minimal local obs: self pos(2), energy(1), buffer(1), base_in_range(1),
        # nearest K tasks: rel pos (K*2), status (K), active flag(K)
        K = 5
        dim = 2 + 1 + 1 + 1 + (K * 2) + (K * 1) + (K * 1)
        return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _make_act_space(self) -> gym.Space:
        # MultiDiscrete: [task_choice 0..num_tasks (0=keep), move_dir 0..7, comm 0..(N+1)]
        # comm: 0=hold, 1=send_to_base, 2..N+1 send_to_neighbor_(idx)
        return spaces.MultiDiscrete([self.cfg.num_tasks_total + 1, 8, self.n + 2])

    def observation_space(self, agent: str) -> gym.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gym.Space:
        return self._act_space

    # --- reset/step ---
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_i = 0
        self.agents = self.possible_agents[:]

        # init agents positions around center
        w, h = self.cfg.map_xy_m
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        jitter = self.rng.normal(0.0, 3.0, size=(self.n, 2)).astype(np.float32)
        self.agent_pos = center + jitter
        self.agent_energy = np.full((self.n,), float(self.cfg.e0_j), dtype=np.float32)
        self.agent_buffer_bytes = np.zeros((self.n,), dtype=np.int32)
        self.agent_task = -np.ones((self.n,), dtype=np.int32)

        # tasks with dynamic arrivals
        self.tasks = self._spawn_tasks()

        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _spawn_tasks(self) -> list[Task]:
        cfg = self.cfg
        w, h = cfg.map_xy_m

        total = cfg.num_tasks_total
        scheduled = list(zip(cfg.spawn_fracs, cfg.spawn_counts))
        late_total = sum(cfg.spawn_counts)
        t0_total = total - late_total
        assert t0_total >= 0, "spawn_counts exceed total tasks"

        appears = [0] * t0_total
        for frac, cnt in scheduled:
            step = int(round(frac * cfg.horizon_steps))
            appears += [step] * cnt
        appears = appears[:total]

        tasks: list[Task] = []
        for tid in range(total):
            pos = np.array([self.rng.uniform(0, w), self.rng.uniform(0, h)], dtype=np.float32)
            tasks.append(Task(tid=tid, pos_xy=pos, appears_step=appears[tid], dwell_left=cfg.dwell_steps))
        return tasks

    def step(self, actions: dict):
        # Convert action dict to arrays
        act_arr = np.zeros((self.n, 3), dtype=np.int32)
        for i, name in enumerate(self.possible_agents):
            if name in self.agents:
                act_arr[i] = np.array(actions.get(name, [0, 0, 0]), dtype=np.int32)

        # Activate tasks that appear now
        for t in self.tasks:
            t.maybe_activate(self.step_i)

        # Apply actions: task select, movement, comm
        energy_used_j = 0.0
        tasks_completed_now = 0
        bytes_delivered_now = 0

        # 1) task selection
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            task_choice = int(act_arr[i, 0])
            if task_choice > 0:
                self.agent_task[i] = task_choice - 1  # 0-based
            # keep if 0

        # 2) movement
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            move_dir = int(act_arr[i, 1])
            delta = self._dir_to_delta(move_dir)
            # clamp by v_max * dt
            step_len = float(self.cfg.v_max_mps * self.cfg.dt_s)
            delta_xy = step_len * delta
            old = self.agent_pos[i].copy()
            self.agent_pos[i] = self._clip_pos(self.agent_pos[i] + delta_xy)

            e = move_energy_j(self.cfg.move_cost_j_per_m, self.agent_pos[i] - old)
            self.agent_energy[i] -= e
            energy_used_j += e

        # 3) sensing progress
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            tid = int(self.agent_task[i])
            if tid < 0 or tid >= len(self.tasks):
                continue
            task = self.tasks[tid]
            if (not task.active) or task.completed:
                continue
            if task.within_visit_radius(self.agent_pos[i], self.cfg.visit_radius_m):
                # consume dwell
                if task.dwell_left > 0:
                    task.dwell_left -= 1
                    e = sensing_energy_j(self.cfg.sense_cost_j_per_step, 1)
                    self.agent_energy[i] -= e
                    energy_used_j += e
                if task.dwell_left == 0 and (not task.completed):
                    task.completed = True
                    tasks_completed_now += 1
                    # On completion: mark stream generation start
                    task.last_generated_step = self.step_i

        # 4) generate periodic task updates into buffers (for active tasks)
        for t in self.tasks:
            if not t.active:
                continue
            # if task not yet generated (including t=0 activation) set baseline
            if t.last_generated_step is None:
                t.last_generated_step = self.step_i
            # periodic updates
            if (self.step_i - t.last_generated_step) >= self.cfg.update_period_steps:
                t.last_generated_step = self.step_i
                # assign update to closest UAV (local pickup abstraction)
                j = int(np.argmin(np.linalg.norm(self.agent_pos - t.pos_xy[None, :], axis=1)))
                self.agent_buffer_bytes[j] = min(
                    self.cfg.buffer_bytes_max,
                    int(self.agent_buffer_bytes[j] + self.cfg.bytes_per_update),
                )

        # 5) comm transfer
        # Each agent can send up to link_capacity per step (either to base, or to neighbor)
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            comm = int(act_arr[i, 2])
            if self.agent_buffer_bytes[i] <= 0:
                continue

            capacity = int(self.cfg.link_capacity_bytes_per_step)
            send_bytes = min(int(self.agent_buffer_bytes[i]), capacity)

            if comm == 0:
                continue  # hold

            if comm == 1:
                # send to base if in range
                if in_range(self.agent_pos[i], self.base_xy, self.cfg.base_range_m):
                    self.agent_buffer_bytes[i] -= send_bytes
                    bytes_delivered_now += send_bytes
                    energy_used_j += tx_energy_j(self.cfg.tx_cost_j_per_byte, send_bytes)
                    self.agent_energy[i] -= tx_energy_j(self.cfg.tx_cost_j_per_byte, send_bytes)
                    # update AoI receive for all tasks that generated earlier (approx: delivery refresh)
                    for t in self.tasks:
                        if t.active:
                            t.last_received_step = self.step_i
                continue

            # send to neighbor k
            k = comm - 2
            if 0 <= k < self.n:
                if in_range(self.agent_pos[i], self.agent_pos[k], self.cfg.comm_range_m):
                    self.agent_buffer_bytes[i] -= send_bytes
                    self.agent_buffer_bytes[k] = min(
                        self.cfg.buffer_bytes_max,
                        int(self.agent_buffer_bytes[k] + send_bytes),
                    )
                    energy_used_j += tx_energy_j(self.cfg.tx_cost_j_per_byte, send_bytes)
                    self.agent_energy[i] -= tx_energy_j(self.cfg.tx_cost_j_per_byte, send_bytes)

        # terminations
        self.step_i += 1
        done_time = self.step_i >= self.cfg.horizon_steps
        # if all energy dead, truncate
        dead = np.all(self.agent_energy <= 0.0)

        terminations = {a: bool(done_time) for a in self.agents}
        truncations = {a: bool(dead and not done_time) for a in self.agents}

        # reward (team -> broadcast)
        mean_aoi = self._mean_aoi()
        r_team = team_reward(tasks_completed_now, bytes_delivered_now, mean_aoi, energy_used_j, self.reward_w)
        rewards = {a: float(r_team) for a in self.agents}

        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        infos = {a: {
            "tasks_completed_now": tasks_completed_now,
            "bytes_delivered_now": bytes_delivered_now,
            "mean_aoi": mean_aoi,
            "energy_used_j": float(energy_used_j),
        } for a in self.agents}

        if done_time or dead:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # --- helpers ---
    def _clip_pos(self, xy: np.ndarray) -> np.ndarray:
        w, h = self.cfg.map_xy_m
        xy[0] = float(np.clip(xy[0], 0.0, w))
        xy[1] = float(np.clip(xy[1], 0.0, h))
        return xy

    def _dir_to_delta(self, move_dir: int) -> np.ndarray:
        # 8-connected directions
        dirs = np.array([
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
        ], dtype=np.float32)
        d = dirs[int(np.clip(move_dir, 0, 7))]
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else d

    def _obs_for(self, i: int) -> np.ndarray:
        cfg = self.cfg
        K = 5
        w, h = cfg.map_xy_m
        pos = self.agent_pos[i].astype(np.float32)

        base_in = 1.0 if in_range(pos, self.base_xy, cfg.base_range_m) else 0.0
        energy = float(self.agent_energy[i]) / float(cfg.e0_j)
        buf = float(self.agent_buffer_bytes[i]) / float(cfg.buffer_bytes_max)

        # nearest active tasks
        active_tasks = [t for t in self.tasks if t.active]
        if active_tasks:
            dists = np.array([np.linalg.norm(pos - t.pos_xy) for t in active_tasks], dtype=np.float32)
            idx = np.argsort(dists)[:K]
            chosen = [active_tasks[j] for j in idx]
        else:
            chosen = []

        rels = []
        status = []
        act = []
        for t in chosen:
            rel = (t.pos_xy - pos) / np.array([w, h], dtype=np.float32)
            rels.extend([float(rel[0]), float(rel[1])])
            status.append(1.0 if t.completed else 0.0)
            act.append(1.0)

        # pad
        while len(rels) < K * 2:
            rels.extend([0.0, 0.0])
        while len(status) < K:
            status.append(0.0)
        while len(act) < K:
            act.append(0.0)

        obs = np.array(
            [
                (pos[0] / w) * 2 - 1,
                (pos[1] / h) * 2 - 1,
                energy * 2 - 1,
                buf * 2 - 1,
                base_in * 2 - 1,
                *rels,
                *status,
                *act,
            ],
            dtype=np.float32,
        )
        return obs

    def _mean_aoi(self) -> float:
        # AoI over active tasks only. If never received, treat AoI as (step - appears).
        aois = []
        for t in self.tasks:
            if not t.active:
                continue
            if t.last_received_step is None:
                aois.append(float(self.step_i - t.appears_step))
            else:
                aois.append(float(self.step_i - t.last_received_step))
        return float(np.mean(aois)) if aois else 0.0