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
    num_users_total: int

    spawn_fracs: list[float]
    spawn_counts: list[int]
    user_spawn_fracs: list[float]
    user_spawn_counts: list[int]

    visit_radius_m: float
    dwell_steps: int
    bytes_per_update: int
    update_period_steps: int
    user_cov_radius_m: float
    user_bytes_per_step: int
    user_speed_mps: float

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
        # Per-agent buffers as {task_id: bytes}
        self.agent_buffer_bytes: list[dict[int, int]] = [dict() for _ in range(self.n)]
        self.agent_task = -np.ones((self.n,), dtype=np.int32)  # -1 none
        self.prev_task_dist = np.zeros((self.n,), dtype=np.float32)
        self.prev_base_dist = np.zeros((self.n,), dtype=np.float32)

        self.base_xy = np.array(cfg.base_xy, dtype=np.float32)

        self.tasks: list[Task] = []
        # user streams
        self.user_pos = np.zeros((cfg.num_users_total, 2), dtype=np.float32)
        self.user_active = np.zeros((cfg.num_users_total,), dtype=bool)
        self.user_appears = [0 for _ in range(cfg.num_users_total)]
        self.user_last_generated = [None for _ in range(cfg.num_users_total)]
        self.user_last_received = [None for _ in range(cfg.num_users_total)]
        self.user_delivered = [False for _ in range(cfg.num_users_total)]
        self.reward_w = RewardWeights()

        # spaces
        self._obs_space = self._make_obs_space()
        self._act_space = self._make_act_space()

    # --- spaces ---
    def _make_obs_space(self) -> gym.Space:
        # Minimal local obs: self pos(2), energy(1), buffer(1), current_task(1), base_in_range(1),
        # nearest K tasks: rel pos (K*2), status (K), active flag(K)
        # nearest U users: rel pos (U*2), active flag(U)
        # neighbors: top-N rel pos (N*2) + link quality (N)
        K = 5
        U = 3
        N = 3
        dim = 2 + 1 + 1 + 1 + 1 + (K * 2) + (K * 1) + (K * 1) + (U * 2) + (U * 1) + (N * 2) + (N * 1)
        return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _make_act_space(self) -> gym.Space:
        # MultiDiscrete: [task_choice 0..num_tasks (0=keep), move_dir 0..7, comm 0..(N+1), mode 0..1]
        # comm: 0=hold, 1=send_to_base, 2..N+1 send_to_neighbor_(idx)
        return spaces.MultiDiscrete([self.cfg.num_tasks_total + 1, 8, self.n + 2, 2])

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
        jitter = self.rng.normal(0.0, 5.0, size=(self.n, 2)).astype(np.float32)
        self.agent_pos = center + jitter
        self.agent_energy = np.full((self.n,), float(self.cfg.e0_j), dtype=np.float32)
        self.agent_buffer_bytes = [dict() for _ in range(self.n)]
        self.agent_task = -np.ones((self.n,), dtype=np.int32)
        self.prev_task_dist = np.zeros((self.n,), dtype=np.float32)
        self.prev_base_dist = np.linalg.norm(self.agent_pos - self.base_xy[None, :], axis=1).astype(np.float32)

        # tasks with dynamic arrivals
        self.tasks = self._spawn_tasks()
        self._spawn_users()

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

        # grid + jitter for mild stochasticity (stable but not fixed)
        grid_n = int(np.ceil(np.sqrt(max(total, 1))))
        spacing_x = float(w) / float(grid_n + 1)
        spacing_y = float(h) / float(grid_n + 1)
        grid_points = []
        for gy in range(grid_n):
            for gx in range(grid_n):
                grid_points.append(
                    np.array([(gx + 1) * spacing_x, (gy + 1) * spacing_y], dtype=np.float32)
                )
        tasks: list[Task] = []
        for tid in range(total):
            base = grid_points[tid % len(grid_points)].copy()
            jitter = self.rng.normal(0.0, 8.0, size=(2,)).astype(np.float32)
            pos = self._clip_pos(base + jitter)
            tasks.append(Task(tid=tid, pos_xy=pos, appears_step=appears[tid], dwell_left=cfg.dwell_steps))
        return tasks

    def _spawn_users(self) -> None:
        cfg = self.cfg
        w, h = cfg.map_xy_m
        total = cfg.num_users_total
        scheduled = list(zip(cfg.user_spawn_fracs, cfg.user_spawn_counts))
        late_total = sum(cfg.user_spawn_counts)
        t0_total = total - late_total
        assert t0_total >= 0, "user_spawn_counts exceed total users"

        appears = [0] * t0_total
        for frac, cnt in scheduled:
            step = int(round(frac * cfg.horizon_steps))
            appears += [step] * cnt
        appears = appears[:total]

        self.user_active[:] = False
        grid_n = int(np.ceil(np.sqrt(max(total, 1))))
        spacing_x = float(w) / float(grid_n + 1)
        spacing_y = float(h) / float(grid_n + 1)
        grid_points = []
        for gy in range(grid_n):
            for gx in range(grid_n):
                grid_points.append(
                    np.array([(gx + 1) * spacing_x, (gy + 1) * spacing_y], dtype=np.float32)
                )
        for uid in range(total):
            base = grid_points[uid % len(grid_points)].copy()
            jitter = self.rng.normal(0.0, 8.0, size=(2,)).astype(np.float32)
            self.user_pos[uid] = self._clip_pos(base + jitter)
            self.user_appears[uid] = int(appears[uid])
            # store activation time in last_generated as negative for now
            self.user_last_generated[uid] = -int(appears[uid])
            self.user_last_received[uid] = None
            self.user_delivered[uid] = False

    def step(self, actions: dict):
        # Convert action dict to arrays
        act_arr = np.zeros((self.n, 4), dtype=np.int32)
        for i, name in enumerate(self.possible_agents):
            if name in self.agents:
                act_arr[i] = np.array(actions.get(name, [0, 0, 0, 0]), dtype=np.int32)

        # Activate tasks that appear now
        for t in self.tasks:
            t.maybe_activate(self.step_i)
        # Activate users that appear now
        for uid in range(self.cfg.num_users_total):
            if self.user_active[uid]:
                continue
            appear = -int(self.user_last_generated[uid])
            if self.step_i >= appear:
                self.user_active[uid] = True
                self.user_last_generated[uid] = None

        # Apply actions: task select, movement, comm
        energy_used_j = 0.0
        tasks_completed_now = 0
        tasks_delivered_now = 0
        dwell_progress_now = 0
        bytes_delivered_now = 0
        per_agent_delivered = np.zeros((self.n,), dtype=np.int32)
        bytes_generated_now = 0
        bytes_dropped_now = 0
        bytes_relayed_now = 0

        # 1) task selection
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            task_choice = int(act_arr[i, 0])
            if task_choice > 0:
                self.agent_task[i] = task_choice - 1  # 0-based
            # keep if 0

        # cache distances before movement for shaping
        pre_task_dist = np.full((self.n,), np.nan, dtype=np.float32)
        pre_nearest_task_dist = np.full((self.n,), np.nan, dtype=np.float32)
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            tid = int(self.agent_task[i])
            if tid >= 0 and tid < len(self.tasks):
                t = self.tasks[tid]
                if t.active and (not t.completed):
                    pre_task_dist[i] = float(np.linalg.norm(self.agent_pos[i] - t.pos_xy))
            # nearest active task (for shaping even if no assignment)
            nearest = None
            best_d = 1e18
            for t in self.tasks:
                if not t.active or t.completed:
                    continue
                d = float(np.linalg.norm(self.agent_pos[i] - t.pos_xy))
                if d < best_d:
                    best_d = d
                    nearest = t
            if nearest is not None:
                pre_nearest_task_dist[i] = float(best_d)
            pre_base = float(np.linalg.norm(self.agent_pos[i] - self.base_xy))
            self.prev_base_dist[i] = pre_base

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
        # First apply sensing energy to each agent within visit radius
        tasks_in_range: dict[int, bool] = {}
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            tid = int(self.agent_task[i])
            task = None
            if 0 <= tid < len(self.tasks):
                t = self.tasks[tid]
                if t.active and (not t.completed):
                    task = t
            # fallback: nearest active task if no valid assignment
            if task is None:
                best = None
                best_d = 1e18
                for t in self.tasks:
                    if not t.active or t.completed:
                        continue
                    d = float(np.linalg.norm(self.agent_pos[i] - t.pos_xy))
                    if d < best_d:
                        best_d = d
                        best = t
                task = best
                if task is None:
                    continue
                # adopt nearest task as current assignment
                self.agent_task[i] = int(task.tid)
                tid = int(task.tid)
            if task.within_visit_radius(self.agent_pos[i], self.cfg.visit_radius_m):
                e = sensing_energy_j(self.cfg.sense_cost_j_per_step, 1)
                self.agent_energy[i] -= e
                energy_used_j += e
                tasks_in_range[tid] = True

        # Then decrement dwell at most once per task per step
        for tid in tasks_in_range.keys():
            task = self.tasks[tid]
            if task.dwell_left > 0:
                task.dwell_left -= 1
                dwell_progress_now += 1
            if task.dwell_left == 0 and (not task.completed):
                task.completed = True
                tasks_completed_now += 1
                # On completion: mark stream generation start
                task.last_generated_step = self.step_i

        # 4) generate periodic task updates into buffers (for active tasks)
        for t in self.tasks:
            if not t.active:
                continue
            if not t.completed:
                continue
            # if task not yet generated (including t=0 activation) set baseline
            if t.last_generated_step is None:
                t.last_generated_step = self.step_i
            # periodic updates
            if (self.step_i - t.last_generated_step) >= self.cfg.update_period_steps:
                t.last_generated_step = self.step_i
                # assign update to closest UAV (local pickup abstraction)
                j = int(np.argmin(np.linalg.norm(self.agent_pos - t.pos_xy[None, :], axis=1)))
                gen, drop = self._buffer_add(j, t.tid, self.cfg.bytes_per_update)
                bytes_generated_now += gen
                bytes_dropped_now += drop

        # 4b) user coverage + data generation (scan mode only)
        for uid in range(self.cfg.num_users_total):
            if not self.user_active[uid]:
                continue
            # random walk users
            if self.cfg.user_speed_mps > 0:
                ang = self.rng.uniform(0, 2 * np.pi)
                step = float(self.cfg.user_speed_mps * self.cfg.dt_s)
                delta = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32) * step
                self.user_pos[uid] = self._clip_pos(self.user_pos[uid] + delta)

            for i in range(self.n):
                if self.possible_agents[i] not in self.agents:
                    continue
                # Mode is automatically derived from local buffer state.
                mode = 1 if self._buffer_total(i) > 0 else 0  # 0=scan, 1=relay
                if mode != 0:
                    continue
                if float(np.linalg.norm(self.agent_pos[i] - self.user_pos[uid])) <= float(self.cfg.user_cov_radius_m):
                    stream_id = self.cfg.num_tasks_total + uid
                    gen, drop = self._buffer_add(i, stream_id, self.cfg.user_bytes_per_step)
                    bytes_generated_now += gen
                    bytes_dropped_now += drop
                    self.user_last_generated[uid] = self.step_i

        # 5) comm transfer
        # Each agent can send up to link_capacity per step (either to base, or to neighbor)
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            comm = int(act_arr[i, 2])
            # Mode is automatically derived from local buffer state.
            mode = 1 if self._buffer_total(i) > 0 else 0  # 0=scan, 1=relay
            if mode == 0:
                continue
            if self._buffer_total(i) <= 0:
                continue

            capacity = int(self.cfg.link_capacity_bytes_per_step)
            send_bytes = self._buffer_take_limit(i, capacity)
            if send_bytes <= 0:
                continue

            if comm == 0:
                continue  # hold

            if comm == 1:
                # send to base if in range
                if in_range(self.agent_pos[i], self.base_xy, self.cfg.base_range_m):
                    delivered = self._buffer_take(i, send_bytes)
                    sent_bytes = int(sum(delivered.values()))
                    bytes_delivered_now += sent_bytes
                    per_agent_delivered[i] += sent_bytes
                    energy_used_j += tx_energy_j(self.cfg.tx_cost_j_per_byte, sent_bytes)
                    self.agent_energy[i] -= tx_energy_j(self.cfg.tx_cost_j_per_byte, sent_bytes)
                    # update AoI receive only for tasks whose updates were delivered
                    for tid, b in delivered.items():
                        if b > 0:
                            if tid < self.cfg.num_tasks_total:
                                self.tasks[tid].last_received_step = self.step_i
                                if not self.tasks[tid].delivered:
                                    tasks_delivered_now += 1
                                self.tasks[tid].delivered = True
                            else:
                                uid = tid - self.cfg.num_tasks_total
                                if 0 <= uid < self.cfg.num_users_total:
                                    self.user_last_received[uid] = self.step_i
                                    self.user_delivered[uid] = True
                continue

            # send to neighbor k
            k = comm - 2
            if 0 <= k < self.n and k != i:
                if in_range(self.agent_pos[i], self.agent_pos[k], self.cfg.comm_range_m):
                    # limit by neighbor buffer space
                    space = int(self.cfg.buffer_bytes_max) - self._buffer_total(k)
                    send_bytes = min(send_bytes, self._buffer_take_limit(i, space))
                    if send_bytes <= 0:
                        continue
                    delivered = self._buffer_take(i, send_bytes)
                    self._buffer_add_many(k, delivered)
                    sent_bytes = int(sum(delivered.values()))
                    bytes_relayed_now += sent_bytes
                    energy_used_j += tx_energy_j(self.cfg.tx_cost_j_per_byte, sent_bytes)
                    self.agent_energy[i] -= tx_energy_j(self.cfg.tx_cost_j_per_byte, sent_bytes)

        # terminations
        self.step_i += 1
        done_time = self.step_i >= self.cfg.horizon_steps
        # if all energy dead, truncate
        dead = np.all(self.agent_energy <= 0.0)

        terminations = {a: bool(done_time) for a in self.agents}
        truncations = {a: bool(dead and not done_time) for a in self.agents}

        # reward (team -> broadcast)
        mean_aoi = self._mean_aoi()
        # shaping: progress to nearest task and to base
        shaping = 0.0
        idle_agents = 0
        idle_move = 0
        bad_mode = 0
        base_ready = 0
        buffer_bytes_total = 0
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            # penalize staying still when tasks are active
            if int(act_arr[i, 1]) == 0 and any(t.active and (not t.completed) for t in self.tasks):
                idle_move += 1
            # idle penalty if tasks exist but agent has no assigned task
            if int(self.agent_task[i]) < 0:
                has_active = any(t.active and (not t.completed) for t in self.tasks)
                if has_active:
                    idle_agents += 1
            # nearest-task progress shaping
            if not np.isnan(pre_nearest_task_dist[i]):
                # recompute nearest post-move distance
                best_d = 1e18
                for t in self.tasks:
                    if not t.active or t.completed:
                        continue
                    d = float(np.linalg.norm(self.agent_pos[i] - t.pos_xy))
                    if d < best_d:
                        best_d = d
                post = float(best_d) if best_d < 1e18 else float(pre_nearest_task_dist[i])
                prog = float(pre_nearest_task_dist[i] - post)
                prog = float(np.clip(prog, -1.0, 1.0))
                shaping += self.reward_w.k_task_prog * prog
            # base progress shaping when buffer non-empty
            if self._buffer_total(i) > 0:
                buffer_bytes_total += self._buffer_total(i)
                post_base = float(np.linalg.norm(self.agent_pos[i] - self.base_xy))
                prog_b = float(self.prev_base_dist[i] - post_base)
                prog_b = float(np.clip(prog_b, -1.0, 1.0))
                shaping += self.reward_w.k_base_prog * prog_b
                if in_range(self.agent_pos[i], self.base_xy, self.cfg.base_range_m):
                    base_ready += 1
            # mode is auto-derived; no action-based bad-mode penalty
        shaping = float(np.clip(shaping, -self.reward_w.shaping_clip, self.reward_w.shaping_clip))
        r_idle = -0.1 * float(idle_agents) - 0.05 * float(idle_move)
        r_mode = -0.5 * float(bad_mode)
        r_base = 1.0 * float(base_ready)
        r_buffer = -0.0005 * float(buffer_bytes_total)

        r_task = self.reward_w.r_task * float(tasks_delivered_now)
        r_bytes = self.reward_w.r_bytes * float(bytes_delivered_now)
        r_gen = self.reward_w.r_gen * float(bytes_generated_now)
        r_sense = self.reward_w.r_sense * float(dwell_progress_now)
        r_aoi = -self.reward_w.w_aoi * float(mean_aoi)
        r_energy = -self.reward_w.w_energy * float(energy_used_j)
        r_team = team_reward(
            tasks_delivered_now,
            bytes_delivered_now,
            mean_aoi,
            energy_used_j,
            shaping,
            bytes_generated_now,
            self.reward_w,
        )
        r_team += r_sense + r_idle + r_gen + r_mode + r_base + r_buffer
        rewards = {a: float(r_team) for a in self.agents}

        obs = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        infos = {}
        for i, a in enumerate(self.agents):
            base_in = in_range(self.agent_pos[i], self.base_xy, self.cfg.base_range_m)
            buf_nonempty = self._buffer_total(i) > 0
            comm = int(act_arr[i, 2])
            infos[a] = {
                "tasks_completed_now": tasks_completed_now,
                "tasks_delivered_now": tasks_delivered_now,
                "bytes_delivered_now": bytes_delivered_now,
                "bytes_delivered_now_agent": int(per_agent_delivered[i]),
                "bytes_generated_now": int(bytes_generated_now),
                "bytes_dropped_now": int(bytes_dropped_now),
                "bytes_relayed_now": int(bytes_relayed_now),
                "mean_aoi": mean_aoi,
                "mean_task_dist_m": float(np.nanmean(pre_task_dist)) if np.any(~np.isnan(pre_task_dist)) else 0.0,
                "mean_base_dist_m": float(np.mean(self.prev_base_dist)),
                "energy_used_j": float(energy_used_j),
                "r_task": float(r_task),
                "r_bytes": float(r_bytes),
                "r_gen": float(r_gen),
                "r_aoi": float(r_aoi),
                "r_energy": float(r_energy),
                "r_shaping": float(shaping),
                "r_idle": float(r_idle),
                "r_mode": float(r_mode),
                "r_base": float(r_base),
                "r_buffer": float(r_buffer),
                "r_sense": float(r_sense),
                "base_in_range": bool(base_in),
                "buffer_nonempty": bool(buf_nonempty),
                "comm_to_base": bool(comm == 1 and base_in),
                "comm_send_to_base_attempt": bool(comm == 1),
            }

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
        U = 3
        w, h = cfg.map_xy_m
        pos = self.agent_pos[i].astype(np.float32)

        base_in = 1.0 if in_range(pos, self.base_xy, cfg.base_range_m) else 0.0
        energy = float(self.agent_energy[i]) / float(cfg.e0_j)
        buf = float(self._buffer_total(i)) / float(cfg.buffer_bytes_max)
        # clip to keep observation within Box bounds
        energy = float(np.clip(energy, 0.0, 1.0))
        buf = float(np.clip(buf, 0.0, 1.0))

        # current task id normalized to [-1, 1] (or -1 if none)
        if int(self.agent_task[i]) < 0:
            cur_task = -1.0
        else:
            denom = max(int(cfg.num_tasks_total) - 1, 1)
            cur_task = (float(self.agent_task[i]) / float(denom)) * 2.0 - 1.0

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

        # nearest active users
        user_rels = []
        user_act = []
        active_users = [u for u in range(self.cfg.num_users_total) if self.user_active[u]]
        if active_users:
            dists_u = np.array([np.linalg.norm(pos - self.user_pos[u]) for u in active_users], dtype=np.float32)
            idx_u = np.argsort(dists_u)[:U]
            chosen_u = [active_users[j] for j in idx_u]
        else:
            chosen_u = []
        for uid in chosen_u:
            rel = (self.user_pos[uid] - pos) / np.array([w, h], dtype=np.float32)
            user_rels.extend([float(rel[0]), float(rel[1])])
            user_act.append(1.0)
        while len(user_rels) < U * 2:
            user_rels.extend([0.0, 0.0])
        while len(user_act) < U:
            user_act.append(0.0)

        # neighbors within comm range (top-N by distance)
        N = 3
        neigh_rels = []
        neigh_linkq = []
        neighs = []
        for j in range(self.n):
            if j == i:
                continue
            d = float(np.linalg.norm(self.agent_pos[j] - pos))
            if d <= float(cfg.comm_range_m):
                neighs.append((d, j))
        neighs = sorted(neighs, key=lambda x: x[0])[:N]
        for d, j in neighs:
            rel = (self.agent_pos[j] - pos) / np.array([w, h], dtype=np.float32)
            neigh_rels.extend([float(rel[0]), float(rel[1])])
            linkq = 1.0 - (d / float(cfg.comm_range_m)) if cfg.comm_range_m > 0 else 0.0
            linkq = float(np.clip(linkq, 0.0, 1.0))
            neigh_linkq.append(linkq * 2.0 - 1.0)

        # pad neighbors
        while len(neigh_rels) < N * 2:
            neigh_rels.extend([0.0, 0.0])
        while len(neigh_linkq) < N:
            neigh_linkq.append(0.0)

        obs = np.array(
            [
                (pos[0] / w) * 2 - 1,
                (pos[1] / h) * 2 - 1,
                energy * 2 - 1,
                buf * 2 - 1,
                cur_task,
                base_in * 2 - 1,
                *rels,
                *status,
                *act,
                *user_rels,
                *user_act,
                *neigh_rels,
                *neigh_linkq,
            ],
            dtype=np.float32,
        )
        return obs

    def _mean_aoi(self) -> float:
        # AoI over active, undelivered streams only (one-shot freshness).
        aois = []
        if self.cfg.num_users_total <= 0:
            for t in self.tasks:
                if not t.active:
                    continue
                if t.delivered:
                    continue
                if t.last_received_step is None:
                    aois.append(float(self.step_i - t.appears_step))
                else:
                    aois.append(float(self.step_i - t.last_received_step))
        # AoI over active users (primary when users exist)
        for uid in range(self.cfg.num_users_total):
            if not self.user_active[uid]:
                continue
            if self.user_delivered[uid]:
                continue
            if self.user_last_received[uid] is None:
                aois.append(float(self.step_i - self.user_appears[uid]))
            else:
                aois.append(float(self.step_i - self.user_last_received[uid]))
        return float(np.mean(aois)) if aois else 0.0

    def _buffer_total(self, i: int) -> int:
        return int(sum(self.agent_buffer_bytes[i].values()))

    def _buffer_add(self, i: int, tid: int, bytes_add: int) -> tuple[int, int]:
        if bytes_add <= 0:
            return 0, 0
        cur = self._buffer_total(i)
        space = int(self.cfg.buffer_bytes_max) - cur
        if space <= 0:
            return 0, int(bytes_add)
        add = min(int(bytes_add), space)
        if add <= 0:
            return 0, int(bytes_add)
        self.agent_buffer_bytes[i][tid] = int(self.agent_buffer_bytes[i].get(tid, 0) + add)
        dropped = int(bytes_add - add)
        return int(add), dropped

    def _buffer_add_many(self, i: int, delivered: dict[int, int]) -> None:
        if not delivered:
            return
        for tid, b in delivered.items():
            self._buffer_add(i, tid, b)

    def _mean_distances(self) -> tuple[float, float]:
        # retained for backward compatibility (not used in reward)
        task_dists = []
        base_dists = []
        for i in range(self.n):
            if self.possible_agents[i] not in self.agents:
                continue
            tid = int(self.agent_task[i])
            if tid >= 0 and tid < len(self.tasks):
                t = self.tasks[tid]
                if t.active and (not t.completed):
                    task_dists.append(float(np.linalg.norm(self.agent_pos[i] - t.pos_xy)))
            base_dists.append(float(np.linalg.norm(self.agent_pos[i] - self.base_xy)))
        mean_task = float(np.mean(task_dists)) if task_dists else 0.0
        mean_base = float(np.mean(base_dists)) if base_dists else 0.0
        return mean_task, mean_base

    def _buffer_take_limit(self, i: int, capacity: int) -> int:
        if capacity <= 0:
            return 0
        total_bytes = self._buffer_total(i)
        if total_bytes <= 0:
            return 0
        return int(min(capacity, total_bytes))

    def _buffer_take(self, i: int, take_bytes: int) -> dict[int, int]:
        """
        Remove up to take_bytes from buffer across streams.
        Returns {task_id: bytes_taken}.
        """
        if take_bytes <= 0:
            return {}
        delivered: dict[int, int] = {}
        # deterministic order
        for tid in sorted(self.agent_buffer_bytes[i].keys()):
            if take_bytes <= 0:
                break
            avail_bytes = int(self.agent_buffer_bytes[i].get(tid, 0))
            if avail_bytes <= 0:
                continue
            use_bytes = int(min(avail_bytes, take_bytes))
            self.agent_buffer_bytes[i][tid] = int(avail_bytes - use_bytes)
            if self.agent_buffer_bytes[i][tid] <= 0:
                del self.agent_buffer_bytes[i][tid]
            delivered[tid] = int(use_bytes)
            take_bytes -= use_bytes
        return delivered
