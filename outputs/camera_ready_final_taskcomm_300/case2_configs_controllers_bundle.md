# Case Study 2: Configs + Controller Definitions Bundle

This single file packages the YAML configurations and key controller/policy functions used in Case Study 2.

Note on requested controller types:
- `PID`: not implemented in this repository.
- `Open-loop`: implemented as fixed/static baseline.
- `Agentic`: implemented as greedy/fixed-relay heuristics and RLlib CTDE policy wrapper.

## 1) YAML Configs

### 1.1 Main eval/training config (`configs/env_easy.yaml`)
```yaml
env:
  map_xy_m: [200.0, 200.0]
  horizon_steps: 300
  dt_s: 1.0

  num_agents: 6
  num_tasks_total: 10
  num_users_total: 10

  task_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [2, 2, 2]
  user_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [2, 2, 2]

  sensing:
    visit_radius_m: 20.0
    dwell_steps: 3
    bytes_per_update: 700
    update_period_steps: 12
  users:
    cov_radius_m: 25.0
    bytes_per_step: 500
    speed_mps: 0.5

  comm:
    comm_range_m: 70.0
    base_pos_xy: [100.0, 100.0]
    base_range_m: 40.0
    link_capacity_bytes_per_step: 700
    buffer_bytes_max: 2500

  motion:
    v_max_mps: 4.0

  energy:
    e0_j: 4500.0
    move_cost_j_per_m: 3.0
    sense_cost_j_per_step: 5.0
    tx_cost_j_per_byte: 0.004
    low_energy_j: 600.0
```

### 1.2 Curriculum stage-1 config (`configs/env_case2_stage1.yaml`)
```yaml
env:
  map_xy_m: [220.0, 220.0]
  horizon_steps: 300
  dt_s: 1.0

  num_agents: 6
  num_tasks_total: 10
  num_users_total: 0

  task_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [2, 2, 2]
  user_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [0, 0, 0]

  sensing:
    visit_radius_m: 18.0
    dwell_steps: 3
    bytes_per_update: 700
    update_period_steps: 12
  users:
    cov_radius_m: 25.0
    bytes_per_step: 500
    speed_mps: 0.5

  comm:
    comm_range_m: 70.0
    base_pos_xy: [110.0, 110.0]
    base_range_m: 34.0
    link_capacity_bytes_per_step: 700
    buffer_bytes_max: 2000

  motion:
    v_max_mps: 4.0

  energy:
    e0_j: 4200.0
    move_cost_j_per_m: 3.0
    sense_cost_j_per_step: 5.0
    tx_cost_j_per_byte: 0.004
    low_energy_j: 600.0
```

### 1.3 Publication stress config (`configs/env_case2_publication.yaml`)
```yaml
env:
  map_xy_m: [300.0, 300.0]
  horizon_steps: 300
  dt_s: 1.0

  num_agents: 6
  num_tasks_total: 10
  num_users_total: 0

  task_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [2, 2, 2]
  user_spawn_schedule:
    spawn_fracs: [0.3, 0.5, 0.7]
    spawn_counts: [0, 0, 0]

  sensing:
    visit_radius_m: 16.0
    dwell_steps: 3
    bytes_per_update: 700
    update_period_steps: 12
  users:
    cov_radius_m: 25.0
    bytes_per_step: 500
    speed_mps: 0.5

  comm:
    comm_range_m: 60.0
    base_pos_xy: [150.0, 150.0]
    base_range_m: 22.0
    link_capacity_bytes_per_step: 500
    buffer_bytes_max: 1500

  motion:
    v_max_mps: 4.0

  energy:
    e0_j: 3800.0
    move_cost_j_per_m: 3.0
    sense_cost_j_per_step: 5.0
    tx_cost_j_per_byte: 0.004
    low_energy_j: 600.0
```

## 2) Core low-level controller helpers

### 2.1 Motion quantizer (`controllers/motion_controller.py`)
```python
import numpy as np

def move_towards(cur_xy: np.ndarray, goal_xy: np.ndarray) -> int:
    d = goal_xy - cur_xy
    if np.linalg.norm(d) < 1e-6:
        return 0
    x, y = float(d[0]), float(d[1])
    if abs(x) > abs(y):
        return 1 if x > 0 else 2
    else:
        return 3 if y > 0 else 4
```

### 2.2 Comm action encoding (`controllers/comm_controller.py`)
```python
def hold() -> int:
    return 0

def send_to_base() -> int:
    return 1

def send_to_neighbor(k: int) -> int:
    return 2 + int(k)
```

## 3) Open-loop / fixed baseline controller (static)

### 3.1 Fixed non-intelligent baseline (`baselines/static_assignment.py`)
```python
class StaticBaseline:
    def __init__(self, num_agents: int, num_tasks: int):
        self.n = num_agents
        self.m = num_tasks
        self.step = 0
        self.home = None

    def act(self, env) -> dict:
        if self.home is None:
            # build fixed home grid once
            ...

        actions = {}
        for i, name in enumerate(env.possible_agents):
            task_choice = 0
            move_dir = move_towards(env.agent_pos[i], self.home[i])

            if self.step % 2 == 0:
                mode = 0
                comm = hold()
            else:
                mode = 1
                base_in = np.linalg.norm(env.agent_pos[i] - env.base_xy) <= env.cfg.base_range_m
                comm = send_to_base() if base_in else hold()

            actions[name] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
        self.step += 1
        return actions
```

## 4) Strong heuristic controllers

### 4.1 Fixed relay tree baseline (`baselines/fixed_relay_tree.py`)
Key methods:
```python
class FixedRelayTreeBaseline:
    def _comm_action(self, env, i: int) -> int:
        if i in self.relays:
            if np.linalg.norm(env.agent_pos[i] - env.base_xy) <= env.cfg.base_range_m:
                return send_to_base()
            return hold()

        best = None
        best_d = 1e18
        for r in self.relays:
            d = np.linalg.norm(env.agent_pos[i] - env.agent_pos[r])
            if d <= env.cfg.comm_range_m and d < best_d:
                best_d = d
                best = r
        if best is None:
            return hold()
        return send_to_neighbor(best)
```

### 4.2 Greedy adaptive baseline (`baselines/greedy_auction.py`)
Key methods:
```python
class GreedyBaseline:
    def act(self, env) -> dict:
        for i, name in enumerate(env.possible_agents):
            user = self._nearest_active_user(env, i)
            task = self._nearest_feasible_task(env, i)
            task_choice = 0 if task is None else (task + 1)

            if env._buffer_total(i) > 0:
                move_dir = move_towards(env.agent_pos[i], env.base_xy)
            elif user is not None:
                move_dir = move_towards(env.agent_pos[i], env.user_pos[user])
            elif task is not None:
                move_dir = move_towards(env.agent_pos[i], env.tasks[task].pos_xy)
            else:
                move_dir = 0

            comm = self._comm_action(env, i)
            mode = 1 if env._buffer_total(i) > 0 else 0
            actions[name] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
```

## 5) Agentic RL controller wrapper (CTDE)

### 5.1 RLlib wrapper and action adapter (`utils/ctde.py`)
Key function used in final runs:
```python
class CTDEWrapper:
    def __init__(self, env: LawnTaskCommParallelEnv, auto_move: bool = True):
        self.auto_move = bool(auto_move)

    def step(self, actions):
        if self.auto_move and actions:
            actions = self._adapt_actions(actions)
        return self.env.step(actions)

    def _adapt_actions(self, actions: dict) -> dict:
        # learned: task + comm
        # heuristicized: movement
        out = {}
        for i, a in enumerate(self.env.possible_agents):
            if a not in actions:
                continue
            act = np.array(actions[a], dtype=np.int32).copy()
            task_choice = int(act[0])

            if self.env._buffer_total(i) > 0:
                move = move_towards(self.env.agent_pos[i], self.env.base_xy)
            else:
                # toward selected/nearest active task
                ...
                move = move_towards(self.env.agent_pos[i], target) if target is not None else 0

            act[1] = int(move)
            out[a] = act
        return out
```

## 6) Observation-bounds fix (review finding reference)

### 6.1 Energy/buffer clipping in observation (`envs/lawn_task_comm_env.py`)
```python
energy = float(self.agent_energy[i]) / float(cfg.e0_j)
buf = float(self._buffer_total(i)) / float(cfg.buffer_bytes_max)
# clip to keep observation within Box bounds
energy = float(np.clip(energy, 0.0, 1.0))
buf = float(np.clip(buf, 0.0, 1.0))
```
This addresses the review issue where negative energy could push normalized observations outside `[-1, 1]`.

## 7) Training/eval entry points used

- RLlib training: `training/train_rllib_ppo_ctde.py`
- RLlib eval: `evaluation/eval_rllib_ppo_ctde.py`
- Camera-ready pack + significance tests: `evaluation/camera_ready_pack.py`
- End-to-end report runner: `scripts/camera_ready_run.sh`

