from __future__ import annotations
import numpy as np

from case_study_2.controllers.motion_controller import move_towards
from case_study_2.controllers.comm_controller import hold, send_to_base, send_to_neighbor


class GreedyBaseline:
    """
    Greedy nearest-feasible task + opportunistic relaying.
    - Picks nearest active uncompleted task (energy-aware)
    - Comm: if base in range -> send to base
            else -> send to neighbor that is closer to base (if in comm range)
            else hold
    """

    def __init__(self, relay_neighbors_max: int = 3):
        self.relay_neighbors_max = relay_neighbors_max

    def act(self, env) -> dict:
        actions = {}
        for i, name in enumerate(env.possible_agents):
            task = self._nearest_feasible_task(env, i)
            task_choice = 0 if task is None else (task + 1)

            if task is None:
                move_dir = 0
            else:
                move_dir = move_towards(env.agent_pos[i], env.tasks[task].pos_xy)

            comm = self._comm_action(env, i)
            actions[name] = np.array([task_choice, move_dir, comm], dtype=np.int32)
        return actions

    def _nearest_feasible_task(self, env, i: int) -> int | None:
        # energy-aware: avoid selecting tasks if too low energy
        if env.agent_energy[i] <= env.cfg.low_energy_j:
            return None

        best = None
        best_d = 1e18
        for t in env.tasks:
            if not t.active or t.completed:
                continue
            d = float(np.linalg.norm(env.agent_pos[i] - t.pos_xy))
            if d < best_d:
                best_d = d
                best = t.tid
        return best

    def _comm_action(self, env, i: int) -> int:
        # send to base if in range
        if float(np.linalg.norm(env.agent_pos[i] - env.base_xy)) <= float(env.cfg.base_range_m):
            return send_to_base()

        # else try neighbor closer to base within comm range
        neighbors = []
        for k in range(env.n):
            if k == i:
                continue
            if float(np.linalg.norm(env.agent_pos[i] - env.agent_pos[k])) <= float(env.cfg.comm_range_m):
                neighbors.append(k)

        if not neighbors:
            return hold()

        # choose neighbor that is closest to base
        neighbors = sorted(neighbors, key=lambda k: float(np.linalg.norm(env.agent_pos[k] - env.base_xy)))
        k = neighbors[0]
        return send_to_neighbor(k)