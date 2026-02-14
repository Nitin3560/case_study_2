from __future__ import annotations
import numpy as np

from case_study_2.controllers.motion_controller import move_towards
from case_study_2.controllers.comm_controller import hold, send_to_base, send_to_neighbor


class GreedyBaseline:
    """
    Greedy nearest-feasible task + energy-aware opportunistic relaying.
    - Picks nearest active uncompleted task (energy-aware).
    - Comm: if base in range -> send to base.
            else -> send to neighbor that is closer to base and has energy headroom.
            else hold.
    """

    def __init__(self, relay_neighbors_max: int = 3):
        self.relay_neighbors_max = relay_neighbors_max

    def act(self, env) -> dict:
        actions = {}
        for i, name in enumerate(env.possible_agents):
            # prioritize active undelivered users for coverage
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
            # mode: relay if buffer non-empty, else scan
            mode = 1 if env._buffer_total(i) > 0 else 0
            actions[name] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
        return actions

    def _nearest_active_user(self, env, i: int) -> int | None:
        best = None
        best_d = 1e18
        for uid in range(env.cfg.num_users_total):
            if not env.user_active[uid] or env.user_delivered[uid]:
                continue
            d = float(np.linalg.norm(env.agent_pos[i] - env.user_pos[uid]))
            if d < best_d:
                best_d = d
                best = uid
        return best

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
                # prefer neighbors with more energy
                if env.agent_energy[k] > env.cfg.low_energy_j:
                    neighbors.append(k)

        if not neighbors:
            return hold()

        # choose neighbor that is closest to base
        neighbors = sorted(neighbors, key=lambda k: float(np.linalg.norm(env.agent_pos[k] - env.base_xy)))
        k = neighbors[0]
        return send_to_neighbor(k)
