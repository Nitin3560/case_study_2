from __future__ import annotations
import numpy as np

from case_study_2.controllers.motion_controller import move_towards
from case_study_2.controllers.comm_controller import hold, send_to_base, send_to_neighbor


class FixedRelayTreeBaseline:
    """
    Static task allocation + fixed relay routing.
    - Tasks are pre-assigned round-robin.
    - A fixed subset of agents act as relays to the base.
    - Non-relays only forward to relays when in range.
    """

    def __init__(self, num_agents: int, num_tasks: int, relay_ids: list[int] | None = None):
        self.n = num_agents
        self.m = num_tasks
        if relay_ids is None:
            k = min(2, max(1, self.n // 3))
            relay_ids = list(range(k))
        self.relays = set(int(r) for r in relay_ids if 0 <= int(r) < self.n)

        self.assign = {i: [] for i in range(self.n)}
        for tid in range(self.m):
            self.assign[tid % self.n].append(tid)
        self.ptr = {i: 0 for i in range(self.n)}

    def _pick_task(self, env, i: int) -> int | None:
        lst = self.assign[i]
        if not lst:
            return None
        for _ in range(len(lst)):
            tid = lst[self.ptr[i] % len(lst)]
            self.ptr[i] = (self.ptr[i] + 1) % len(lst)
            t = env.tasks[tid]
            if t.active and (not t.completed):
                return tid
        return None

    def _comm_action(self, env, i: int) -> int:
        # Relays should loiter near base and forward to base when in range
        if i in self.relays:
            if float(np.linalg.norm(env.agent_pos[i] - env.base_xy)) <= float(env.cfg.base_range_m):
                return send_to_base()
            return hold()

        # Non-relays send to closest relay in range, else hold
        best = None
        best_d = 1e18
        for r in self.relays:
            if r == i:
                continue
            d = float(np.linalg.norm(env.agent_pos[i] - env.agent_pos[r]))
            if d <= float(env.cfg.comm_range_m) and d < best_d:
                best_d = d
                best = r
        if best is None:
            return hold()
        return send_to_neighbor(best)

    def _relay_move(self, env, i: int) -> int:
        # keep relay near base
        return move_towards(env.agent_pos[i], env.base_xy)

    def act(self, env) -> dict:
        actions = {}
        for i, name in enumerate(env.possible_agents):
            if i in self.relays:
                # relay stays near base, no tasking
                task_choice = 0
                move_dir = self._relay_move(env, i)
                comm = self._comm_action(env, i)
                mode = 1
                actions[name] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
                continue

            # task selection
            task = self._pick_task(env, i)
            task_choice = 0 if task is None else (task + 1)

            # movement
            if env._buffer_total(i) > 0:
                move_dir = move_towards(env.agent_pos[i], env.base_xy)
            else:
                user = self._nearest_active_user(env, i)
                if user is not None:
                    move_dir = move_towards(env.agent_pos[i], env.user_pos[user])
                elif task is not None:
                    goal = env.tasks[task].pos_xy
                    move_dir = move_towards(env.agent_pos[i], goal)
                else:
                    move_dir = 0

            # comm routing
            comm = self._comm_action(env, i)
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
