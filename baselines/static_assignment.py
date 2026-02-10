from __future__ import annotations
import numpy as np

from case_study_2.controllers.motion_controller import move_towards
from case_study_2.controllers.comm_controller import hold, send_to_base


class StaticBaseline:
    """
    Static task allocation (round-robin) + direct-to-base if possible else hold.
    This is intentionally weak but reproducible.
    """

    def __init__(self, num_agents: int, num_tasks: int):
        self.n = num_agents
        self.m = num_tasks
        self.assign = {i: [] for i in range(self.n)}
        for tid in range(self.m):
            self.assign[tid % self.n].append(tid)
        self.ptr = {i: 0 for i in range(self.n)}

    def act(self, env) -> dict:
        actions = {}
        # env internals used only for baseline simplicity
        for i, name in enumerate(env.possible_agents):
            # choose current assigned task if active & not done
            task = self._pick_task(env, i)
            task_choice = 0 if task is None else (task + 1)

            # move towards chosen task if exists
            if task is None:
                move_dir = 0
            else:
                goal = env.tasks[task].pos_xy
                move_dir = move_towards(env.agent_pos[i], goal)

            # comm: send to base if in range else hold
            base_in = np.linalg.norm(env.agent_pos[i] - env.base_xy) <= env.cfg.base_range_m
            comm = send_to_base() if base_in else hold()

            actions[name] = np.array([task_choice, move_dir, comm], dtype=np.int32)
        return actions

    def _pick_task(self, env, i: int) -> int | None:
        lst = self.assign[i]
        if not lst:
            return None
        # advance ptr until we find an active, not completed task
        for _ in range(len(lst)):
            tid = lst[self.ptr[i] % len(lst)]
            self.ptr[i] = (self.ptr[i] + 1) % len(lst)
            t = env.tasks[tid]
            if t.active and (not t.completed):
                return tid
        return None