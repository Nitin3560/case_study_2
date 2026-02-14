from __future__ import annotations
import numpy as np

from case_study_2.controllers.motion_controller import move_towards
from case_study_2.controllers.comm_controller import hold, send_to_base


class StaticBaseline:
    """
    Fixed non-intelligent baseline:
    - Each drone sticks to a predetermined location.
    - Equal split between scan mode and communication mode.
    - No adaptive task reassignment, no adaptive relay routing.
    """

    def __init__(self, num_agents: int, num_tasks: int):
        self.n = num_agents
        self.m = num_tasks
        self.step = 0
        self.home = None

    def act(self, env) -> dict:
        if self.home is None:
            w, h = env.cfg.map_xy_m
            grid_n = int(np.ceil(np.sqrt(max(self.n, 1))))
            spacing_x = float(w) / float(grid_n + 1)
            spacing_y = float(h) / float(grid_n + 1)
            grid_points = []
            for gy in range(grid_n):
                for gx in range(grid_n):
                    grid_points.append(
                        np.array([(gx + 1) * spacing_x, (gy + 1) * spacing_y], dtype=np.float32)
                    )
            self.home = [grid_points[i % len(grid_points)] for i in range(self.n)]

        actions = {}
        for i, name in enumerate(env.possible_agents):
            # no task allocation in static baseline
            task_choice = 0

            # stick to predetermined home location
            move_dir = move_towards(env.agent_pos[i], self.home[i])

            # equal split between scan and comm phases
            if self.step % 2 == 0:
                mode = 0  # scan
                comm = hold()
            else:
                mode = 1  # comm
                base_in = np.linalg.norm(env.agent_pos[i] - env.base_xy) <= env.cfg.base_range_m
                comm = send_to_base() if base_in else hold()

            actions[name] = np.array([task_choice, move_dir, comm, mode], dtype=np.int32)
        self.step += 1
        return actions
