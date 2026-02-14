from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
import torch.nn as nn

from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.controllers.motion_controller import move_towards


def global_state(env: LawnTaskCommParallelEnv) -> np.ndarray:
    cfg = env.cfg
    w, h = cfg.map_xy_m
    parts = []
    for i in range(env.n):
        pos = env.agent_pos[i] / np.array([w, h], dtype=np.float32)
        energy = np.array([env.agent_energy[i] / cfg.e0_j], dtype=np.float32)
        buf = np.array([env._buffer_total(i) / cfg.buffer_bytes_max], dtype=np.float32)
        tid = env.agent_task[i]
        if tid < 0:
            tid_norm = -1.0
        else:
            denom = max(cfg.num_tasks_total - 1, 1)
            tid_norm = (float(tid) / float(denom)) * 2.0 - 1.0
        parts.append(np.array([pos[0], pos[1], energy[0], buf[0], tid_norm], dtype=np.float32))

    # Keep fixed dimensionality independent of reset order.
    for tid in range(cfg.num_tasks_total):
        if tid < len(env.tasks):
            t = env.tasks[tid]
            p = t.pos_xy / np.array([w, h], dtype=np.float32)
            active = 1.0 if t.active else 0.0
            completed = 1.0 if t.completed else 0.0
        else:
            p = np.array([0.0, 0.0], dtype=np.float32)
            active = 0.0
            completed = 0.0
        parts.append(np.array([p[0], p[1], active, completed], dtype=np.float32))

    for uid in range(env.cfg.num_users_total):
        p = env.user_pos[uid] / np.array([w, h], dtype=np.float32)
        parts.append(np.array([p[0], p[1], 1.0 if env.user_active[uid] else 0.0], dtype=np.float32))

    base = np.array([env.base_xy[0] / w, env.base_xy[1] / h], dtype=np.float32)
    parts.append(base)
    return np.concatenate(parts, axis=0)


class CTDEWrapper:
    """
    Wraps LawnTaskCommParallelEnv to expose dict observations:
    obs: local observation
    state: global state (critic only)
    """
    def __init__(self, env: LawnTaskCommParallelEnv, auto_move: bool = True):
        self.env = env
        self.auto_move = bool(auto_move)
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

        obs_space = env.observation_space(env.possible_agents[0])
        self.obs_dim = int(obs_space.shape[0])
        self.state_dim = int(global_state(env).shape[0])
        self._obs_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(self.obs_dim + self.state_dim,),
            dtype=np.float32,
        )

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        state = global_state(self.env).astype(np.float32)
        obs2 = {a: np.concatenate([obs[a].astype(np.float32), state], axis=0) for a in obs}
        return obs2, infos

    def step(self, actions):
        if self.auto_move and actions:
            actions = self._adapt_actions(actions)
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents
        state = global_state(self.env).astype(np.float32)
        obs2 = {a: np.concatenate([obs[a].astype(np.float32), state], axis=0) for a in obs}
        return obs2, rewards, terminations, truncations, infos

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()

    def _adapt_actions(self, actions: dict) -> dict:
        # Keep learned task/comm, but derive movement heuristically.
        out = {}
        for i, a in enumerate(self.env.possible_agents):
            if a not in actions:
                continue
            act = np.array(actions[a], dtype=np.int32).copy()
            task_choice = int(act[0])

            if self.env._buffer_total(i) > 0:
                move = move_towards(self.env.agent_pos[i], self.env.base_xy)
            else:
                target = None
                tid = task_choice - 1
                if 0 <= tid < self.env.cfg.num_tasks_total:
                    t = self.env.tasks[tid]
                    if t.active and (not t.completed):
                        target = t.pos_xy
                if target is None:
                    best = None
                    best_d = 1e18
                    for t in self.env.tasks:
                        if not t.active or t.completed:
                            continue
                        d = float(np.linalg.norm(self.env.agent_pos[i] - t.pos_xy))
                        if d < best_d:
                            best_d = d
                            best = t
                    if best is not None:
                        target = best.pos_xy
                move = move_towards(self.env.agent_pos[i], target) if target is not None else 0

            act[1] = int(move)
            out[a] = act
        return out


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Policy uses local obs, value uses global state.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = int(model_config.get("custom_model_config", {}).get("obs_dim"))
        state_dim = int(model_config.get("custom_model_config", {}).get("state_dim"))
        assert obs_dim > 0 and state_dim > 0, "custom_model_config must include obs_dim and state_dim"
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        hidden = model_config.get("fcnet_hiddens", [128, 128])

        out_dim = int(np.sum(action_space.nvec))

        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.policy_net = nn.Sequential(*layers)

        v_layers = []
        last = state_dim
        for h in hidden:
            v_layers.append(nn.Linear(last, h))
            v_layers.append(nn.Tanh())
            last = h
        v_layers.append(nn.Linear(last, 1))
        self.value_net = nn.Sequential(*v_layers)

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        full = input_dict["obs"].float()
        obs = full[:, :self.obs_dim]
        state_in = full[:, self.obs_dim:self.obs_dim + self.state_dim]
        logits = self.policy_net(obs)
        self._value = self.value_net(state_in).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value
