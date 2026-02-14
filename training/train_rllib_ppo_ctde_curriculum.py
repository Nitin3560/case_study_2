from __future__ import annotations

import argparse
import os

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.utils.config_loader import build_env_cfg, load_yaml
from case_study_2.utils.ctde import CTDEWrapper, CentralizedCriticModel


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_env(env_cfg_path: str, seed: int):
    env_yaml = load_yaml(env_cfg_path)
    cfg = build_env_cfg(env_yaml, seed=seed)
    base = LawnTaskCommParallelEnv(cfg)
    return CTDEWrapper(base)


def _build_algo(env_cfg_path: str, seed: int, num_workers: int) -> Algorithm:
    tmp_env = _make_env(env_cfg_path, seed)
    obs_dim = tmp_env.obs_dim
    state_dim = tmp_env.state_dim

    env_name = "ctde_env_curriculum"

    def env_creator(env_config):
        path = env_config.get("env_config_path")
        s = int(env_config.get("seed", 1))
        return ParallelPettingZooEnv(_make_env(path, s))

    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)
    register_env(env_name, env_creator)

    config = (
        PPOConfig()
        .environment(env_name, env_config={"env_config_path": env_cfg_path, "seed": seed})
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(num_env_runners=num_workers)
        .training(
            model={
                "custom_model": "centralized_critic",
                "fcnet_hiddens": [128, 128],
                "custom_model_config": {"obs_dim": obs_dim, "state_dim": state_dim},
            },
            train_batch_size=4000,
            minibatch_size=512,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
        )
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    return config.build()


def _train_stage(algo: Algorithm, iters: int, stage_name: str) -> None:
    for i in range(1, iters + 1):
        r = algo.train()
        if i % 5 == 0 or i == iters:
            ret = float(r.get("env_runners", {}).get("episode_return_mean", 0.0))
            print(f"[{stage_name}] iter={i}/{iters} episode_return_mean={ret:.3f}")


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-config", default=os.path.join(root, "configs", "env_case2_stage1.yaml"))
    ap.add_argument("--stage2-config", default=os.path.join(root, "configs", "env_case2_publication.yaml"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters-stage1", type=int, default=40)
    ap.add_argument("--iters-stage2", type=int, default=80)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(root, "outputs", "checkpoints", "rllib_ppo_ctde_curriculum"),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building stage-1 algorithm...")
    algo1 = _build_algo(args.stage1_config, args.seed, args.num_workers)
    _train_stage(algo1, args.iters_stage1, "stage1")
    ckpt1 = algo1.save(args.out_dir)
    print("Stage-1 checkpoint:", ckpt1)
    weights = algo1.get_weights()
    algo1.stop()

    print("Building stage-2 algorithm...")
    algo2 = _build_algo(args.stage2_config, args.seed, args.num_workers)
    algo2.set_weights(weights)
    _train_stage(algo2, args.iters_stage2, "stage2")
    ckpt2 = algo2.save(args.out_dir)
    print("Final checkpoint:", ckpt2)
    algo2.stop()


if __name__ == "__main__":
    main()
