from __future__ import annotations

import argparse
import os

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

from case_study_2.utils.config_loader import load_yaml, build_env_cfg
from case_study_2.envs.lawn_task_comm_env import LawnTaskCommParallelEnv
from case_study_2.utils.ctde import CTDEWrapper, CentralizedCriticModel


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_env(env_cfg_path: str, seed: int):
    env_yaml = load_yaml(env_cfg_path)
    cfg = build_env_cfg(env_yaml, seed=seed)
    base = LawnTaskCommParallelEnv(cfg)
    return CTDEWrapper(base)


def env_creator(env_config):
    path = env_config.get("env_config_path")
    seed = int(env_config.get("seed", 1))
    return ParallelPettingZooEnv(make_env(path, seed))


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default=os.path.join(root, "configs", "env.yaml"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stop-iters", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default=os.path.join(root, "outputs", "checkpoints", "rllib_ppo_ctde"))
    args = ap.parse_args()

    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

    register_env("ctde_env", env_creator)

    tmp_env = make_env(args.env_config, args.seed)
    obs_dim = tmp_env.obs_dim
    state_dim = tmp_env.state_dim

    config = (
        PPOConfig()
        .environment("ctde_env", env_config={"env_config_path": args.env_config, "seed": args.seed})
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(num_env_runners=args.num_workers)
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
            vf_clip_param=100000.0,
        )
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )

    stop = {"training_iteration": args.stop_iters}
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, storage_path=args.checkpoint_dir),
    )
    tuner.fit()
    print("Training complete. Checkpoints in:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
