#!/usr/bin/env python3

"""
Unified training entry using Hydra/OmegaConf for Isaac Lab + SKRL PPO.

- Select environment config class (e.g., Navrl/OpenAir/Dyanmic/City) via conf/env/*.yaml
- Select model (feature extractor) via conf/model/*.yaml
- Select PPO hyperparameters via conf/algo/ppo.yaml
- Training/runtime settings in conf/train/default.yaml

This script mirrors the functionality of learning/skrl/train_navrl.py but generalizes
to multiple environment/model combinations through configuration.
"""

import os
import argparse
from datetime import datetime
from types import SimpleNamespace
import importlib

import torch
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


def _import_from_string(path: str):
    module_path, name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def _apply_nested_overrides(env_cfg, cfg_env):
    # Common overrides supported by our env configs
    if hasattr(cfg_env, "seed"):
        env_cfg.seed = cfg_env.seed
    if hasattr(cfg_env, "scene") and cfg_env.scene is not None:
        if hasattr(env_cfg, "scene") and cfg_env.scene.get("num_envs") is not None:
            env_cfg.scene.num_envs = cfg_env.scene.num_envs
    # action manager
    if hasattr(cfg_env, "action_manager") and cfg_env.action_manager is not None:
        am = cfg_env.action_manager
        if getattr(am, "action_space_type", None) is not None:
            env_cfg.action_manager.action_space_type = am.action_space_type
        if getattr(am, "action_mode", None) is not None:
            env_cfg.action_manager.action_mode = am.action_mode
    # observation & reward modules
    if hasattr(cfg_env, "observation_cfg") and cfg_env.observation_cfg is not None:
        if getattr(cfg_env.observation_cfg, "modules", None) is not None:
            env_cfg.observation_cfg.modules = list(cfg_env.observation_cfg.modules)
    if hasattr(cfg_env, "reward_cfg") and cfg_env.reward_cfg is not None:
        if getattr(cfg_env.reward_cfg, "modules", None) is not None:
            env_cfg.reward_cfg.modules = list(cfg_env.reward_cfg.modules)
    # traffic overrides
    if hasattr(cfg_env, "traffic_sim") and cfg_env.traffic_sim is not None:
        if getattr(cfg_env.traffic_sim, "num_drones", None) is not None:
            env_cfg.traffic_sim.num_drones = cfg_env.traffic_sim.num_drones
        if getattr(cfg_env.traffic_sim, "num_evtols", None) is not None:
            env_cfg.traffic_sim.num_evtols = cfg_env.traffic_sim.num_evtols
    # simple scalars
    for k in ["arrival_threshold", "predict_steps", "rew_evtol_future_penalty", "rew_drone_future_penalty"]:
        if getattr(cfg_env, k, None) is not None:
            setattr(env_cfg, k, getattr(cfg_env, k))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # Resolve config & build experiment name/dirs
    resolved = OmegaConf.to_container(cfg, resolve=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    experiment_name = cfg.project.experiment_name if cfg.project.experiment_name else f"hydra_{timestamp}"
    save_dir = os.path.join(cfg.project.save_dir_base, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Isaac Sim AppLauncher setup using default args, then override
    from omni.isaac.lab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    default_args = parser.parse_args([])  # take Isaac default args
    default_args.headless = cfg.train.app.headless
    default_args.enable_cameras = cfg.train.app.enable_cameras or bool(cfg.train.video.enabled)
    app_launcher = AppLauncher(default_args)
    simulation_app = app_launcher.app

    # Import after AppLauncher per Isaac Lab requirement
    from omni.isaac.lab.envs.common import ViewerCfg
    from omni.isaac.lab.utils.io import dump_yaml
    from isaac_lab_envs.direct.uam_env import UamEnv
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    # Build environment config class and apply overrides
    env_cfg_cls = _import_from_string(cfg.env.cfg_class)
    env_cfg = env_cfg_cls()
    _apply_nested_overrides(env_cfg, cfg.env)

    # Optional viewer override similar to train_navrl behavior
    if getattr(cfg.train.viewer, "override", False):
        bounds = env_cfg.area_bounds
        range_x = bounds.xmax - bounds.xmin
        env_cfg.viewer = ViewerCfg(
            resolution=tuple(cfg.train.viewer.resolution),
            eye=(0.0, 0.0, range_x * float(cfg.train.viewer.eye_scale_factor)),
            lookat=(0.0, 0.0, 1.0),
        )

    # Save env config
    dump_yaml(os.path.join(save_dir, "env_config.yaml"), env_cfg)

    # Create environment
    record_video = bool(cfg.train.video.enabled)
    render_mode = "rgb_array" if record_video else None
    base_env = UamEnv(cfg=env_cfg, render_mode=render_mode)

    if record_video:
        import gymnasium as gym
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": (lambda step: step % int(cfg.train.video.interval) == 0),
            "video_length": int(cfg.train.video.length),
            "disable_logger": True,
        }
        base_env = gym.wrappers.RecordVideo(base_env, **video_kwargs)
        env_cfg.debug_vis = True

    env = SkrlVecEnvWrapper(base_env)
    env.seed(int(cfg.train.seed))

    # Build SKRL PPO agent with shared model
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    # Model classes
    model_class = _import_from_string(cfg.model.model_class)
    features_extractor_cls = _import_from_string(cfg.model.features_extractor_class)

    shared_model = model_class(
        env.observation_space,
        env.action_space,
        device=torch.device(cfg.train.device),
        features_dim=int(cfg.model.features_dim),
        net_arch=list(cfg.model.net_arch),
        log_std_init=float(cfg.model.log_std_init),
        features_extractor_cls=features_extractor_cls,
    )

    models = {"policy": shared_model, "value": shared_model}

    # PPO config
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["learning_epochs"] = int(cfg.algo.n_epochs)
    ppo_cfg["mini_batches"] = int(cfg.algo.mini_batches)
    ppo_cfg["discount_factor"] = float(cfg.algo.gamma)
    ppo_cfg["lambda"] = 0.95
    ppo_cfg["learning_rate"] = float(cfg.algo.learning_rate)
    ppo_cfg["learning_rate_scheduler"] = None
    ppo_cfg["learning_rate_scheduler_kwargs"] = {}
    ppo_cfg["ratio_clip"] = float(cfg.algo.clip_range)
    ppo_cfg["value_clip"] = float(cfg.algo.clip_range)
    ppo_cfg["clip_predicted_values"] = True
    ppo_cfg["entropy_loss_scale"] = float(cfg.algo.ent_coef)
    ppo_cfg["value_loss_scale"] = float(cfg.algo.value_loss_scale)
    ppo_cfg["kl_threshold"] = float(cfg.algo.kl_threshold)

    # Normalization
    if bool(cfg.algo.normalization.norm_obs):
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": torch.device(cfg.train.device)}
    if bool(cfg.algo.normalization.norm_reward):
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": torch.device(cfg.train.device)}

    # Experiment settings for SKRL
    ppo_cfg["experiment"] = {
        "directory": cfg.project.save_dir_base,
        "experiment_name": experiment_name,
        "write_interval": int(cfg.algo.experiment.write_interval),
        "checkpoint_interval": int(cfg.algo.experiment.checkpoint_interval),
    }

    # Memory
    memory = RandomMemory(memory_size=int(cfg.algo.n_steps), num_envs=env.num_envs, device=torch.device(cfg.train.device))

    # Agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=torch.device(cfg.train.device),
    )

    # Save Hydra config snapshot
    with open(os.path.join(save_dir, "hydra_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Trainer
    timesteps = int(cfg.train.total_timesteps) // env.num_envs
    trainer_cfg = {"timesteps": timesteps, "headless": True}
    trainer = _import_from_string("skrl.trainers.torch.SequentialTrainer")(cfg=trainer_cfg, env=env, agents=agent)

    print(f"experiment configuration:")
    print(f"- experiment name: {experiment_name}")
    print(f"- save directory: {save_dir}")
    print(f"- num_envs: {env.num_envs}")
    print(f"- device: {cfg.train.device}")

    # Train
    trainer.train()

    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pt"))


if __name__ == "__main__":
    main()


