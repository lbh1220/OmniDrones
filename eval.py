#!/usr/bin/env python3

"""
Unified evaluation entry using Hydra/OmegaConf for Isaac Lab + SKRL PPO.

Loads a saved PPO agent checkpoint and runs evaluation episodes with optional video.
"""

import os
import argparse
from datetime import datetime
import importlib
from typing import Optional

import torch
import numpy as np
from omegaconf import OmegaConf
import hydra


def _import_from_string(path: str):
    module_path, name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def _apply_nested_overrides(env_cfg, cfg_env):
    if hasattr(cfg_env, "seed"):
        env_cfg.seed = cfg_env.seed
    if hasattr(cfg_env, "scene") and cfg_env.scene is not None:
        if hasattr(env_cfg, "scene") and cfg_env.scene.get("num_envs") is not None:
            env_cfg.scene.num_envs = cfg_env.scene.num_envs
    if hasattr(cfg_env, "action_manager") and cfg_env.action_manager is not None:
        am = cfg_env.action_manager
        if getattr(am, "action_space_type", None) is not None:
            env_cfg.action_manager.action_space_type = am.action_space_type
        if getattr(am, "action_mode", None) is not None:
            env_cfg.action_manager.action_mode = am.action_mode
    if hasattr(cfg_env, "observation_cfg") and cfg_env.observation_cfg is not None:
        if getattr(cfg_env.observation_cfg, "modules", None) is not None:
            env_cfg.observation_cfg.modules = list(cfg_env.observation_cfg.modules)
    if hasattr(cfg_env, "reward_cfg") and cfg_env.reward_cfg is not None:
        if getattr(cfg_env.reward_cfg, "modules", None) is not None:
            env_cfg.reward_cfg.modules = list(cfg_env.reward_cfg.modules)
    if hasattr(cfg_env, "traffic_sim") and cfg_env.traffic_sim is not None:
        if getattr(cfg_env.traffic_sim, "num_drones", None) is not None:
            env_cfg.traffic_sim.num_drones = cfg_env.traffic_sim.num_drones
        if getattr(cfg_env.traffic_sim, "num_evtols", None) is not None:
            env_cfg.traffic_sim.num_evtols = cfg_env.traffic_sim.num_evtols
    for k in ["arrival_threshold", "predict_steps", "rew_evtol_future_penalty", "rew_drone_future_penalty"]:
        if getattr(cfg_env, k, None) is not None:
            setattr(env_cfg, k, getattr(cfg_env, k))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    if cfg.eval.checkpoint_path is None:
        raise ValueError("eval.checkpoint_path is required for evaluation")

    # Prepare output/video dir near checkpoint by default
    ckpt_dir = os.path.dirname(cfg.eval.checkpoint_path)
    video_dir = os.path.join(ckpt_dir, "eval_videos")
    os.makedirs(video_dir, exist_ok=True)

    # Isaac AppLauncher defaults
    from omni.isaac.lab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    default_args = parser.parse_args([])
    default_args.headless = cfg.eval.app.headless
    default_args.enable_cameras = cfg.eval.app.enable_cameras or bool(cfg.eval.video.enabled)
    app_launcher = AppLauncher(default_args)
    simulation_app = app_launcher.app

    # Import after AppLauncher
    from isaac_lab_envs.direct.uam_env import UamEnv
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    # Build environment
    env_cfg_cls = _import_from_string(cfg.env.cfg_class)
    env_cfg = env_cfg_cls()
    _apply_nested_overrides(env_cfg, cfg.env)

    record_video = bool(cfg.eval.video.enabled)
    render_mode = "rgb_array" if record_video else None
    base_env = UamEnv(cfg=env_cfg, render_mode=render_mode)
    if record_video:
        import gymnasium as gym
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": (lambda step: step % int(cfg.eval.video.interval) == 0),
            "video_length": int(cfg.eval.video.length),
            "disable_logger": True,
        }
        base_env = gym.wrappers.RecordVideo(base_env, **video_kwargs)

    env = SkrlVecEnvWrapper(base_env)

    device = torch.device(cfg.eval.device)

    # Recreate agent and load checkpoint
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    model_class = _import_from_string(cfg.model.model_class)
    features_extractor_cls = _import_from_string(cfg.model.features_extractor_class)
    shared_model = model_class(
        env.observation_space,
        env.action_space,
        device=device,
        features_dim=int(cfg.model.features_dim),
        net_arch=list(cfg.model.net_arch),
        log_std_init=float(cfg.model.log_std_init),
        features_extractor_cls=features_extractor_cls,
    )

    models = {"policy": shared_model, "value": shared_model}

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)
    agent = PPO(models=models, memory=memory, cfg=ppo_cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)
    agent.load(cfg.eval.checkpoint_path)

    # Rollout episodes using the policy model deterministically (mean actions)
    policy_model = models["policy"]
    policy_model.eval()
    episodes = int(cfg.eval.num_episodes)

    for ep in range(episodes):
        obs, _ = env.reset()
        dones = np.array([False] * env.num_envs)
        ep_rewards = np.zeros(env.num_envs, dtype=np.float32)
        while not dones.all():
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mean_actions, log_std, _ = policy_model.compute({"states": obs_t}, role="policy")
                actions = mean_actions.clamp(-1.0, 1.0)
                actions_np = actions.cpu().numpy()
            obs, rewards, terms, truncs, infos = env.step(actions_np)
            dones = np.logical_or(terms, truncs)
            ep_rewards += rewards
        print(f"Episode {ep+1}: mean_reward={ep_rewards.mean():.3f}")


if __name__ == "__main__":
    main()


