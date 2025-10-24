#!/usr/bin/env python3

"""
SKRL training script for City Navigation Environment (simplified)
使用SKRL框架训练城市环境，采用最简网络：分别处理 robot_node 与 lidar 后融合。
通过可插拔的 feature network 复用 SharedAttentionContinuous。
"""

import argparse
import os
import time
from datetime import datetime
import yaml
import torch

# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher


def create_env(cfg, headless=True, record_video=False, video_kwargs=None):
    """创建并包装 City 环境为 SKRL 兼容格式，支持视频录制"""
    from isaac_lab_envs.direct.uam_env import UamEnv
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = UamEnv(cfg=cfg, render_mode="rgb_array" if record_video else None)

    # 录制视频
    if record_video and video_kwargs is not None:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)
    return env


def create_agent(env, device, args, experient_cfg):
    """创建简化的 PPO agent，复用 SharedAttentionContinuous + CityFeaturesNetwork"""
    from learning.skrl.custom_agent import SharedAttentionContinuous
    from learning.skrl.networks import CityFeaturesNetwork

    models = {}
    shared_model = SharedAttentionContinuous(
        env.observation_space,
        env.action_space,
        device,
        features_dim=128,
        net_arch=[256, 256],
        features_extractor_cls=CityFeaturesNetwork
    )

    models["policy"] = shared_model
    models["value"] = shared_model

    cfg = PPO_DEFAULT_CONFIG.copy()
    # Training parameters
    cfg["learning_epochs"] = args.n_epochs
    cfg["mini_batches"] = args.num_mini_batch
    cfg["discount_factor"] = args.gamma
    cfg["lambda"] = 0.95  # GAE lambda
    cfg["learning_rate"] = args.learning_rate
    cfg["learning_rate_scheduler"] = None  # Start with no scheduler
    cfg["learning_rate_scheduler_kwargs"] = {}
    
    # PPO specific parameters
    cfg["ratio_clip"] = args.clip_range
    cfg["value_clip"] = args.clip_range
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = args.ent_coef
    cfg["value_loss_scale"] = 0.5
    cfg["kl_threshold"] = 0
    
    # Normalization
    if args.norm_obs:
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    
    if args.norm_reward:
        cfg["value_preprocessor"] = RunningStandardScaler  
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Configure agent experiment settings
    cfg["experiment"] = experient_cfg
    
    # Memory configuration
    cfg["rollouts"] = args.n_steps  # Steps per rollout

    memory = RandomMemory(memory_size=args.n_steps, num_envs=env.num_envs, device=device)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train City Navigation Environment with SKRL (simplified)")

    # Environment parameters
    parser.add_argument("--num_envs", type=int, default=128)

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=5000000, help="Total timesteps")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of steps per update")
    parser.add_argument("--num_mini_batch", type=int, default=32, help="Number of mini batches")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--clip_range", type=float, default=0.1, help="PPO clip range")
    parser.add_argument("--seed", type=int, default=425, help="Seed")
    
    # Normalization
    parser.add_argument("--norm_obs", action="store_true", help="Normalize observations")
    parser.add_argument("--norm_reward", action="store_true", help="Normalize rewards")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Video recording
    parser.add_argument("--video", action="store_true", default=True, help="Record videos")
    parser.add_argument("--video_interval", type=int, default=10000, help="Video interval (steps)")
    parser.add_argument("--video_length", type=int, default=500, help="Video length (frames)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force headless for training
    args.headless = True
    # Enable cameras only when recording
    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import env cfg after AppLauncher
    from isaac_lab_envs.direct.uam_env_cfg import CityUamEnvCfg
    from omni.isaac.lab.envs.common import ViewerCfg

    # Build env cfg
    cfg = CityUamEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    bounds = cfg.area_bounds
    range_x = bounds.xmax - bounds.xmin
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(0, 0.0, range_x*1.5),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    # Ensure continuous action space in [-1, 1]
    cfg.action_manager.action_space_type = "gaussian"
    cfg.action_manager.action_mode = "velocity_components"

    # global path
    cfg.use_global_path = True
    from isaac_lab_envs.direct.mdp.observations import CityNavObservationProcessorWithPath, CityNavObservationProcessor
    cfg.observation_processor_cls = CityNavObservationProcessorWithPath
    from isaac_lab_envs.direct.mdp.rewards import CityNavRewardCalculatorWithPath, CityNavRewardCalculator
    cfg.reward_calculator_cls = CityNavRewardCalculatorWithPath

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"city_skrl_{timestamp}"
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"{args.experiment_name}_{timestamp}"

    # Create save directory
    save_dir = f"runs/city_skrl/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"experiment configuration:")
    print(f"- experiment name: {args.experiment_name}")
    print(f"- save directory: {save_dir}")
    print(f"- num_envs: {args.num_envs}")
    print(f"- device: {args.device}")

    # Set device and seed
    device = torch.device(args.device)
    set_seed(args.seed)

    # Video config
    video_kwargs = None
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": (lambda step: step % args.video_interval == 0),
            "video_length": args.video_length,
            "disable_logger": True,
        }
        # 打开可视化辅助
        cfg.debug_vis = True

    # Save environment and training configurations
    from omni.isaac.lab.utils.io import dump_yaml
    dump_yaml(os.path.join(save_dir, "env_config.yaml"), cfg)
    
    # Save training arguments
    args_dict = vars(args)
    serializable_args = {}
    for key, value in args_dict.items():
        try:
            yaml.dump({key: value})
            serializable_args[key] = value
        except:
            serializable_args[key] = str(value)
    
    with open(os.path.join(save_dir, "training_args.yaml"), 'w') as f:
        yaml.dump(serializable_args, f, default_flow_style=False, indent=2)
    
    # Create env
    env = create_env(cfg, headless=True, record_video=args.video, video_kwargs=video_kwargs)

    # Build agent
    experient_cfg = {
        "directory": "runs/city_skrl",
        "experiment_name": args.experiment_name,
        "write_interval": 1,
        "checkpoint_interval": 100_000,
    }
    agent = create_agent(env, device, args, experient_cfg)


    # Trainer
    timesteps = args.total_timesteps // args.num_envs
    cfg_trainer = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    print("start training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pt"))

    print(f"\ntraining completed! time: {(end_time - start_time) / 3600:.2f} hours")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


