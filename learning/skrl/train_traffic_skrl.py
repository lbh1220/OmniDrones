#!/usr/bin/env python3

"""SKRL training script for Traffic Environment
使用SKRL框架和注意力机制训练无人机交通环境
支持离散和连续动作空间
"""

import argparse
import os
import time
import yaml
from datetime import datetime
from dataclasses import replace
import torch
import torch.nn as nn

# SKRL imports
import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

from attention_networks import (
    SharedAttentionContinuous, 
    SharedAttentionDiscrete,
    SharedAttentionGRUContinuous,
    SharedAttentionGRUDiscrete
)
# Wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


def create_env(cfg, headless=True, record_video=False, video_kwargs=None):
    """创建并包装环境为SKRL兼容格式"""
    from isaac_lab_envs.direct.traffic_env import TrafficEnv
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    # Create environment
    env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if record_video else None)
    
    # Add video recording if requested
    if record_video and video_kwargs is not None:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap with SKRL wrapper
    env = SkrlVecEnvWrapper(env)
    
    return env


def create_agent(env, device, args, experient_cfg):
    """创建SKRL PPO agent，使用官方的共享模型模式"""
    
    # Check action space type
    is_discrete = hasattr(env.action_space, 'n')
    
    # Create shared model following official SKRL pattern
    # Both policy and value will use the same instance
    models = {}
    
    if is_discrete:
        # Create shared model for discrete actions
        if args.use_rnn:
            shared_model = SharedAttentionGRUDiscrete(
                env.observation_space,
                env.action_space,
                device,
                features_dim=128,
                net_arch=[256, 256],
                num_envs=args.num_envs,
                num_layers=1,
                hidden_size=128,
                sequence_length=args.n_steps,
                unnormalized_log_prob=True
            )
        else:
            shared_model = SharedAttentionDiscrete(
                env.observation_space,
                env.action_space,
                device,
                features_dim=128,
                net_arch=[256, 256],
                unnormalized_log_prob=True
            )
    else:
        # Create shared model for continuous actions
        if args.use_rnn:
            shared_model = SharedAttentionContinuous(
                env.observation_space,
                env.action_space,
                device,
                features_dim=128,
                net_arch=[256, 256],
                num_envs=args.num_envs,
                num_layers=1,
                hidden_size=128,
                sequence_length=args.n_steps,
                log_std_init=0.0,
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2,
                reduction="sum"
            )
        else:
            shared_model = SharedAttentionContinuous(
                env.observation_space,
                env.action_space,
                device,
                features_dim=128,
                net_arch=[256, 256],
                log_std_init=0.0,
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2,
                reduction="sum"
            )
    
    # Both policy and value use the same shared model instance
    models["policy"] = shared_model
    models["value"] = shared_model  # same instance: shared model
    
    # Configure PPO
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
    memory_size = args.n_steps  # Memory size should match rollouts
    
    # Create memory
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
    
    # Create agent
    if args.use_rnn:
        agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    else:
        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )
    
    return agent


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Traffic Environment with SKRL")
    
    # Environment parameters
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--drones_num", type=int, default=10, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=0, help="Number of evtols")
    parser.add_argument("--evtol_radius", type=float, default=10.0, help="Evtol radius")
    parser.add_argument("--predict_steps", type=int, default=5, help="Prediction steps")
    
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
    
    # Reward parameters
    parser.add_argument("--rew_success", type=float, default=15.0, help="Success reward")
    parser.add_argument("--rew_collision", type=float, default=-16.0, help="Collision penalty")
    parser.add_argument("--drone_future_penalty", type=float, default=0.0, help="Drone future penalty")
    parser.add_argument("--evtol_future_penalty", type=float, default=0.0, help="Evtol future penalty")
    parser.add_argument("--drones_threshold_factor", type=float, default=None, help="Drones threshold factor")
    parser.add_argument("--drones_decay_factor", type=float, default=None, help="Drones decay factor")
    parser.add_argument("--evtols_threshold_factor", type=float, default=None, help="Evtols threshold factor")
    parser.add_argument("--evtols_decay_factor", type=float, default=None, help="Evtols decay factor")
    parser.add_argument("--rew_action_penalty", type=float, default=0.0, help="Action penalty")
    
    parser.add_argument("--use_global_path", action="store_true", default=False, help="Use global path")
    parser.add_argument("--rew_cross_track_coeff", type=float, default=0.0, help="Cross track coeff")
    parser.add_argument("--rew_cross_track_alpha", type=float, default=1.0, help="Cross track alpha")

    parser.add_argument("--action_space_type", type=str, default="discrete", help="Action space type")
    parser.add_argument("--action_space_num_per_dim", type=int, default=7, help="Action space num per dim")
    parser.add_argument("--action_mode", type=str, default="velocity_components", help="Action mode")

    # Normalization
    parser.add_argument("--norm_obs", action="store_true", help="Normalize observations")
    parser.add_argument("--norm_reward", action="store_true", help="Normalize rewards")

    parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save_freq", type=int, default=100000, help="Save frequency")
    parser.add_argument("--eval_freq", type=int, default=50000, help="Evaluation frequency")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval")
    
    # Video recording
    parser.add_argument("--video", action="store_true", help="Record videos")
    parser.add_argument("--video_interval", type=int, default=2500, help="Video interval")
    parser.add_argument("--video_length", type=int, default=250, help="Video length")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="traffic_skrl", help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Force headless mode for training
    args.headless = True
    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False
        
    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Import environment config (must be after AppLauncher)
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # Create environment configuration
    cfg = TrafficEnvCfg()
    cfg.seed = args.seed
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    
    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),
        lookat=(0., 0., 1.)
    )
    
    if args.video:
        cfg.debug_vis = True
    
    # Traffic configuration
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    cfg.traffic_sim.evtol.safety_radius = args.evtol_radius
    cfg.predict_steps = args.predict_steps
    
    # Reward configuration
    cfg.rew_success = args.rew_success
    cfg.rew_collision = -abs(args.rew_collision)
    cfg.rew_drone_future_penalty = -abs(args.drone_future_penalty)
    cfg.rew_evtol_future_penalty = -abs(args.evtol_future_penalty)
    cfg.rew_action_penalty = -abs(args.rew_action_penalty)

    if args.drones_threshold_factor is not None:
        cfg.rew_drones_threshold_factor = args.drones_threshold_factor
    if args.drones_decay_factor is not None:
        cfg.rew_drones_decay_factor = args.drones_decay_factor
    if args.evtols_threshold_factor is not None:
        cfg.rew_evtols_threshold_factor = args.evtols_threshold_factor
    if args.evtols_decay_factor is not None:
        cfg.rew_evtols_decay_factor = args.evtols_decay_factor

    cfg.use_global_path = args.use_global_path
    cfg.rew_cross_track_coeff = args.rew_cross_track_coeff
    cfg.rew_cross_track_alpha = args.rew_cross_track_alpha

    # Action space configuration
    cfg.action_space_type = args.action_space_type
    cfg.action_mode = args.action_mode
    cfg.action_space_num_per_dim = args.action_space_num_per_dim
    
    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"traffic_skrl_{timestamp}"
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"{args.experiment_name}_{timestamp}"

    # Create save directory
    save_dir = f"runs/traffic_skrl/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"experiment configuration:")
    print(f"- experiment name: {args.experiment_name}")
    print(f"- save directory: {save_dir}")
    print(f"- num_envs: {args.num_envs}")
    print(f"- drones_num: {args.drones_num}")
    print(f"- evtols_num: {args.evtols_num}")
    print(f"- predict_steps: {args.predict_steps}")
    print(f"- action_space_type: {args.action_space_type}")
    print(f"- device: {args.device}")
    
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
    
    print(f"configuration files saved:")
    print(f"- env_config: {os.path.join(save_dir, 'env_config.yaml')}")
    print(f"- training_args: {os.path.join(save_dir, 'training_args.yaml')}")
    
    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            config=vars(args),
            sync_tensorboard=True,
            dir=save_dir
        )
        print(f"Wandb initialized: {wandb.run.url}")
    
    # Set device
    device = torch.device(args.device)
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    video_kwargs = None
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
    
    env = create_env(cfg, headless=True, record_video=args.video, video_kwargs=video_kwargs)
    
    # Create agent

    experient_cfg = {
        "directory": "runs/traffic_skrl",
        "experiment_name": args.experiment_name,
        "write_interval": args.log_interval,
        "checkpoint_interval": args.save_freq,
    }
    # Configure Wandb if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        experient_cfg["wandb"] = True
        experient_cfg["wandb_kwargs"] = {"sync_tensorboard": True}
    
    agent = create_agent(env, device, args, experient_cfg)
    
    # Create callbacks


    # Create trainer
    timesteps = args.total_timesteps//args.num_envs
    cfg_trainer = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    # Log training parameters
    print("start training...")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"learning rate: {args.learning_rate}")
    print(f"n_steps: {args.n_steps}")
    print(f"batch_size: {args.n_steps * args.num_envs // args.num_mini_batch}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"action space type: {args.action_space_type}")
    
    # Start training with custom callback monitoring
    start_time = time.time()
    trainer.train()
    # # Simple training loop with callback support
    # for timestep in range(0, args.total_timesteps, args.n_steps * env.num_envs):
    #     # Train for one rollout
    #     trainer.train()
        
    #     # Call callbacks
    #     for callback in callbacks:
    #         if hasattr(callback, 'on_timestep_end'):
    #             callback.on_timestep_end(trainer, timestep, args.total_timesteps)
    
    end_time = time.time()
    
    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pt"))

    
    print(f"\ntraining completed!")
    print(f"training time: {(end_time - start_time) / 3600:.2f} hours")
    print(f"final model saved to: {save_dir}")
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.save(model_path)
        wandb.finish()
        
    # Clean up
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main() 