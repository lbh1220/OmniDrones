#!/usr/bin/env python3

"""Standard SB3 training script for Traffic Environment
使用标准SB3 PPO和注意力机制特征提取器训练无人机交通环境
"""

import argparse
import os
import time
import yaml
from datetime import datetime
from dataclasses import replace
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

# SB3-contrib imports for recurrent PPO
from sb3_contrib import RecurrentPPO
from rl.sb3.vec_normalize import VecNormalize
from rl.sb3.network_utils import linear_schedule_with_min
from rl.sb3.custom_callback import SucessRateCallback
# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher



# Import our custom features extractor and GRU policy
from rl.sb3.attention_features_extractor import AttentionFeaturesExtractor
from rl.sb3.gru_recurrent_policy import GRUMultiInputActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
# Wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


def create_env(cfg, headless=True, record_video=False, video_kwargs=None):
    """创建并包装环境"""
    from isaac_lab_envs.direct.traffic_env import TrafficEnv
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    # Create environment
    env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if record_video else None)

    # Add video recording if requested
    if record_video and video_kwargs is not None:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # Wrap with SB3 wrapper
    env = Sb3VecEnvWrapper(env)

    
    return env


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Traffic Environment with Standard SB3")
    
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
    
    # Recurrent network parameters
    parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN-based recurrent policy")
    parser.add_argument("--shared_gru", action="store_true", default=True, help="Share GRU between actor and critic")
    parser.add_argument("--rnn_net_arch", type=str, default="lstm", help="RNN network architecture")
    
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
    parser.add_argument("--wandb_project", type=str, default="traffic_standard", help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    
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
        eye=(125, 0., 125),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    
    if args.video:
        cfg.debug_vis = True
    
    # Traffic configuration
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    cfg.traffic_sim.evtol.safety_radius = args.evtol_radius
    cfg.predict_steps = args.predict_steps
    # cfg.traffic_sim.evtol.random_safety_radius = True
    
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

    # algo_args.action_space_type = cfg.action_space_type
    # this will used when set beta distribution action space
    
    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"traffic_standard_{timestamp}"
    else:
        # add timestamp to experiment name
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"{args.experiment_name}_{timestamp}"

    # Create save directory
    save_dir = f"runs/traffic_standard/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"experiment configuration:")
    print(f"- experiment name: {args.experiment_name}")
    print(f"- save directory: {save_dir}")
    print(f"- num_envs: {args.num_envs}")
    print(f"- drones_num: {args.drones_num}")
    print(f"- evtols_num: {args.evtols_num}")
    print(f"- predict_steps: {args.predict_steps}")
    print(f"- use_rnn: {args.use_rnn}")
    if args.use_rnn:
        print(f"- shared_gru: {args.shared_gru}")
    
    # Save environment and training configurations
    from omni.isaac.lab.utils.io import dump_yaml
    dump_yaml(os.path.join(save_dir, "env_config.yaml"), cfg)
    
    # Save training arguments as YAML for human readability
    args_dict = vars(args)
    # Convert any non-serializable objects to strings
    serializable_args = {}
    for key, value in args_dict.items():
        try:
            yaml.dump({key: value})  # Test if serializable
            serializable_args[key] = value
        except:
            serializable_args[key] = str(value)  # Convert to string if not serializable
    
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
    
    # Set seed
    env.seed(seed=cfg.seed)
    
    # Apply normalization if requested
    if args.norm_obs or args.norm_reward:
        norm_obs_keys = ['robot_node', 'spatial_edges', 'temporal_edges']
        env = VecNormalize(
            env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
            training=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
            norm_obs_keys=norm_obs_keys,
            shared_across_agents=True
        )
    
    # Configure policy with our attention features extractor
    # Matching original network architecture from selfAttn_srnn_temp_node.py
    policy_kwargs = dict(
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),  # concat robot_states and hidden_attn_weighted
        net_arch=dict(pi=[256, 256], vf=[256, 256]),       # Same as original actor/critic
        activation_fn=nn.Tanh,  
        ortho_init=True,
    )
    
    # Add GRU-specific configuration if using GRU
    if args.use_rnn:
        if args.rnn_net_arch == "gru":
            policy_kwargs.update({
                'gru_hidden_size': 128,
                'n_gru_layers': 1,
                'shared_gru': args.shared_gru,
                'enable_critic_gru': not args.shared_gru,  # 如果不共享，则启用critic GRU
            })
        elif args.rnn_net_arch == "lstm":
            policy_kwargs.update({
                'lstm_hidden_size': 128,
                'n_lstm_layers': 1,
                'shared_lstm': args.shared_gru,
                'enable_critic_lstm': not args.shared_gru,  # 如果不共享，则启用critic GRU
            })
        else:
            raise ValueError(f"Invalid RNN network architecture: {args.rnn_net_arch}")
    
        
    # Create learning rate schedule
    lr_schedule = linear_schedule_with_min(args.learning_rate, args.learning_rate * 0.1)
    
    # Create model based on whether to use GRU or not
    if args.use_rnn:
        # Calculate batch size for recurrent PPO
        batch_size = args.n_steps * args.num_envs // args.num_mini_batch
        policy_class = GRUMultiInputActorCriticPolicy if args.rnn_net_arch == "gru" else MultiInputLstmPolicy
        model = RecurrentPPO(
            policy_class,
            env,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            learning_rate=lr_schedule,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
        )
    else:
        # Standard PPO
        batch_size = args.n_steps * args.num_envs // args.num_mini_batch
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            learning_rate=lr_schedule,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
        )
    
    # Set up logger
    new_logger = configure(os.path.join(save_dir, "logs"), ["stdout", "tensorboard", "log"])
    model.set_logger(new_logger)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    SR_check_callback = SucessRateCallback(check_freq=getattr(args, 'log_interval', 10),
                                    save_path=os.path.join(save_dir, 'checkpoints'),
                                    queue_size=args.num_envs,
                                    name_prefix='SR')
    callbacks.append(SR_check_callback)
    

    
    # Log training parameters
    model.logger.info("start training...")
    model.logger.info(f"total timesteps: {args.total_timesteps}")
    model.logger.info(f"learning rate: {args.learning_rate}")
    model.logger.info(f"n_steps: {args.n_steps}")
    model.logger.info(f"batch_size: {batch_size}")
    model.logger.info(f"n_epochs: {args.n_epochs}")
    if args.use_rnn:
        model.logger.info(f"shared_gru: {args.shared_gru}")
    
    # Start training
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=getattr(args, 'log_interval', 10)
    )
    end_time = time.time()
    
    # Save final model
    model_path = os.path.join(save_dir, "final_model")
    model.save(model_path)
    model.get_vec_normalize_env().save(os.path.join(save_dir, "final_model_vecnormalize.pkl"))
    
    # Save normalization parameters if used
    if hasattr(env, 'save'):
        env.save(os.path.join(save_dir, "final_model_vecnormalize.pkl"))
    
    print(f"\ntraining completed!")
    print(f"training time: {(end_time - start_time) / 3600:.2f} hours")
    print(f"final model saved to: {model_path}")
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.save(model_path + ".zip")
        wandb.finish()
        
    # Clean up
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
