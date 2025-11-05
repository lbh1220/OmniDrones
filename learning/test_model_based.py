#!/usr/bin/env python3

"""Test script for model-based policies on Traffic Environment"""

import argparse
import os
import time
import numpy as np
import torch
from dataclasses import replace
import json
from typing import Dict, Any

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher
from stable_baselines3.common.logger import configure
import gymnasium as gym

# Policy imports
from isaac_lab_envs.direct.policies.base_policy import ModelBasedPolicy
from isaac_lab_envs.direct.policies.simple_policies import PurePursuitPolicy, ORCAPolicy


class PolicyConfig:
    """Simple config class for model-based policies"""
    def __init__(self, **kwargs):
        # Default values
        self.arrival_threshold = 1.0
        self.repulsion_gain = 2.0
        self.attraction_gain = 1.0
        self.obstacle_threshold = 15.0
        self.lookahead_distance = 3.0
        
        # ORCA-specific parameters
        self.time_step = 0.16
        self.neighbor_dist = 100.0
        self.max_neighbors = 10
        self.time_horizon = 11.0
        self.time_horizon_obst = 11.0
        self.safety_space = 1.5
        
        # Update with provided values
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_policy(policy_type: str, policy_config: PolicyConfig, env_cfg) -> Any:
    """Create policy based on type"""
    
    if policy_type == "pure_pursuit":
        return PurePursuitPolicy(policy_config, env_cfg, "PurePursuit")
    
    elif policy_type == "orca":
        return ORCAPolicy(policy_config, env_cfg, "ORCA")
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_test_env(cfg, args):
    """Create test environment"""
    from isaac_lab_envs.direct.traffic_env import TrafficEnv
    
    # Create environment
    env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
    
    return env

from rl.sb3.evaluate_policy import evaluate_policy

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test model-based policies on Traffic Environment")
    
    # Environment parameters
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes for evaluation")
    parser.add_argument("--drones_num", type=int, default=10, help="Number of traffic drones")
    parser.add_argument("--evtols_num", type=int, default=2, help="Number of traffic evtols")
    
    # Policy parameters
    parser.add_argument("--policy_type", type=str, 
                       choices=["pure_pursuit", "orca"],
                       default="orca", help="Policy type to test")
    
    # Policy hyperparameters
    parser.add_argument("--repulsion_gain", type=float, default=2.0, help="Repulsion gain for potential field")
    parser.add_argument("--attraction_gain", type=float, default=1.0, help="Attraction gain for potential field")
    parser.add_argument("--obstacle_threshold", type=float, default=15.0, help="Obstacle detection threshold")
    
    # ORCA-specific parameters
    parser.add_argument("--safety_space", type=float, default=1.5, help="ORCA safety space")
    parser.add_argument("--time_horizon", type=float, default=11.0, help="ORCA time horizon")
    parser.add_argument("--neighbor_dist", type=float, default=100.0, help="ORCA neighbor distance")
    parser.add_argument("--max_neighbors", type=int, default=10, help="ORCA max neighbors")

    parser.add_argument("--seed", type=int, default=425, help="Seed")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--video", action="store_true", default=True, help="Record video")
    parser.add_argument("--video_interval", type=int, default=10000, help="Video recording interval")
    parser.add_argument("--video_length", type=int, default=500, help="Video length")
    
    # Isaac Lab parameters
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Setup Isaac Lab
    args.headless = True
    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False
    
    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Import environment configuration (must be after AppLauncher)
    from omni.isaac.lab.utils.io import load_yaml
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # Create environment configuration
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    cfg.orca.enable = False
    cfg.traffic_sim.evtol.safety_radius = 8.0

    cfg.use_global_path = True
    cfg.debug_vis = True
    cfg.action_space_type = "gaussian"
    cfg.action_mode = "velocity_components"
    cfg.predict_steps = 5
    cfg.seed = args.seed
    # Setup viewer
    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),
        lookat=(0., 0., 1.)
    )
    
    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%m%d_%H%M%S")
        args.output_dir = f"runs/model_based_test/{args.policy_type}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    from omni.isaac.lab.utils.io import dump_yaml
    dump_yaml(os.path.join(args.output_dir, "env.yaml"), cfg)
    # Create test environment
    print(f"Creating test environment with {args.policy_type} policy...")
    base_env = create_test_env(cfg, args)
    env = base_env
    
    # Add video recording if requested
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(args.output_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during testing.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap with SB3 wrapper
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    env = Sb3VecEnvWrapper(env)
    
    # Create policy configuration
    policy_config = PolicyConfig(
        repulsion_gain=args.repulsion_gain,
        attraction_gain=args.attraction_gain,
        obstacle_threshold=args.obstacle_threshold,
        # ORCA parameters
        safety_space=args.safety_space,
        time_horizon=args.time_horizon,
        neighbor_dist=args.neighbor_dist,
        max_neighbors=args.max_neighbors
    )
    
    # Create policy
    print(f"Creating {args.policy_type} policy...")
    policy = create_policy(args.policy_type, policy_config, cfg)
    
    # Setup logging
    logger = configure(args.output_dir, ["stdout", "log"])
    logger.info(f"Starting evaluation with {args.policy_type} policy for {args.num_episodes} episodes")
    logger.info(f"Environment: {args.drones_num} drones, {args.evtols_num} evtols")
    
    # Set random seed
    env.seed(seed=args.seed)
    
    
    # Run evaluation
    results = evaluate_policy(policy, env, args.num_envs, args.num_episodes, logger)
    
    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Save configuration
    config_file = os.path.join(args.output_dir, "test_config.json")
    config_data = {
        "policy_type": args.policy_type,
        "num_envs": args.num_envs,
        "num_episodes": args.num_episodes,
        "drones_num": args.drones_num,
        "evtols_num": args.evtols_num,
        "policy_config": vars(policy_config)
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print(f"Testing completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
