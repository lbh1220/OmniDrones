# file: eval_skrl.py
import os
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_class, instantiate
from dataclasses import replace
# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from datetime import datetime
# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Policy imports
from isaac_lab_envs.direct.policies.base_policy import ModelBasedPolicy, PolicyConfig
from isaac_lab_envs.direct.policies.orca import ORCAPolicy
from isaac_lab_envs.direct.policies.pdc_policy import PDCPolicy


# 导入您的环境

from learning.skrl.models.utils import select_skrl_model

def run_evaluation(experiment_path: str, 
                    num_episodes: int = 10, 
                    headless: bool = False, 
                    livestream: int = -1,
                    num_envs: int = 10, 
                    record_video: bool = False, 
                    vis_velocity: bool = False, 
                    model_name: str = "orca", 
                    policy_overrides: dict | None = None,
                    seed: int = 425):
    """
    加载已训练的 PPO agent，并在环境中运行评估。

    Args:
        experiment_path (str): 指向包含 'hydra_config.yaml' 和 'final_model.pt' 的实验目录路径。
        num_episodes (int): 要运行的评估 episode 数量。
        headless (bool): 是否在无头模式下运行。
    """
    # --- 2. 启动 Isaac Sim ---
    print("Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=headless, livestream=livestream, enable_cameras=record_video)
    simulation_app = app_launcher.app
    
    # --- 1. 加载配置 ---
    print(f"Loading configuration from: {experiment_path}")
    
    # a. 加载已保存的纯字典配置
    from omni.isaac.lab.utils.io import load_yaml
    cfg_dict = load_yaml(os.path.join(experiment_path, "hydra_config.yaml"))
    
    # b. 将纯字典转回 OmegaConf 对象，以便我们可以使用 instantiate
    #    这是为了完美复现训练时的实例化过程
    cfg = OmegaConf.create(cfg_dict)

    policy_config = PolicyConfig()
    # Optional: override policy parameters without changing CLI surface
    if policy_overrides:
        for k, v in policy_overrides.items():
            setattr(policy_config, k, v)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    if policy_overrides:
        base_dir = os.path.join(experiment_path, model_name)
        os.makedirs(base_dir, exist_ok=True)
        # build subdirectory name from only the overrides kv
        sorted_items = [f"{k}={policy_overrides[k]}" for k in policy_overrides.keys()]
        subdir_name = "_".join(sorted_items) if len(sorted_items) > 0 else "default"
        save_dir = os.path.join(base_dir, f"{subdir_name}_{timestamp}")
    else:
        save_dir = os.path.join(experiment_path, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    # --- 3. 实例化环境 ---
    #    (这与 train_skrl.py 中的逻辑完全相同)
    print("Instantiating environment...")
    env_cfg_instance = instantiate(cfg.env)
    env_cfg_instance.seed = seed
    env_cfg_instance.num_envs = num_envs
    # env_cfg_instance.traffic_sim.num_evtols = 0

    env_cfg_instance.area_bounds.xmin = -55.0
    env_cfg_instance.area_bounds.xmax = 55.0
    env_cfg_instance.area_bounds.ymin = -55.0
    env_cfg_instance.area_bounds.ymax = 55.0

    env_cfg_instance.scene = replace(env_cfg_instance.scene, num_envs=num_envs)
    env_cfg_instance.arrival_threshold = 2.0 # 测试时必须可以到达终点才行
    env_cfg_instance.action_manager.action_space_type = "gaussian"
    env_cfg_instance.action_manager.action_mode = "velocity_components"
    env_cfg_instance.action_manager.rl_action_frame = "world"
    video_kwargs = None
    video_kwargs = None
    if record_video:
        if num_envs > 1:
            video_kwargs = {
                "video_folder": os.path.join(save_dir, "videos"),
                "step_trigger": (lambda step: step % cfg.video_interval == 0),
                "video_length": cfg.video_length,
                "disable_logger": True,
            }
        else:
            video_kwargs = {
                "video_folder": os.path.join(save_dir, "videos"),
                "episode_trigger": (lambda x: True), # Record every episode
                "disable_logger": True, 
            }
            # num_envs = 1 的情况
        env_cfg_instance.debug_vis = True    # (您可以在此处添加或修改 cfg 以进行评估，例如更改相机)
    # env_cfg_instance.viewer = ViewerCfg(...)
    from isaac_lab_envs.direct.uam_env import UamEnv
    from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg
    base_env = UamEnv(cfg=env_cfg_instance, render_mode="rgb_array" if record_video else None) #
    env = base_env
    if video_kwargs is not None:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    env = SkrlVecEnvWrapper(env)
    env.seed(env_cfg_instance.seed) #

    # --- 4. 实例化 Agent ---
    #    (这与 train_skrl.py 中的逻辑几乎相同)

    if model_name == "orca":
        agent = ORCAPolicy(policy_config, env_cfg_instance, "ORCA")
    elif model_name == "pdc":
        agent = PDCPolicy(policy_config, env_cfg_instance, "PDC")
    else:
        raise ValueError(f"Unknown policy: {model_name}")
    agent.bind_env(base_env)



    # --- 6. 运行评估 ---
    if vis_velocity:
        from learning.vis_velocity import visualize_model_velocity
        visualize_model_velocity(agent, env_cfg_instance, base_env, save_dir, grid_res=env_cfg_instance.env_scale)
    else:
        print(f"Running evaluation for {num_episodes} episodes...")
        from learning.skrl.evaluate_agent import evaluate_policy
        eval_results = evaluate_policy(agent, env, num_envs, num_episodes)
        # save 这个json的result
        eval_summary = {
            "test_config": {
                "model_dir": model_name,
                "num_episodes": num_episodes,
                "num_envs": num_envs,
                "policy_cfg": policy_config.__dict__
            },
            "results": eval_results
        }
        import json

        with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
            json.dump(eval_summary, f, indent=2)
    env.close()
    simulation_app.close()
    print("Done.")
    # Return artifacts for programmatic callers (e.g., sweeps)

if __name__ == "__main__":
    # 使用 argparse 来接收实验路径
    parser = argparse.ArgumentParser(description="Evaluate a trained SKRL agent.")
    parser.add_argument("--path", type=str, default="outputs/large_traffic/traffic_attn_large_acc/traffic_attn_20251123_0033", help="Path to the experiment directory (e.g., 'outputs/debug/2025-11-01_20-38')")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=425, help="Seed")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode (no UI).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument("--livestream", type=int, default=-1, help="Livestream.")
    parser.add_argument("--record_video", action="store_true", default=True, help="Record video.")
    parser.add_argument("--vis_velocity", action="store_true", default=False, help="Visualize velocity.")
    parser.add_argument("--model_name", type=str, default="orca", help="Policy to use (orca or pdc).")  
    # ORCA optional overrides (only applied if provided)
    parser.add_argument("--orca_safety_space", type=float, default=None, help="ORCA: safety_space")
    parser.add_argument("--orca_neighbor_dist", type=float, default=None, help="ORCA: neighbor_dist")
    parser.add_argument("--orca_max_neighbors", type=int, default=None, help="ORCA: max_neighbors")
    parser.add_argument("--orca_time_horizon", type=float, default=None, help="ORCA: time_horizon")
    parser.add_argument("--orca_time_horizon_obst", type=float, default=None, help="ORCA: time_horizon_obst")
    # PDC optional overrides (only applied if provided)
    parser.add_argument("--pdc_k1", type=float, default=None, help="PDC: k1 (attraction gain)")
    parser.add_argument("--pdc_k2", type=float, default=None, help="PDC: k2 (repulsion gain)")
    parser.add_argument("--pdc_l_i", type=float, default=None, help="PDC: l_i (filter gain)")
    parser.add_argument("--pdc_d1", type=float, default=None, help="PDC: d1 scale")
    parser.add_argument("--pdc_d2", type=float, default=None, help="PDC: d2 scale")
    parser.add_argument("--pdc_epsilon", type=float, default=None, help="PDC: epsilon")
    parser.add_argument("--pdc_epsilon_s", type=float, default=None, help="PDC: epsilon_s")
    args = parser.parse_args()
    if args.livestream > 0:
        args.headless = False
    if args.vis_velocity:
        args.record_video = False
    # Build policy_overrides only from provided args
    policy_overrides = None
    if args.model_name == "orca":
        mapping = {
            "safety_space": args.orca_safety_space,
            "neighbor_dist": args.orca_neighbor_dist,
            "max_neighbors": args.orca_max_neighbors,
            "time_horizon": args.orca_time_horizon,
            "time_horizon_obst": args.orca_time_horizon_obst,
        }
        filtered = {k: v for k, v in mapping.items() if v is not None}
        policy_overrides = filtered if len(filtered) > 0 else None
    elif args.model_name == "pdc":
        mapping = {
            "k1": args.pdc_k1,
            "k2": args.pdc_k2,
            "l_i": args.pdc_l_i,
            "d1": args.pdc_d1,
            "d2": args.pdc_d2,
            "epsilon": args.pdc_epsilon,
            "epsilon_s": args.pdc_epsilon_s,
        }
        filtered = {k: v for k, v in mapping.items() if v is not None}
        policy_overrides = filtered if len(filtered) > 0 else None
    
    run_evaluation(args.path, 
    args.episodes, 
    args.headless, 
    args.livestream,
    args.num_envs, 
    args.record_video, 
    args.vis_velocity, 
    args.model_name, 
    policy_overrides, 
    args.seed) 