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
from isaac_lab_envs.direct.policies.base_policy import ModelBasedPolicy
from isaac_lab_envs.direct.policies.simple_policies import PurePursuitPolicy, ORCAPolicy, PolicyConfig



# 导入您的环境

from learning.skrl.models.utils import select_skrl_model

def run_evaluation(experiment_path: str, num_episodes: int = 10, headless: bool = False, num_envs: int = 10, record_video: bool = False):
    """
    加载已训练的 PPO agent，并在环境中运行评估。

    Args:
        experiment_path (str): 指向包含 'hydra_config.yaml' 和 'final_model.pt' 的实验目录路径。
        num_episodes (int): 要运行的评估 episode 数量。
        headless (bool): 是否在无头模式下运行。
    """
    # --- 2. 启动 Isaac Sim ---
    print("Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=headless, enable_cameras=record_video)
    simulation_app = app_launcher.app
    
    # --- 1. 加载配置 ---
    print(f"Loading configuration from: {experiment_path}")
    
    # a. 加载已保存的纯字典配置
    from omni.isaac.lab.utils.io import load_yaml
    cfg_dict = load_yaml(os.path.join(experiment_path, "hydra_config.yaml"))
    
    # b. 将纯字典转回 OmegaConf 对象，以便我们可以使用 instantiate
    #    这是为了完美复现训练时的实例化过程
    cfg = OmegaConf.create(cfg_dict)

    save_dir = os.path.join(experiment_path, f"orca_{datetime.now().strftime('%Y%m%d_%H%M')}")
    # --- 3. 实例化环境 ---
    #    (这与 train_skrl.py 中的逻辑完全相同)
    print("Instantiating environment...")
    env_cfg_instance = instantiate(cfg.env)
    env_cfg_instance.num_envs = num_envs
    env_cfg_instance.scene = replace(env_cfg_instance.scene, num_envs=num_envs)
    env_cfg_instance.arrival_threshold = 2.0 # 测试时必须可以到达终点才行
    env_cfg_instance.action_manager.action_space_type = "gaussian"
    env_cfg_instance.action_manager.action_mode = "velocity_components"
    env_cfg_instance.action_manager.rl_action_frame = "world"
    video_kwargs = None
    if record_video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": (lambda step: step % cfg.video_interval == 0),
            "video_length": cfg.video_length,
            "disable_logger": True,
        }
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
    env.seed(cfg.seed) #

    # --- 4. 实例化 Agent ---
    #    (这与 train_skrl.py 中的逻辑几乎相同)
    policy_config = PolicyConfig()
    agent = ORCAPolicy(policy_config, env_cfg_instance, "ORCA")
    agent.bind_env(base_env)



    # --- 6. 运行评估 ---
    print(f"Running evaluation for {num_episodes} episodes...")
    from learning.skrl.evaluate_agent import evaluate_policy
    eval_results = evaluate_policy(agent, env, num_envs, num_episodes)
    # save 这个json的result
    eval_summary = {
        "test_config": {
            "model_dir": "orca",
            "num_episodes": num_episodes,
            "num_envs": num_envs
        },
        "results": eval_results
    }
    import json

    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
    env.close()
    simulation_app.close()
    print("Done.")

if __name__ == "__main__":
    # 使用 argparse 来接收实验路径
    parser = argparse.ArgumentParser(description="Evaluate a trained SKRL agent.")
    parser.add_argument("--path", type=str, default="outputs/intent_attn/dynamic_traffic_features_20251104_1854", help="Path to the experiment directory (e.g., 'outputs/debug/2025-11-01_20-38')")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (no UI).")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments.")
    parser.add_argument("--record_video", action="store_true", default=True, help="Record video.")
    args = parser.parse_args()
    
    run_evaluation(args.path, args.episodes, args.headless, args.num_envs, args.record_video)