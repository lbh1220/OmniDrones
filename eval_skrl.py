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


# 导入您的环境

from learning.skrl.models.utils import select_skrl_model
MODEL_LIST = [
    "final_model.pt",
    "best_model.pt",
    "best_model_1.pt",
    "best_model_2.pt",
    "best_model_3.pt",
    "best_model_4.pt",
    "best_model_5.pt",
]

def run_evaluation(experiment_path: str, 
                    num_episodes: int = 10, 
                    headless: bool = False, 
                    num_envs: int = 10, 
                    record_video: bool = False, 
                    model_name: str = "agent_190000.pt",
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

    save_dir = os.path.join(experiment_path, f"test_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(save_dir, exist_ok=True)
    # --- 3. 实例化环境 ---
    #    (这与 train_skrl.py 中的逻辑完全相同)
    print("Instantiating environment...")
    env_cfg_instance = instantiate(cfg.env)
    env_cfg_instance.seed = seed
    env_cfg_instance.num_envs = num_envs
    env_cfg_instance.scene = replace(env_cfg_instance.scene, num_envs=num_envs)
    env_cfg_instance.arrival_threshold = 2.0 # 测试时必须可以到达终点才行

    env_cfg_instance.area_bounds.xmin = -55.0
    env_cfg_instance.area_bounds.xmax = 55.0
    env_cfg_instance.area_bounds.ymin = -55.0
    env_cfg_instance.area_bounds.ymax = 55.0

    env_cfg_instance.debug_vis_num_envs = 1
    # env_cfg_instance.env_scale = 10.0
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
    env.seed(env_cfg_instance.seed) #

    # --- 4. 实例化 Agent ---
    #    (这与 train_skrl.py 中的逻辑几乎相同)
    print("Instantiating agent...")
    
    # a. 实例化模型
    models = {}
    model_params = OmegaConf.to_container(cfg.model, resolve=True) #
    model_params.pop("name", None)
    use_rnn = model_params.pop("use_rnn", False)
    feat_ext_cls_path = model_params.pop("features_extractor_cls")
    FeatureExtractorClass = get_class(feat_ext_cls_path)
    feat_ext_kwargs = model_params.pop("features_extractor_kwargs", {})
    MainModelClass = select_skrl_model(env.cfg.action_manager.action_space_type, use_rnn=use_rnn)
    if use_rnn:
        if "num_envs" not in model_params:
            model_params["num_envs"] = env.num_envs
        if "sequence_length" not in model_params:
            model_params["sequence_length"] = cfg.algo["rollouts"]
    shared_model = MainModelClass(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg.device,
        features_extractor_cls=FeatureExtractorClass,
        features_extractor_kwargs=feat_ext_kwargs,
        **model_params
    )
    models["policy"] = shared_model
    models["value"] = shared_model
    


    # c. 准备 PPO 配置
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_hyperparams = OmegaConf.to_container(cfg.algo, resolve=True) #
    # (复制预处理器逻辑)
    norm_obs = ppo_hyperparams.pop("norm_obs", False)
    norm_reward = ppo_hyperparams.pop("norm_reward", False)
    if norm_obs:
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": cfg.device}
    if norm_reward:
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": cfg.device}
    ppo_cfg.update(ppo_hyperparams)

    # d. 实例化 Agent
    rl_class = PPO_RNN if use_rnn else PPO
    agent = rl_class(
        models=models,
        memory=None,  # 评估时不需要 memory
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg.device,
    )

    # --- 5. 加载模型权重 ---
    if not model_name.endswith(".pt"):
        model_name = model_name + ".pt"
    model_path = os.path.join(experiment_path, model_name) #
    if not os.path.exists(model_path):
        model_path = os.path.join(experiment_path, 'checkpoints', model_name)
    print(f"Loading model weights from: {model_path}")
    agent.init()
    agent.load(model_path) #
    agent.set_running_mode("eval")

    # --- 6. 运行评估 ---
    
    if args.vis_velocity:
        from learning.vis_velocity import visualize_model_velocity
        visualize_model_velocity(agent, env_cfg_instance, base_env, save_dir, grid_res=env_cfg_instance.env_scale)
    else:
        print(f"Running evaluation for {num_episodes} episodes...")
        from learning.skrl.evaluate_agent import evaluate_policy
        eval_results = evaluate_policy(agent, env, num_envs, num_episodes)

        # save 这个json的result
        eval_summary = {
            "test_config": {
                "model_dir": model_path,
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
    parser.add_argument("--path", type=str, default="outputs/heter_traffic/traffic_attn_large/pot0p5", help="Path to the experiment directory (e.g., 'outputs/debug/2025-11-01_20-38')")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=425, help="Seed")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode (no UI).")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments.")
    parser.add_argument("--record_video", action="store_true", default=True, help="Record video.")
    parser.add_argument("--vis_velocity", action="store_true", default=False, help="Visualize velocity.")
    parser.add_argument("--model_name", type=str, default="final_model.pt", help="Model name.")
    args = parser.parse_args()
    if args.vis_velocity:
        args.record_video = False
    run_evaluation(args.path, args.episodes, args.headless, args.num_envs, args.record_video, args.model_name, args.seed)