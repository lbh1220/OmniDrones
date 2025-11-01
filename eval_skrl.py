# file: eval_skrl.py
import os
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_class, instantiate

# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators import deterministic_model
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from omni.isaac.lab.utils.io import load_yaml # <--- 使用 load_yaml

# 导入您的环境
from isaac_lab_envs.direct.uam_env import UamEnv
from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg
from learning.skrl.models.utils import select_skrl_model


def run_evaluation(experiment_path: str, num_episodes: int = 10, headless: bool = False):
    """
    加载已训练的 PPO agent，并在环境中运行评估。

    Args:
        experiment_path (str): 指向包含 'hydra_config.yaml' 和 'final_model.pt' 的实验目录路径。
        num_episodes (int): 要运行的评估 episode 数量。
        headless (bool): 是否在无头模式下运行。
    """
    
    # --- 1. 加载配置 ---
    print(f"Loading configuration from: {experiment_path}")
    
    # a. 加载已保存的纯字典配置
    cfg_dict = load_yaml(os.path.join(experiment_path, "hydra_config.yaml"))
    
    # b. 将纯字典转回 OmegaConf 对象，以便我们可以使用 instantiate
    #    这是为了完美复现训练时的实例化过程
    cfg = OmegaConf.create(cfg_dict)

    # --- 2. 启动 Isaac Sim ---
    print("Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    
    # --- 3. 实例化环境 ---
    #    (这与 train_skrl.py 中的逻辑完全相同)
    print("Instantiating environment...")
    env_cfg_instance = instantiate(cfg.env)
    
    # (您可以在此处添加或修改 cfg 以进行评估，例如更改相机)
    # env_cfg_instance.viewer = ViewerCfg(...)
    
    env = UamEnv(cfg=env_cfg_instance, render_mode="rgb_array" if not headless else None)
    env = SkrlVecEnvWrapper(env)
    env.seed(cfg.seed) #

    # --- 4. 实例化 Agent ---
    #    (这与 train_skrl.py 中的逻辑几乎相同)
    print("Instantiating agent...")
    
    # a. 实例化模型
    models = {}
    model_params = OmegaConf.to_container(cfg.model, resolve=True) #
    model_params.pop("name", None)
    feat_ext_cls_path = model_params.pop("features_extractor_cls")
    FeatureExtractorClass = get_class(feat_ext_cls_path)
    feat_ext_kwargs = model_params.pop("features_extractor_kwargs", {})
    MainModelClass = select_skrl_model(env.cfg.action_manager.action_space_type, use_rnn=False)
    
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
    
    # b. 实例化确定性模型（用于评估）
    deterministic_model(models["policy"]) # SKRL 辅助函数，将策略转为确定性

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
    agent = PPO(
        models=models,
        memory=None,  # 评估时不需要 memory
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg.device,
    )

    # --- 5. 加载模型权重 ---
    model_path = os.path.join(experiment_path, "final_model.pt") #
    print(f"Loading model weights from: {model_path}")
    agent.load(model_path) #

    # --- 6. 运行评估 ---
    print(f"Running evaluation for {num_episodes} episodes...")
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = torch.tensor([False] * env.num_envs, device=cfg.device)
        truncated = torch.tensor([False] * env.num_envs, device=cfg.device)
        total_reward = 0
        
        while not (terminated.any() or truncated.any()):
            # 1. Agent 采取行动 (使用 .act() 进行评估)
            actions = agent.act(obs, role="policy")[0]
            
            # 2. 环境步进
            obs, reward, terminated, truncated, info = env.step(actions)
            
            total_reward += reward[0].item() # 假设我们只关心第一个 env 的奖励
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")

    # --- 7. 报告结果和清理 ---
    mean_reward = sum(episode_rewards) / num_episodes
    print("\n--- Evaluation Finished ---")
    print(f"Mean reward over {num_episodes} episodes: {mean_reward:.2f}")
    
    env.close()
    simulation_app.close()
    print("Done.")

if __name__ == "__main__":
    # 使用 argparse 来接收实验路径
    parser = argparse.ArgumentParser(description="Evaluate a trained SKRL agent.")
    parser.add_argument("path", type=str, help="Path to the experiment directory (e.g., 'outputs/debug/2025-11-01_20-38')")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no UI).")
    
    args = parser.parse_args()
    
    run_evaluation(args.path, args.episodes, args.headless)