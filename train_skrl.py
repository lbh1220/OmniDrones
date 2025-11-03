# file: train_skrl.py
import os
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_class, instantiate
from dataclasses import replace
from omni.isaac.lab.app import AppLauncher


# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler

import logging
# 限制第三方库日志级别，避免 DEBUG 噪音
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    # --- 1. 启动 Isaac Sim ---
    print("Launching Isaac Sim...")
    app_launcher = AppLauncher(headless=cfg.headless, enable_cameras=cfg.record_video)
    simulation_app = app_launcher.app
    
    # --- 2. 解析和保存配置 ---
    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Hydra 会自动创建输出目录，我们只需使用它
    save_dir = os.getcwd() # Hydra 自动设置当前工作目录
    print(f"Saving configs and logs to: {save_dir}")

    from omni.isaac.lab.utils.io import dump_yaml
        # 您的环境
    from isaac_lab_envs.direct.uam_env import UamEnv # 导入您唯一的 UamEnv
    from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg # 导入您唯一的 UamEnvCfg
    # 保存完整的合并后配置
    # dump_yaml(os.path.join(save_dir, "hydra_config.yaml"), cfg)
    dump_yaml(os.path.join(save_dir, "hydra_config.yaml"), OmegaConf.to_container(cfg, resolve=True))
    # --- 3. 设置随机种子 ---
    set_seed(cfg.seed, deterministic=True)

    # --- 4. 实例化环境 ---
    env_cfg_instance = instantiate(cfg.env)

    env_cfg_instance.scene = replace(env_cfg_instance.scene, num_envs=env_cfg_instance.num_envs)
    video_kwargs = None
    if cfg.record_video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": (lambda step: step % cfg.video_interval == 0),
            "video_length": cfg.video_length,
            "disable_logger": True,
        }
        env_cfg_instance.debug_vis = True

    env_cfg_instance.seed = cfg.seed
    # d. 实例化环境
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    env = UamEnv(cfg=env_cfg_instance, render_mode="rgb_array" if cfg.record_video else None) #
    if video_kwargs is not None:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = SkrlVecEnvWrapper(env)
    env.seed(cfg.seed)


    # --- 3. 【关键】实例化 SKRL 模型 ---
    print(f"Loading model config: {cfg.model.name}")
    models = {}

    # a. 从 cfg.model (YAML) 中解析参数
    #    (使用 OmegaConf.to_container 确保我们得到的是 Python 字典和列表, 而不是 OmegaConf 对象)
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params.pop("name", None) # 移除 name, 它不是 __init__ 参数
    use_rnn = model_params.pop("use_rnn", False)
    # b. 解析特征提取器【类】
    feat_ext_cls_path = model_params.pop("features_extractor_cls")
    FeatureExtractorClass = get_class(feat_ext_cls_path)

    # c. (可选) 提取特征提取器的参数
    feat_ext_kwargs = model_params.pop("features_extractor_kwargs", {})

    # d. 【您的 IF 逻辑】根据动作空间选择主模型类
    from learning.skrl.models.utils import select_skrl_model
    MainModelClass = select_skrl_model(env.cfg.action_manager.action_space_type, use_rnn=use_rnn)
    # e. 实例化主模型
    #    model_params 现在只包含 net_arch, features_dim, log_std_init 等...
    if use_rnn:
        if "num_envs" not in model_params:
            model_params["num_envs"] = env.num_envs
        if "sequence_length" not in model_params:
            model_params["sequence_length"] = cfg.algo["rollouts"]

    shared_model = MainModelClass(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg.device,
        features_extractor_cls=FeatureExtractorClass, # 传入【类】
        features_extractor_kwargs=feat_ext_kwargs,   # 传入【参数字典】
        **model_params # 传入【net_arch, features_dim 等...】
    )
    models["policy"] = shared_model
    models["value"] = shared_model

    # b. 设置 PPO 配置
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    # 将 cfg.algo (来自 conf/algo/ppo_skrl.yaml) 中的所有超参数合并
    # 使用 OmegaConf.to_container 转换为纯 Python 字典
    ppo_hyperparams = OmegaConf.to_container(cfg.algo, resolve=True)
    norm_obs = ppo_hyperparams.pop("norm_obs", False)
    norm_reward = ppo_hyperparams.pop("norm_reward", False)
    if norm_obs:
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": cfg.device}
    if norm_reward:
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": cfg.device}
    ppo_cfg.update(ppo_hyperparams)

    # c. 设置 SKRL 实验日志目录
    ppo_cfg["experiment"] = {
        "directory": os.path.dirname(save_dir), # 父目录, e.g., "outputs/"
        "experiment_name": os.path.basename(save_dir), # Hydra 生成的唯一目录名
        "write_interval": 100, # (可以从 cfg.log_interval 获取)
        "checkpoint_interval": 10000, # (可以从 cfg.checkpoint_interval 获取)
    }

    # d. 实例化 Memory
    memory = RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=env.num_envs, device=cfg.device)

    # e. 实例化 Agent
    rl_class = PPO_RNN if use_rnn else PPO
    agent = rl_class(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg.device,
    )

    # --- 6. 实例化 Trainer 并开始训练 ---
    print("Starting training...")
    trainer_cfg = {"timesteps": cfg.total_steps // env.num_envs, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # try:
    trainer.train()
    # try:
    # except KeyboardInterrupt:
    #     print("Training interrupted by user.")
    # finally:
    # --- 7. 清理 ---
    print("Training finished or interrupted. Saving final model...")
    agent.save(os.path.join(save_dir, "final_model.pt"))
    env.close()
    simulation_app.close()
    print("Done.")

if __name__ == "__main__":
    main()