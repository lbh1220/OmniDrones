#!/usr/bin/env python3

"""
SKRL-based traffic environment training script
迁移自SB3的训练架构，使用SKRL进行原生GPU训练
"""

import argparse
import os
import sys
import time
from datetime import datetime
from dataclasses import replace
import yaml
from pathlib import Path

import torch
import torch.nn as nn

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher

# 导入SKRL相关包
import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer

# 导入自定义模块
from skrl_models import GraphAttentionPolicy, GraphAttentionPolicyDiscrete
from skrl_callbacks import (
    SuccessRateCallback, TensorboardLogger, WandbLogger,
    EpisodeStatsCallback, ModelCheckpointCallback
)

# 添加wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


def create_env(cfg, headless=True):
    """创建并包装SKRL环境"""
    from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvWithCurriculum
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    
    # 创建环境
    if cfg.curriculum_learning:
        env = TrafficEnvWithCurriculum(cfg=cfg)
    else:
        env = TrafficEnv(cfg=cfg)
    
    # 使用SKRL包装器包装
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    return env


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_agent_config(config: dict) -> dict:
    """根据配置创建SKRL PPO配置"""
    agent_cfg = config["agent"]
    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    
    ppo_config = PPO_DEFAULT_CONFIG.copy()
    
    # 基本训练参数
    ppo_config["rollouts"] = agent_cfg["rollouts"]
    ppo_config["mini_batches"] = agent_cfg["mini_batches"]
    ppo_config["learning_epochs"] = agent_cfg["learning_epochs"]
    ppo_config["learning_rate"] = agent_cfg["learning_rate"]
    
    if agent_cfg.get("learning_rate_scheduler"):
        if agent_cfg["learning_rate_scheduler"] == "KLAdaptiveLR":
            ppo_config["learning_rate_scheduler"] = KLAdaptiveLR
            ppo_config["learning_rate_scheduler_kwargs"] = agent_cfg.get(
                "learning_rate_scheduler_kwargs", {"kl_threshold": 0.008}
            )
    
    # PPO特定参数
    ppo_config["ratio_clip"] = agent_cfg["ratio_clip"]
    ppo_config["value_clip"] = agent_cfg["value_clip"]
    ppo_config["clip_predicted_values"] = agent_cfg["clip_predicted_values"]
    ppo_config["entropy_loss_scale"] = agent_cfg["entropy_loss_scale"]
    ppo_config["value_loss_scale"] = agent_cfg["value_loss_scale"]
    ppo_config["kl_threshold"] = agent_cfg.get("kl_threshold", 0.008)
    
    # 训练参数
    ppo_config["discount_factor"] = agent_cfg["discount_factor"]
    ppo_config["lambda"] = agent_cfg["lambda"]
    ppo_config["random_timesteps"] = agent_cfg.get("random_timesteps", 0)
    ppo_config["learning_starts"] = agent_cfg.get("learning_starts", 0)
    ppo_config["grad_norm_clip"] = agent_cfg["grad_norm_clip"]
    
    # 经验回放
    ppo_config["memory_size"] = agent_cfg["memory_size"]
    
    # 实验和保存
    ppo_config["experiment"]["write_interval"] = experiment_cfg["write_interval"]
    ppo_config["experiment"]["checkpoint_interval"] = experiment_cfg["checkpoint_interval"]
    
    return ppo_config


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Environment with SKRL PPO")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="learning/skrl/skrl_config.yaml",
                       help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (overrides config)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # 可以覆盖配置文件的关键参数
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (overrides config)")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config)")
    
    # Traffic specific overrides
    parser.add_argument("--drones_num", type=int, default=None, help="Number of drones (overrides config)")
    parser.add_argument("--evtols_num", type=int, default=None, help="Number of evtols (overrides config)")
    
    # Logging overrides
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")

    # Isaac Lab args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 加载配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    
    # 使用命令行参数覆盖配置
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    if args.num_envs:
        config["environment"]["num_envs"] = args.num_envs
    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.learning_rate:
        config["agent"]["learning_rate"] = args.learning_rate
    if args.drones_num is not None:
        config["traffic"]["drones_num"] = args.drones_num
    if args.evtols_num is not None:
        config["traffic"]["evtols_num"] = args.evtols_num
    
    # Wandb覆盖
    if args.use_wandb:
        config["logging"]["use_wandb"] = True
    if args.wandb_project:
        config["logging"]["wandb_project"] = args.wandb_project
    if args.wandb_entity:
        config["logging"]["wandb_entity"] = args.wandb_entity
    if args.wandb_run_name:
        config["logging"]["wandb_run_name"] = args.wandb_run_name
    
    # 设置headless模式
    args.headless = config["isaac_lab"]["headless"]
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 设置实验名称
    if config["experiment"]["name"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment"]["name"] = f"traffic_skrl_ppo_{timestamp}"

    # 创建保存目录
    save_dir = f"{config['experiment']['directory']}/{config['experiment']['name']}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 导入环境配置
        from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
        
        # 创建环境配置
        cfg = TrafficEnvCfg()
        cfg.scene = replace(cfg.scene, num_envs=config["environment"]["num_envs"])
        
        # 从配置文件设置traffic参数
        traffic_cfg = config["traffic"]
        cfg.traffic_sim.num_drones = traffic_cfg["drones_num"]
        cfg.traffic_sim.num_evtols = traffic_cfg["evtols_num"]
        
        # 设置奖励参数
        if traffic_cfg["rew_success"] is not None:
            cfg.rew_success = traffic_cfg["rew_success"]
        if traffic_cfg["rew_collision"] is not None:
            cfg.rew_collision = traffic_cfg["rew_collision"]
        cfg.rew_drone_future_penalty = traffic_cfg["drone_future_penalty"]
        cfg.rew_evtol_future_penalty = traffic_cfg["evtol_future_penalty"]
        
        if traffic_cfg["drones_threshold_factor"] is not None:
            cfg.rew_drones_threshold_factor = traffic_cfg["drones_threshold_factor"]
        if traffic_cfg["drones_decay_factor"] is not None:
            cfg.rew_drones_decay_factor = traffic_cfg["drones_decay_factor"]
        if traffic_cfg["evtols_threshold_factor"] is not None:
            cfg.rew_evtols_threshold_factor = traffic_cfg["evtols_threshold_factor"]
        if traffic_cfg["evtols_decay_factor"] is not None:
            cfg.rew_evtols_decay_factor = traffic_cfg["evtols_decay_factor"]
        
        # 设置预测步数
        if "predict_steps" in traffic_cfg:
            cfg.predict_steps = traffic_cfg["predict_steps"]

        # 创建环境
        env = create_env(cfg, headless=config["isaac_lab"]["headless"])
        
        # 设置随机种子
        set_seed(config["seed"])
        
        # 创建智能体配置
        agent_config = create_agent_config(config)
        
        # 创建memory
        memory = RandomMemory(
            memory_size=config["agent"]["memory_size"], 
            num_envs=config["environment"]["num_envs"], 
            device=device
        )
        
        # 创建GAT policy网络模型
        models = {}
        
        # 根据动作空间类型选择合适的policy
        from gymnasium import spaces
        if isinstance(env.action_space, spaces.Box):
            models["policy"] = GraphAttentionPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                cfg=config
            )
        elif isinstance(env.action_space, spaces.Discrete):
            models["policy"] = GraphAttentionPolicyDiscrete(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                cfg=config
            )
        else:
            raise ValueError(f"Unsupported action space: {env.action_space}")
            
        models["value"] = models["policy"]  # 共享网络
        
        # 创建PPO智能体
        agent = PPO(
            models=models,
            memory=memory,
            cfg=agent_config,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )
        
        # 设置训练器配置
        trainer_cfg = {
            "timesteps": config["training"]["total_timesteps"],
            "headless": config["isaac_lab"]["headless"],
        }
        
        # 创建回调函数
        callbacks = []
        
        # Success Rate Callback
        if config["callbacks"]["success_rate"]["enabled"]:
            success_callback = SuccessRateCallback(
                check_freq=config["callbacks"]["success_rate"]["check_freq"],
                save_path=os.path.join(save_dir, 'checkpoints'),
                queue_size=config["callbacks"]["success_rate"]["queue_size"],
                name_prefix='best_sr',
                success_threshold=config["callbacks"]["success_rate"]["success_threshold"]
            )
            callbacks.append(success_callback)
        
        # Episode Stats Callback
        if config["callbacks"]["episode_stats"]["enabled"]:
            episode_stats_callback = EpisodeStatsCallback(
                log_interval=config["callbacks"]["episode_stats"]["log_interval"]
            )
            callbacks.append(episode_stats_callback)
        
        # Model Checkpoint Callback
        if config["callbacks"]["model_checkpoint"]["enabled"]:
            checkpoint_callback = ModelCheckpointCallback(
                save_interval=config["callbacks"]["model_checkpoint"]["save_interval"],
                save_path=os.path.join(save_dir, 'checkpoints'),
                name_prefix='model_checkpoint'
            )
            callbacks.append(checkpoint_callback)
        
        # TensorBoard Logger
        if config["logging"]["use_tensorboard"]:
            tb_log_dir = os.path.join(save_dir, config["logging"]["tensorboard_dir"])
            tb_logger = TensorboardLogger(log_dir=tb_log_dir)
            callbacks.append(tb_logger)
        
        # Wandb Logger (如果启用)
        if config["logging"]["use_wandb"] and WANDB_AVAILABLE:
            wandb_config_dict = {
                "num_envs": config["environment"]["num_envs"],
                "experiment_name": config["experiment"]["name"],
                "traffic_drones": cfg.traffic_sim.num_drones,
                "traffic_evtols": cfg.traffic_sim.num_evtols,
                "learning_rate": config["agent"]["learning_rate"],
                "discount_factor": config["agent"]["discount_factor"],
                "entropy_loss_scale": config["agent"]["entropy_loss_scale"],
                "learning_epochs": config["agent"]["learning_epochs"],
                "rollouts": config["agent"]["rollouts"],
            }
            wandb_config_dict.update(config)  # 包含完整配置
            
            # 确定run name
            run_name = config["logging"].get("wandb_run_name")
            if not run_name:
                run_name = f"traffic_skrl_ppo_{config['experiment']['name']}"
            
            wandb_logger = WandbLogger(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity"),
                name=run_name,
                config=wandb_config_dict,
                dir=save_dir,
                tags=config["logging"].get("wandb_tags", [])
            )
            callbacks.append(wandb_logger)
            print(f"Wandb initialized for project: {config['logging']['wandb_project']}")
        elif config["logging"]["use_wandb"] and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without wandb logging.")

        # 创建训练器
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        
        # 设置回调函数
        for callback in callbacks:
            trainer.add_callback(callback)
        
        # 打印训练信息
        print(f"开始SKRL训练:")
        print(f"- 实验名称: {config['experiment']['name']}")
        print(f"- 保存目录: {save_dir}")
        print(f"- 总时间步数: {config['training']['total_timesteps']}")
        print(f"- 环境数量: {config['environment']['num_envs']}")
        print(f"- 学习率: {config['agent']['learning_rate']}")
        print(f"- 设备: {device}")
        print(f"- 无人机数量: {cfg.traffic_sim.num_drones}")
        print(f"- 电动飞行器数量: {cfg.traffic_sim.num_evtols}")
        print(f"- 配置文件: {args.config}")
        
        # 保存完整配置到实验目录
        config_save_path = os.path.join(save_dir, "config.yaml")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"- 配置已保存到: {config_save_path}")
        
        # 开始训练
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"训练完成！总用时: {training_time:.2f}秒")
        
        # 保存最终模型
        model_path = os.path.join(save_dir, "skrl_traffic_final")
        agent.save(model_path)
        print(f"最终模型已保存到: {model_path}")
        
        # 完成wandb run
        if config["logging"]["use_wandb"] and WANDB_AVAILABLE:
            try:
                import wandb
                wandb.finish()
                print("Wandb run completed and synced")
            except:
                pass
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
