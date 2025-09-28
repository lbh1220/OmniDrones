#!/usr/bin/env python3

"""Stable-Baselines3 training script for Nav Environment
使用PPO算法训练Direct RL导航环境，支持定期视频录制
"""

import argparse
import os
import time
from datetime import datetime
from dataclasses import replace
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.logger import configure
# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
from rl.sb3.custom_callback import SucessRateCallback
from rl.sb3.network_utils import linear_schedule_with_min

class SimpleFeatureExtractor(BaseFeaturesExtractor):
    """
    简单的特征提取器，将robot_node和temporal_edges连接成向量
    只使用最基本的特征，忽略spatial_edges等复杂信息
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        # 计算输入维度
        robot_node_dim = observation_space.spaces['robot_node'].shape[1]  # 7 或 5
        temporal_edges_dim = observation_space.spaces['temporal_edges'].shape[1]  # 2

        one_spatial_edge_dim = observation_space.spaces['spatial_edges'].shape[1]
        
        self.input_dim = robot_node_dim + temporal_edges_dim + one_spatial_edge_dim
        
        # 简单的MLP网络
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, features_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, observations):
        # 提取robot_node和temporal_edges
        robot_node = observations['robot_node'].squeeze(1)  # 从(1, 7)变为(7,)
        temporal_edges = observations['temporal_edges'].squeeze(1)  # 从(1, 2)变为(2,)
        if len(observations['spatial_edges'].shape) == 3:
            one_spatial_edge = observations['spatial_edges'][:, 0, :]
        else:
            one_spatial_edge = observations['spatial_edges'][0, :]
        # 连接特征
        combined_features = torch.cat([robot_node, temporal_edges, one_spatial_edge], dim=-1)
        
        # 通过MLP
        return self.mlp(combined_features)

def create_env(cfg, headless=True):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.nav_env import NavEnv
    from isaac_lab_envs.direct.traffic_env import TrafficEnv
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    # 创建环境
    env = NavEnv(cfg=cfg)
    

    
    # 使用SB3包装器包装
    env = Sb3VecEnvWrapper(env)
    
    return env


def create_normalized_env(cfg, headless=True):
    """创建带归一化的环境"""
    env = create_env(cfg, headless)
    
    # 添加归一化（可选，但通常有助于训练稳定性）
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    return env


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Train Nav Environment with SB3 PPO")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
    parser.add_argument("--drone_model", type=str, default="hummingbird", 
                       choices=["firefly", "crazyflie", "hummingbird", "iris"], 
                       help="Drone model to use")
    parser.add_argument("--total_timesteps", type=int, default=5000000, 
                       help="Total timesteps for training")
    parser.add_argument("--save_freq", type=int, default=50000, 
                       help="Save model every N timesteps")
    parser.add_argument("--feature_dim", type=int, default=256,
                       help="Feature dimension")

    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                       help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for saving")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_drones", type=int, default=1,
                       help="Number of drones")
    parser.add_argument("--drone_future_penalty", type=float, default=0.0,
                       help="Drone future penalty")
    
    
    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置headless模式
    args.headless = True  # 强制使用headless模式进行训练
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    try:
        # 导入环境配置（必须在AppLauncher之后）
        from isaac_lab_envs.direct.nav_env import NavEnvCfg
        from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
        # 创建环境配置
        cfg = TrafficEnvCfg()
        cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
        cfg.drone_model = args.drone_model
        cfg.num_actions = 2  # vx, vy
        cfg.num_observations = 7  # robot_node(5) + temporal_edges(2)
        cfg.traffic_sim.num_drones = args.num_drones    
        cfg.traffic_sim.num_evtols = 0
        cfg.rew_evtol_future_penalty = 0.0  
        cfg.rew_drone_future_penalty = args.drone_future_penalty
        # cfg.pred_timestep = 0
        cfg.use_discrete_action = True

        print(f"创建训练环境...")
        print(f"- 无人机模型: {args.drone_model}")
        print(f"- 环境数量: {args.num_envs}")
        print(f"- 动作维度: {cfg.num_actions}")
        print(f"- 观测维度: {cfg.num_observations}")
        
        # 创建训练环境
        env = create_normalized_env(cfg, headless=True)
        
        # 验证观测空间格式
        print("=== 观测空间信息 ===")
        print(f"观测空间: {env.observation_space}")
        
        
        # 设置实验名称
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"nav_ppo_{args.drone_model}_{timestamp}"
        
        # 创建保存目录
        save_dir = f"runs/simple/{args.experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"实验名称: {args.experiment_name}")
        print(f"保存目录: {save_dir}")
        
        # 配置PPO策略
        policy_kwargs = dict(
            features_extractor_class=SimpleFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=args.feature_dim),
            net_arch=[128, 128],  # Actor和Critic的网络架构
            activation_fn=nn.ReLU,
        )
        n_steps = 100
        batch_size = n_steps * args.num_envs // 16
        learning_rate = linear_schedule_with_min(args.learning_rate, 1e-6)
        # 创建PPO模型
        model = PPO(
            policy="MultiInputPolicy",  # 支持字典观测空间
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,  # 每次更新收集的步数
            batch_size=batch_size,  # 批次大小
            n_epochs=10,    # 每次更新的epoch数
            gamma=0.99,     # 折扣因子
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.15,   # PPO裁剪范围
            ent_coef=0.01,    # 熵系数
            vf_coef=0.5,      # 价值函数系数
            max_grad_norm=0.5,  # 梯度裁剪
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=f"{save_dir}/tensorboard/",
        )
        
        print(f"PPO模型创建完成！")
        print(f"- 学习率: {args.learning_rate}")
        print(f"- 设备: {model.device}")
        
        # 创建回调函数
        callbacks = []
        SR_check_callback = SucessRateCallback(check_freq=2,
                                                save_path=os.path.join(save_dir, 'checkpoints'),
                                                name_prefix='SR')
        callbacks.append(SR_check_callback)
        
        # 检查点保存回调
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=f"{save_dir}/checkpoints/",
            name_prefix="nav_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        
        print(f"开始训练...")
        print(f"- 总时间步: {args.total_timesteps}")
        print(f"- 保存频率: {args.save_freq}")

        new_logger = configure(os.path.join(save_dir, 'logs'), ["stdout","tensorboard", "log"])
        model.set_logger(new_logger)

        # 开始训练
        start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="nav_ppo",
            reset_num_timesteps=True,
        )
        
        training_time = time.time() - start_time
        print(f"训练完成！总用时: {training_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = f"{save_dir}/final_model"
        model.save(final_model_path)
        env.save(f"{final_model_path}_vecnormalize.pkl")
        
        print(f"最终模型已保存到: {final_model_path}")
        
        
        print("训练和测试完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 关闭环境和仿真
        try:
            env.close()
        except:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()