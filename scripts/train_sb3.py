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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher



class VideoRecordingCallback(BaseCallback):
    """Custom callback for video recording during training."""
    
    def __init__(
        self,
        eval_env,
        save_dir: str,
        record_freq: int = 50000,
        n_eval_episodes: int = 3,
        deterministic: bool = True,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.record_freq = record_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.last_record_step = 0
        
        # 创建视频保存目录
        self.video_dir = os.path.join(save_dir, "training_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.recorder = DirectRLVideoRecorder(
            save_dir=self.video_dir,
            fps=30.0,
            resolution=(960, 720),
            interval=2  # 每2步录制一帧
        )
    
    def _on_step(self) -> bool:
        # 检查是否需要录制视频
        if self.n_calls - self.last_record_step >= self.record_freq:
            self._record_videos()
            self.last_record_step = self.n_calls
        return True
    
    def _record_videos(self):
        """Record videos using current model."""
        print(f"\n正在录制训练视频 (步骤 {self.n_calls})...")
        
        for episode in range(self.n_eval_episodes):
            self.recorder.reset_episode()
            
            obs = self.eval_env.reset()
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done and step_count < 500:  # 最大500步
                # 录制帧
                self.recorder.record_frame(self.eval_env)
                
                # 获取动作
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # 执行动作
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward.mean() if hasattr(reward, 'mean') else reward
                step_count += 1
                
                # 检查是否完成
                if hasattr(done, 'any') and done.any():
                    break
            
            # 保存视频
            video_filename = f"training_step_{self.n_calls}_episode_{episode}.mp4"
            saved_path = self.recorder.save_video(video_filename)
            
            if self.verbose > 0:
                print(f"  回合 {episode}: {step_count} 步, 奖励: {episode_reward:.4f}")
        
        print(f"训练视频录制完成 (步骤 {self.n_calls})")


class SimpleMLPExtractor(BaseFeaturesExtractor):
    """
    简单的MLP特征提取器，适用于7维观测向量
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 对于7维输入的简单MLP网络
        input_dim = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        return self.mlp(observations)


def create_env(cfg, headless=True):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.nav_env import NavEnv
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    # 创建环境
    env = NavEnv(cfg=cfg)
    
    # 设置环境的render_mode为rgb_array以支持视频录制
    if hasattr(env, 'render_mode'):
        env.render_mode = "rgb_array"
    
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
    parser.add_argument("--drone_model", type=str, default="firefly", 
                       choices=["firefly", "crazyflie", "hummingbird", "iris"], 
                       help="Drone model to use")
    parser.add_argument("--total_timesteps", type=int, default=1000000, 
                       help="Total timesteps for training")
    parser.add_argument("--save_freq", type=int, default=50000, 
                       help="Save model every N timesteps")

    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                       help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for saving")
    
    
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
        
        # 创建环境配置
        cfg = NavEnvCfg()
        cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
        cfg.drone_model = args.drone_model
        cfg.num_actions = 2  # vx, vy
        cfg.num_observations = 7  # robot_node(5) + temporal_edges(2)
        
        print(f"创建训练环境...")
        print(f"- 无人机模型: {args.drone_model}")
        print(f"- 环境数量: {args.num_envs}")
        print(f"- 动作维度: {cfg.num_actions}")
        print(f"- 观测维度: {cfg.num_observations}")
        
        # 创建训练环境
        env = create_normalized_env(cfg, headless=True)
        
        
        # 设置实验名称
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"nav_ppo_{args.drone_model}_{timestamp}"
        
        # 创建保存目录
        save_dir = f"runs/{args.experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"实验名称: {args.experiment_name}")
        print(f"保存目录: {save_dir}")
        
        # 配置PPO策略
        policy_kwargs = dict(
            features_extractor_class=SimpleMLPExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[128, 128],  # Actor和Critic的网络架构
            activation_fn=nn.ReLU,
        )
        
        # 创建PPO模型
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=100,  # 每次更新收集的步数
            batch_size=64,  # 批次大小
            n_epochs=10,    # 每次更新的epoch数
            gamma=0.99,     # 折扣因子
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,   # PPO裁剪范围
            ent_coef=0.01,    # 熵系数
            vf_coef=0.5,      # 价值函数系数
            max_grad_norm=0.5,  # 梯度裁剪
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=f"{save_dir}/tensorboard/",
        )
        
        print(f"PPO模型创建完成！")
        print(f"- 学习率: {args.learning_rate}")
        print(f"- 设备: {model.device}")
        
        # 创建回调函数
        callbacks = []
        
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
        
        
        # 运行一些测试步骤
        print("\n运行测试...")
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if i % 20 == 0:
                mean_reward = rewards.mean()
                print(f"测试步骤 {i}: 平均奖励 = {mean_reward:.4f}")
        
        print("训练和测试完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 关闭环境和仿真
        try:
            env.close()
            if eval_env is not None:
                eval_env.close()
        except:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()