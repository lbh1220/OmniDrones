#!/usr/bin/env python3

"""Test script for trained SB3 models
加载训练好的模型并进行可视化测试，支持视频录制保存
"""

import argparse
import os
import time
import numpy as np
from dataclasses import replace

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
# 移除evaluation导入，Isaac Lab SB3包装器不支持evaluate_policy

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher




def create_test_env(cfg, headless=False):
    """创建测试环境（可视化模式）"""

    from isaac_lab_envs.direct.nav_env import NavEnv
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    
    # 创建环境
    env = NavEnv(cfg=cfg)
    
    # 使用SB3包装器包装
    env = Sb3VecEnvWrapper(env)
    
    return env


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Test trained SB3 model")
    parser.add_argument("--model_path", type=str, 
                        default="runs/nav_ppo_firefly_20250827_175138",
                       help="Path to the trained model directory")
    parser.add_argument("--num_envs", type=int, default=16,
                       help="Number of test environments")
    parser.add_argument("--drone_model", type=str, default="firefly",
                       choices=["firefly", "crazyflie", "hummingbird", "iris"],
                       help="Drone model to use")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy")
    parser.add_argument("--render_mode", type=str, default="human",
                       choices=["human", "rgb_array"],
                       help="Render mode")
    parser.add_argument("--episode_length", type=int, default=1000,
                       help="Maximum episode length for manual testing")
    
    
    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置为非headless模式以便可视化
    args.headless = True
    # args.off
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    try:
        # 导入环境配置（必须在AppLauncher之后）
        from isaac_lab_envs.direct.nav_env import NavEnvCfg
        
        # 检查模型路径
        model_file = os.path.join(args.model_path, "final_model.zip")
        vecnormalize_file = os.path.join(args.model_path, "final_model_vecnormalize.pkl")
        
        if not os.path.exists(model_file):
            print(f"错误：找不到模型文件 {model_file}")
            return
        
        print(f"加载模型: {model_file}")
        if os.path.exists(vecnormalize_file):
            print(f"加载归一化参数: {vecnormalize_file}")
        
        # 设置视频保存目录
        if args.video_dir is None:
            args.video_dir = os.path.join(args.model_path, "test_videos")
        os.makedirs(args.video_dir, exist_ok=True)
        
        # 创建环境配置
        cfg = NavEnvCfg()
        cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
        cfg.drone_model = args.drone_model
        cfg.num_actions = 2  # vx, vy
        cfg.num_observations = 7  # robot_node(5) + temporal_edges(2)
        
        print(f"创建测试环境...")
        print(f"- 无人机模型: {args.drone_model}")
        print(f"- 环境数量: {args.num_envs}")
        print(f"- 动作维度: {cfg.num_actions}")
        print(f"- 观测维度: {cfg.num_observations}")
        print(f"- 渲染模式: {args.render_mode}")

        
        # 创建测试环境
        env = create_test_env(cfg, headless=False)
        
        # 如果存在VecNormalize文件，加载归一化参数
        if os.path.exists(vecnormalize_file):
            env = VecNormalize.load(vecnormalize_file, env)
            # 测试时不更新归一化统计
            env.training = False
            env.norm_reward = False
        
        # 加载训练好的模型
        model = PPO.load(model_file, env=env)
        print(f"模型加载成功！")
        print(f"- 模型设备: {model.device}")
        print(f"- 策略类型: {type(model.policy)}")

        # 原有的交互测试逻辑（不录制视频）
        print(f"\n开始手动交互测试...")
        print(f"- 最大步数: {args.episode_length}")
        print(f"- 按 Ctrl+C 退出")
        
        episode_count = 0
        try:
            while episode_count < args.n_eval_episodes:
                episode_count += 1
                print(f"\n=== 第 {episode_count} 回合 ===")
                
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(args.episode_length):
                    # 预测动作
                    action, _states = model.predict(obs, deterministic=args.deterministic)
                    
                    # 执行动作
                    obs, rewards, dones, infos = env.step(action)
                    
                    episode_reward += rewards.mean()
                    episode_length += 1
                    
                    # 打印信息
                    if step % 50 == 0:
                        print(f"  步骤 {step:3d}: 平均奖励 = {rewards.mean():.4f}, "
                                f"完成环境数 = {dones.sum()}")
                    
                    # # 检查是否有环境完成
                    # if dones.any():
                    #     print(f"  有 {dones.sum()} 个环境完成了回合")
                    #     break
                
                print(f"回合 {episode_count} 完成:")
                print(f"  - 总奖励: {episode_reward:.4f}")
                print(f"  - 回合长度: {episode_length}")
        except KeyboardInterrupt:
            print("\n用户中断测试")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
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