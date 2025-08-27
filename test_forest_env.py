#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Test

"""Test script for the Forest environment implementations."""

import argparse
import torch

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher

def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Test Forest Environment")
    parser.add_argument("--workflow", type=str, default="direct", choices=["direct", "manager"], 
                       help="The workflow to use: 'direct' or 'manager'")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--drone_model", type=str, default="firefly", 
                       choices=["firefly", "crazyflie", "hummingbird", "iris"], 
                       help="Drone model to use")
    # parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
    
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = False

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入环境（在AppLauncher之后）
    if args_cli.workflow == "direct":
        from isaac_lab_envs.direct.forest_env import ForestEnv, ForestEnvCfg
        
        # 创建配置
        cfg = ForestEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.drone_model = args_cli.drone_model
        cfg.num_actions = 2
        # 根据drone_model调整num_actions和num_observations
        if args_cli.drone_model == "firefly":
            # cfg.num_actions = 6  # firefly 有6个旋翼
            cfg.num_observations = 23 + 6 + (36 * 4)  # drone_state + lidar_scan
        elif args_cli.drone_model in ["crazyflie", "hummingbird", "iris"]:
            # cfg.num_actions = 4  # 这些无人机有4个旋翼
            cfg.num_observations = 19 + 4 + (36 * 4)  # drone_state + lidar_scan
        
        print(f"创建Direct RL Forest环境，使用{args_cli.drone_model}无人机")
        print(f"动作维度: {cfg.num_actions}, 观测维度: {cfg.num_observations}")
        
        # 创建环境
        env = ForestEnv(cfg=cfg)
        
    elif args_cli.workflow == "manager":
        from isaac_lab_envs.manager_based.forest_env_cfg import ForestManagerEnvCfg
        from omni.isaac.lab.envs import ManagerBasedRLEnv
        
        # 创建配置
        cfg = ForestManagerEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        
        print(f"创建Manager-based RL Forest环境，环境数量: {args_cli.num_envs}")
        
        # 创建环境
        env = ManagerBasedRLEnv(cfg=cfg)
    
    print(f"环境创建成功！")
    print(f"观测空间: {env.observation_manager.group_obs_dim if hasattr(env, 'observation_manager') else 'N/A'}")
    print(f"动作空间: {env.action_manager.total_action_dim if hasattr(env, 'action_manager') else env.num_actions}")
    print(f"环境数量: {env.num_envs}")

    # 运行简单测试
    print("\n开始环境测试...")
    
    # 重置环境
    obs, _ = env.reset()
    print(f"重置完成！观测形状: {[obs[key].shape for key in obs.keys()]}")
    
    # 运行几个步骤
    for step in range(10):
        # 随机动作
        if args_cli.workflow == "direct":
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        actions = actions.clamp(-1.0, 1.0)  # 限制到有效范围
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # 打印信息
        if step % 5 == 0:
            print(f"步骤 {step:2d}: 平均奖励 = {reward.mean():.4f}, "
                  f"终止 = {terminated.sum().item():2d}, "
                  f"截断 = {truncated.sum().item():2d}")
    
    print("\n测试完成！")
    
    # 如果是GUI模式，保持窗口打开
    if not args_cli.headless:
        print("GUI模式：按Ctrl+C退出...")
        try:
            while True:
                # 继续运行环境
                actions = torch.randn(env.num_envs, env.num_actions if args_cli.workflow == "direct" else env.action_manager.total_action_dim, device=env.device)
                actions = actions.clamp(-1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(actions)
                
                # 检查是否有环境需要重置
                reset_mask = terminated | truncated
                if reset_mask.any():
                    env.reset()
                    
        except KeyboardInterrupt:
            print("收到退出信号")
    
    # 关闭环境
    env.close()
    
    # 关闭仿真
    simulation_app.close()


if __name__ == "__main__":
    main() 