#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Test

"""Test script for the Forest environment implementations."""

import argparse
import torch

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
from dataclasses import replace
headless = True
def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Test Forest Environment")
    parser.add_argument("--workflow", type=str, default="direct", choices=["direct", "manager"], 
                       help="The workflow to use: 'direct' or 'manager'")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--traffic", type=bool, default=True, help="Whether to use traffic")
    
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = headless

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入环境（在AppLauncher之后）
    if args_cli.workflow == "direct":
        if args_cli.traffic:
            from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvCfg
            cfg = TrafficEnvCfg()
            cfg.scene = replace(cfg.scene, num_envs=args_cli.num_envs)
            cfg.num_actions = 2
            cfg.num_observations = 7
            cfg.traffic_sim.num_drones = 10
            cfg.traffic_sim.num_evtols = 1
            cfg.action_space_type = "discrete"
            cfg.action_mode = "velocity_components"
            print(f"动作维度: {cfg.num_actions}, 观测维度: {cfg.num_observations}")
            env = TrafficEnv(cfg=cfg)
        else:
            from isaac_lab_envs.direct.nav_env import NavEnv, NavEnvCfg
            
            # 创建配置
            cfg = NavEnvCfg()
            # cfg.scene.num_envs = args_cli.num_envs
            cfg.scene = replace(cfg.scene, num_envs=args_cli.num_envs)
            cfg.num_actions = 2
            cfg.num_observations = 7

            print(f"动作维度: {cfg.num_actions}, 观测维度: {cfg.num_observations}")
            
            # 创建环境
            env = NavEnv(cfg=cfg)
    
    print(f"环境创建成功！")

    # 运行简单测试
    print("\n开始环境测试...")
    
    # 重置环境
    obs, _ = env.reset()
    
    # 运行几个步骤
    import time
    start_time = time.time()
    for step in range(2000):
        # 随机动作

        if cfg.action_space_type == "discrete":
            actions = torch.randint(0, env.cfg.action_space_num_per_dim * env.cfg.action_space_num_per_dim, (env.num_envs,), device=env.device)
        else:
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
            actions = actions.clamp(-1.0, 1.0)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # 打印信息
        if step % 100 == 0:
            print(f"步骤 {step:2d}: 平均奖励 = {reward.mean():.4f}, "
                  f"终止 = {terminated.sum().item():2d}, "
                  f"截断 = {truncated.sum().item():2d}")
            print(f"帧率: {100 / (time.time() - start_time):.2f}帧/秒")
            start_time = time.time()
    
    print("\n测试完成！")
    
    # 如果是GUI模式，保持窗口打开
    if not args_cli.headless:
        print("GUI模式：按Ctrl+C退出...")
        try:
            while True:
                # 继续运行环境
                if env.cfg.use_discrete_action:
                    actions = torch.randint(0, env.cfg.action_space_num_per_dim * env.cfg.action_space_num_per_dim, (env.num_envs,), device=env.device)
                else:
                    actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
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