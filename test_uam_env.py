#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Test

"""Test script for the Forest environment implementations."""

import argparse
import torch
import gymnasium as gym
import os
# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
from dataclasses import replace
headless = False
def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Test Forest Environment")
    parser.add_argument("--workflow", type=str, default="direct", choices=["direct", "manager"], 
                       help="The workflow to use: 'direct' or 'manager'")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--traffic", type=bool, default=True, help="Whether to use traffic")
    parser.add_argument("--map", type=bool, default=True, help="Whether to use map")
    parser.add_argument("--video", type=bool, default=False, help="Whether to use video")
    parser.add_argument("--video_interval", type=int, default=10000, help="Video interval")
    parser.add_argument("--video_length", type=int, default=500, help="Video length")

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = headless
    args_cli.enable_cameras = True

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入环境（在AppLauncher之后）
    from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg, CityUamEnvCfg, OpenAirEnvCfg, DyanmicUamEnvCfg
    # if args_cli.map:
    #     cfg = CityUamEnvCfg()
    # else:
    #     cfg = UamEnvCfg()
    cfg = CityUamEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args_cli.num_envs)
    cfg.num_actions = 2
    cfg.num_observations = 7
    cfg.action_manager.action_space_type = 'discrete'
    print(f"动作维度: {cfg.num_actions}, 观测维度: {cfg.num_observations}")
    
    cfg.use_global_path = True
    from isaac_lab_envs.direct.mdp.observations import CityNavObservationProcessorWithPath
    cfg.observation_processor_cls = CityNavObservationProcessorWithPath
    from isaac_lab_envs.direct.mdp.rewards import CityNavRewardCalculatorWithPath
    cfg.reward_calculator_cls = CityNavRewardCalculatorWithPath
            

    cfg.debug_vis = True
    from omni.isaac.lab.envs.common import ViewerCfg
    bounds = cfg.area_bounds
    range_x = bounds.xmax - bounds.xmin
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(0, 0.0, range_x*1.5),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    from isaac_lab_envs.direct.uam_env import UamEnv
    env = UamEnv(cfg=cfg, render_mode="rgb_array")
    print(f"环境创建成功！")
    save_dir = "runs/test_uam_env"
    os.makedirs(save_dir, exist_ok=True)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # 运行简单测试
    print("\n开始环境测试...")
    
    # 重置环境
    obs, _ = env.reset()
    
    # 运行几个步骤
    import time
    start_time = time.time()
    for step in range(2000):
        # 随机动作
        action_space_num_per_dim = env.cfg.action_manager.action_space_num_per_dim
        action_space_type = env.cfg.action_manager.action_space_type
        if action_space_type == "discrete":
            actions = torch.randint(0, action_space_num_per_dim * action_space_num_per_dim, (env.num_envs,), device=env.device)
            # actions = torch.full((env.num_envs,), 0, device=env.device)
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
                if env.cfg.action_space_type == "discrete":
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