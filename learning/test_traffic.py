#!/usr/bin/env python3

"""Test script for trained SB3 models
加载训练好的模型并进行可视化测试，支持视频录制保存
"""

import argparse
import os
import time
import numpy as np
from dataclasses import replace
from rl.sb3.config import ArgsConfig
import torch
from stable_baselines3.common.logger import configure
from rl.sb3.custom_ppo import CustomPPO
from rl.sb3.vec_normalize import VecNormalize
# 移除evaluation导入，Isaac Lab SB3包装器不支持evaluate_policy

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher

import gymnasium as gym


def create_test_env(cfg, args=None):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvWithCurriculum

    # 创建环境
    if cfg.curriculum_learning:
        env = TrafficEnvWithCurriculum(cfg=cfg, render_mode="rgb_array" if args.video else None)
    else:
        env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)

    
    return env


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Test trained SB3 model")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of environments")
    parser.add_argument("--model_dir", type=str, 
                        default="runs/traffic/path/u10/mult_test/1_0926_003558",
                       help="Path to the trained model directory")
    parser.add_argument("--model_name", type=str, default="final_model.zip", help="Model name")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for evaluation")
    

    # add args, drones_num and evtols_num, drone_future_penalty and evtol_future_penalty
    parser.add_argument("--drones_num", type=int, default=10, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=0, help="Number of evtols")
    parser.add_argument("--use_global_path", action="store_true", default=False, help="Use global path")
    parser.add_argument("--video", action="store_true", default=True, help="Record video")
    parser.add_argument("--video_interval", type=int, default=1000, help="Video interval")
    parser.add_argument("--video_length", type=int, default=500, help="Video length")
    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置为非headless模式以便可视化
    args.headless = True
    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False
    # args.off
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app


    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    

        # 导入环境配置（必须在AppLauncher之后）
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # 检查模型路径
    model_file = os.path.join(args.model_dir, args.model_name)
    
    
    if not os.path.exists(model_file):
        model_file = os.path.join(args.model_dir, 'checkpoints', args.model_name)
        if not os.path.exists(model_file):
            print(f"错误：找不到模型文件 {model_file}")
            return
    
    print(f"加载模型: {model_file}")
    vecnormalize_file = model_file.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vecnormalize_file):
        print(f"加载归一化参数: {vecnormalize_file}")
    


    
    # 创建环境配置
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    cfg.seed = algo_args.seed
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    cfg.use_global_path = args.use_global_path

    from isaac_lab_envs.direct.traffic_env import TrafficCurriculumCfg
    course_list = [
                TrafficCurriculumCfg(drones_num=2, evtol_num=0),
                TrafficCurriculumCfg(drones_num=2, evtol_num=0),
            ]
    args.course_num = len(course_list)
    cfg.curriculum_list = course_list
    cfg.curriculum_learning = False
    # cfg.traffic_sim.num_drones = course_list[-1].drones_num
    # cfg.traffic_sim.num_evtols = course_list[-1].evtol_num


    cfg.use_global_path = True




    cfg.debug_vis = True




    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) 
    algo_args.human_human_edge_input_size = algo_args.human_human_edge_input_size + 1
      

    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    print(f"创建测试环境...")
    output_dir = os.path.join(args.model_dir, 'test_results', args.model_name.replace(".zip", "_") + time.strftime("%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试环境
    env = create_test_env(cfg, args=args)
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(output_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during testing.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # 使用SB3包装器包装
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    env = Sb3VecEnvWrapper(env)
    # 如果存在VecNormalize文件，加载归一化参数
    if os.path.exists(vecnormalize_file):
        env = VecNormalize.load(vecnormalize_file, env)
        # 测试时不更新归一化统计
        env.training = False
        env.norm_reward = False

    algo_args.robot_node_input_size = env.observation_space['robot_node'].shape[1] + env.observation_space['temporal_edges'].shape[1]
    # 加载训练好的模型

    model = CustomPPO.load(model_file, env=env, args=algo_args)



    
    # 设置日志
    new_logger = configure(output_dir, ["stdout", "log"])
    model.set_logger(new_logger)
    new_logger.info(f"Starting evaluation for {args.num_episodes} episodes")
    episode_count = 0

    obs = env.reset()
    states = None
    episode_starts = np.ones((args.num_envs,), dtype=bool)
    episode_count = 0
    success_count = 0
    collision_count = 0
    episode_rewards = []
    episode_lengths = []
    base_env = model.get_env().unwrapped

    step_count = 0
    while episode_count < args.num_episodes:
        action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        if done.any():
            for i, done_ in enumerate(done):
                if done_:
                    episode_count += 1
                    ep_length = info[i]['episode']['l']
                    ep_reward = info[i]['episode']['r']
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    if info[i]['goal_reached']:
                        success_count += 1
                        new_logger.info(f'Episode {episode_count} Success in {ep_length} steps, reward={ep_reward:.4f}')
                    elif info[i]['collision']:
                        collision_count += 1
                        new_logger.info(f'Episode {episode_count} Collision in {ep_length} steps, reward={ep_reward:.4f}')


        episode_starts = done

    new_logger.info(f"Success rate: {success_count / args.num_episodes:.4f}")
    new_logger.info(f"Collision rate: {collision_count / args.num_episodes:.4f}")
    new_logger.info(f"Episode length: {np.mean(episode_lengths):.4f} +/- {np.std(episode_lengths):.4f}")
    new_logger.info(f"Episode reward: {np.mean(episode_rewards):.4f} +/- {np.std(episode_rewards):.4f}")

    env.close()

    simulation_app.close()


if __name__ == "__main__":
    main()