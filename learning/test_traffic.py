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
from stable_baselines3.common.vec_env import VecNormalize
# 移除evaluation导入，Isaac Lab SB3包装器不支持evaluate_policy

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher




def create_test_env(cfg, headless=True):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvWithCurriculum
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    # 创建环境
    if cfg.curriculum_learning:
        env = TrafficEnvWithCurriculum(cfg=cfg)
    else:
        env = TrafficEnv(cfg=cfg)
    
    # 使用SB3包装器包装
    env = Sb3VecEnvWrapper(env)
    
    return env


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Test trained SB3 model")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of environments")
    parser.add_argument("--model_dir", type=str, 
                        default="runs/traffic/homoe/u0e2_c-50.0_f3.0/checkpoints",
                       help="Path to the trained model directory")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of episodes for evaluation")
    

    # add args, drones_num and evtols_num, drone_future_penalty and evtol_future_penalty
    parser.add_argument("--drones_num", type=int, default=0, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=2, help="Number of evtols")
  
    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置为非headless模式以便可视化
    args.headless = True
    # args.off
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app


    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    

        # 导入环境配置（必须在AppLauncher之后）
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # 检查模型路径
    model_file = os.path.join(args.model_dir, "SR_11730944_steps_.zip")
    vecnormalize_file = os.path.join(args.model_dir, "SR_11730944_steps_steps.pkl")
    
    if not os.path.exists(model_file):
        print(f"错误：找不到模型文件 {model_file}")
        return
    
    print(f"加载模型: {model_file}")
    if os.path.exists(vecnormalize_file):
        print(f"加载归一化参数: {vecnormalize_file}")
    


    
    # 创建环境配置
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num

    from isaac_lab_envs.direct.traffic_env import TrafficCurriculumCfg
    course_list = [
                TrafficCurriculumCfg(drones_num=0, evtol_num=1),
                TrafficCurriculumCfg(drones_num=0, evtol_num=2),
            ]
    args.course_num = len(course_list)
    cfg.curriculum_list = course_list
    cfg.curriculum_learning = True
    cfg.traffic_sim.num_drones = course_list[-1].drones_num
    cfg.traffic_sim.num_evtols = course_list[-1].evtol_num




    cfg.debug_vis = True




    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) 
    algo_args.human_human_edge_input_size = algo_args.human_human_edge_input_size + 1
      

    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(100, 0., 100),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    print(f"创建测试环境...")

    
    # 创建测试环境
    env = create_test_env(cfg, headless=False)
    
    # 如果存在VecNormalize文件，加载归一化参数
    if os.path.exists(vecnormalize_file):
        env = VecNormalize.load(vecnormalize_file, env)
        # 测试时不更新归一化统计
        env.training = False
        env.norm_reward = False
    
    # 加载训练好的模型

    model = CustomPPO.load(model_file, env=env, args=algo_args)


    output_dir = os.path.join(args.model_dir, 'test_results', time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
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


    while episode_count < args.num_episodes:
        action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, info = env.step(action)

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