#!/usr/bin/env python3

"""Batch test script for multiple trained SB3 models
批量测试多个训练好的模型并进行性能对比分析
"""

import argparse
import os
import time
import numpy as np
import json
import glob
from dataclasses import replace
from typing import List, Dict, Any

from tensordict import base
from rl.sb3.config import ArgsConfig
import torch
from stable_baselines3.common.logger import configure
from rl.sb3.custom_ppo import CustomPPO
from rl.sb3.vec_normalize import VecNormalize
from dataclasses import dataclass, field

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
import gymnasium as gym

from test_traffic import create_test_env, get_latest_checkpoint_models
from rl.sb3.evaluate_policy import evaluate_policy



def wrap_env_for_model(base_env, algo_args, vecnorm_file=None, video_kwargs=None):
    """为特定模型包装环境（重用base_env）"""
    # 从base_env开始
    current_test_env = base_env
    
    # 如果需要视频录制，添加RecordVideo wrapper
    if video_kwargs:
        current_test_env = gym.wrappers.RecordVideo(current_test_env, **video_kwargs)
    
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    current_test_env = Sb3VecEnvWrapper(current_test_env)
    
    # 如果存在VecNormalize文件，加载归一化参数
    if vecnorm_file and os.path.exists(vecnorm_file):
        current_test_env = VecNormalize.load(vecnorm_file, current_test_env)
        # 测试时不更新归一化统计
        current_test_env.training = False
        current_test_env.norm_reward = False
    
    # 设置seed
    current_test_env.seed(seed=algo_args.seed)
    
    return current_test_env


def get_models_to_test(model_dir: str, model_list: List[str], include_checkpoints: bool = True) -> List[Dict[str, str]]:
    """获取要测试的模型列表"""
    models_to_test = []
    
    # 添加指定的模型
    for model_name in model_list:
        model_file = os.path.join(model_dir, model_name)
        if not os.path.exists(model_file):
            # 尝试在checkpoints目录中查找
            checkpoint_file = os.path.join(model_dir, 'checkpoints', model_name)
            if os.path.exists(checkpoint_file):
                model_file = checkpoint_file
            else:
                print(f"Warning: Model {model_name} not found in {model_dir} or checkpoints/")
                continue
        
        vecnorm_file = model_file.replace(".zip", "_vecnormalize.pkl")
        display_name = model_name.replace(".zip", "")
        
        models_to_test.append({
            "name": display_name,
            "model_file": model_file,
            "vecnorm_file": vecnorm_file if os.path.exists(vecnorm_file) else None
        })
    
    # 如果启用，添加最新的checkpoint模型
    if include_checkpoints:
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        latest_checkpoints = get_latest_checkpoint_models(checkpoints_dir, num_models=2)
        
        for checkpoint_path in latest_checkpoints:
            checkpoint_name = os.path.basename(checkpoint_path).replace('.zip', '')
            # 避免重复添加已经在列表中的模型
            if not any(model["name"] == checkpoint_name for model in models_to_test):
                vecnorm_path = checkpoint_path.replace('.zip', '_vecnormalize.pkl')
                models_to_test.append({
                    "name": f"latest_{checkpoint_name}",
                    "model_file": checkpoint_path,
                    "vecnorm_file": vecnorm_path if os.path.exists(vecnorm_path) else None
                })
    
    return models_to_test


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Batch test multiple trained SB3 models")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments")
    parser.add_argument("--model_dir", type=str, 
                        default="runs/traffic/ablation/u10e1/fu-2.0fe-2.0_1005_201144",
                       help="Path to the trained model directory")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes for evaluation")
    
    # 模型相关参数
    parser.add_argument("--drones_num", type=int, default=10, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=1, help="Number of evtols")
    parser.add_argument("--use_global_path", action="store_true", default=True, help="Use global path")
    parser.add_argument("--use_rnn", action="store_true", default=True, help="Use RNN-based recurrent policy")
    
    # 批量测试参数
    parser.add_argument("--include_checkpoints", action="store_true", default=True, 
                       help="Include latest checkpoint models in testing")
    parser.add_argument("--video", action="store_true", default=True, help="Record video (only for first model)")
    parser.add_argument("--video_interval", type=int, default=1000, help="Video interval")
    parser.add_argument("--video_length", type=int, default=500, help="Video length")
    
    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置为headless模式
    args.headless = True
    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 在这里定义要测试的模型列表 - 用户可以修改这个列表
    MODEL_LIST = [
        "final_model.zip",
        # "SR_26214400_steps.zip",
        # "SR_30801920_steps.zip",
        # "SR_36044800_steps.zip",
        # 可以添加更多模型文件名
        # "SR_30000000_steps.zip",
        # "SR_40000000_steps.zip",
    ]
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    checkpoint_list = get_latest_checkpoint_models(checkpoint_dir, num_models=3)
    for checkpoint in checkpoint_list:
        # checkpoint is runs/..../checkpoints/*.zip, should remove runs/..../checkpoints/
        checkpoint = checkpoint.replace(args.model_dir + '/checkpoints/', '')
        MODEL_LIST.append(os.path.basename(checkpoint))

    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    algo_args.use_rnn = args.use_rnn

    # 导入环境配置（必须在AppLauncher之后）
    from omni.isaac.lab.utils.io import load_yaml
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg, TrafficCurriculumCfg

    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f"错误：找不到模型目录 {args.model_dir}")
        return

    # 加载环境配置
    cfg_yaml = os.path.join(args.model_dir, "env.yaml")
    if os.path.exists(cfg_yaml):
        yaml_cfg = load_yaml(cfg_yaml)
        yaml_cfg = TrafficEnvCfg(**yaml_cfg)
    else:
        print(f"Warning: env.yaml not found in {args.model_dir}, using default config")
        yaml_cfg = TrafficEnvCfg()

    # 创建环境配置
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)

    # 从保存的配置中加载关键参数
    if os.path.exists(cfg_yaml):
        loaded_keys = ["action_space_type", "action_space_num_per_dim", "action_mode", "predict_steps"]
        for key in loaded_keys:
            if hasattr(yaml_cfg, key):
                setattr(cfg, key, getattr(yaml_cfg, key))

    cfg.seed = algo_args.seed
    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    cfg.use_global_path = args.use_global_path
    cfg.debug_vis = True

    algo_args.action_space_type = cfg.action_space_type
    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) + 1

    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),
        lookat=(0., 0., 1.)
    )

    # 创建输出目录
    timestamp = time.strftime("%m%d_%H%M%S")
    output_dir = os.path.join(args.model_dir, 'batch_test_results', f"batch_{args.num_episodes}episodes_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    main_logger = configure(output_dir, ["stdout", "log"])
    main_logger.info("="*60)
    main_logger.info("BATCH MODEL TESTING STARTED")
    main_logger.info("="*60)
    main_logger.info(f"Model directory: {args.model_dir}")
    main_logger.info(f"Number of episodes per model: {args.num_episodes}")
    main_logger.info(f"Number of environments: {args.num_envs}")
    main_logger.info(f"Models to test: {MODEL_LIST}")

    # 获取要测试的模型列表
    models_to_test = get_models_to_test(args.model_dir, MODEL_LIST, args.include_checkpoints)
    
    if not models_to_test:
        main_logger.error("No valid models found for testing!")
        return

    main_logger.info(f"Total models to test: {len(models_to_test)}")
    for model_info in models_to_test:
        main_logger.info(f"  - {model_info['name']}: {model_info['model_file']}")

    # 创建一次base_env（Isaac Sim只能实例化一个）
    main_logger.info("Creating base test environment...")
    base_env = create_test_env(cfg, args=args)
    main_logger.info("Base environment created successfully.")

    # 开始批量测试
    all_results = {}
    
    for i, model_info in enumerate(models_to_test):
        model_name = model_info["name"]
        model_file = model_info["model_file"]
        vecnorm_file = model_info["vecnorm_file"]
        
        main_logger.info("="*50)
        main_logger.info(f"Testing Model {i+1}/{len(models_to_test)}: {model_name}")
        main_logger.info("="*50)
        main_logger.info(f"Model file: {model_file}")
        if vecnorm_file:
            main_logger.info(f"VecNormalize file: {vecnorm_file}")
        
        try:
            # 准备视频录制参数（只为第一个模型录制视频）
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            video_kwargs = None
            if args.video and i == 0:
                video_kwargs = {
                    "video_folder": os.path.join(model_output_dir, "videos", model_name),
                    "step_trigger": lambda step: step % args.video_interval == 0,
                    "video_length": args.video_length,
                    "disable_logger": True,
                }
                main_logger.info("[INFO] Recording videos for the first model.")
            
            # 为当前模型包装环境（重用base_env）
            current_test_env = wrap_env_for_model(base_env, algo_args, vecnorm_file, video_kwargs)

            # 计算观测空间大小（只需要计算一次）
            if i == 0:
                algo_args.robot_node_input_size = (current_test_env.observation_space['robot_node'].shape[1] + 
                                                 current_test_env.observation_space['temporal_edges'].shape[1])
            
            # 加载模型
            model = CustomPPO.load(model_file, env=current_test_env, args=algo_args)
            
            # 创建模型专用的日志

            model_logger = configure(model_output_dir, ["stdout", "log"])
            model.set_logger(model_logger)
            
            model_logger.info(f"Starting evaluation for {args.num_episodes} episodes")
            
            # 进行评估
            start_time = time.time()
            results = evaluate_policy(model, current_test_env, args.num_envs, args.num_episodes, model_logger)
            evaluation_time = time.time() - start_time
            
            results["evaluation_time_seconds"] = evaluation_time
            results["model_file"] = model_file
            results["vecnorm_file"] = vecnorm_file
            
            # 保存单个模型的结果
            model_results_file = os.path.join(model_output_dir, "evaluation_results.json")
            with open(model_results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            all_results[model_name] = results
            
            main_logger.info(f"Model {model_name} evaluation completed in {evaluation_time:.2f} seconds")
            main_logger.info(f"Success Rate: {results['success_rate']:.4f}")
            main_logger.info(f"Collision Rate: {results['collision_rate']:.4f}")
            
        except Exception as e:
            main_logger.error(f"Error testing model {model_name}: {str(e)}")
            import traceback
            main_logger.error(traceback.format_exc())
            all_results[model_name] = {"error": str(e)}

    # 输出最终比较结果
    main_logger.info("="*60)
    main_logger.info("BATCH TESTING COMPLETED - RESULTS SUMMARY")
    main_logger.info("="*60)
    
    # 创建比较表格
    main_logger.info(f"{'Model Name':<25} {'Success Rate':<15} {'Collision Rate':<15} {'Avg Episode Length':<20} {'Avg Reward':<15}")
    main_logger.info("-" * 90)
    
    valid_results = []
    for model_name, results in all_results.items():
        if "error" not in results:
            main_logger.info(f"{model_name:<25} {results['success_rate']:<15.4f} {results['collision_rate']:<15.4f} {results['episode_length']['mean']:<20.2f} {results['episode_reward']['mean']:<15.4f}")
            valid_results.append((model_name, results))
        else:
            main_logger.info(f"{model_name:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR':<20} {'ERROR':<15}")
    
    # 找出最佳模型
    if valid_results:
        best_success_model = max(valid_results, key=lambda x: x[1]['success_rate'])
        lowest_collision_model = min(valid_results, key=lambda x: x[1]['collision_rate'])
        best_reward_model = max(valid_results, key=lambda x: x[1]['episode_reward']['mean'])
        
        main_logger.info("\nBest Performing Models:")
        main_logger.info(f"Highest Success Rate: {best_success_model[0]} ({best_success_model[1]['success_rate']:.4f})")
        main_logger.info(f"Lowest Collision Rate: {lowest_collision_model[0]} ({lowest_collision_model[1]['collision_rate']:.4f})")
        main_logger.info(f"Highest Average Reward: {best_reward_model[0]} ({best_reward_model[1]['episode_reward']['mean']:.4f})")

    # 保存完整的批量测试结果
    batch_results_file = os.path.join(output_dir, "batch_evaluation_results.json")
    batch_summary = {
        "test_config": {
            "model_dir": args.model_dir,
            "num_episodes": args.num_episodes,
            "num_envs": args.num_envs,
            "models_tested": list(all_results.keys()),
            "timestamp": timestamp
        },
        "results": all_results
    }
    
    with open(batch_results_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    main_logger.info(f"\nBatch evaluation results saved to: {batch_results_file}")
    main_logger.info(f"Individual model results saved in: {output_dir}")

    # 关闭base环境
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
