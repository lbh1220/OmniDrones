#!/usr/bin/env python3

import argparse
import os
import time
from datetime import datetime
from dataclasses import replace
import torch
import torch.nn as nn
from rl.sb3.config import ArgsConfig
import yaml

import numpy as np
# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
from stable_baselines3.common.logger import configure
from rl.sb3.custom_callback import SucessRateCallback, CourseWithSuccessRateCallback
from rl.sb3.custom_ppo import CustomPPO
from rl.sb3.custom_policy import CustomSelfAttnPolicy
from rl.sb3.network_utils import linear_schedule_with_min
from rl.sb3.custom_callback import SucessRateCallback
from rl.sb3.vec_normalize import VecNormalize
import gymnasium as gym

# 添加wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

def create_env(cfg, args=None):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvWithCurriculum

    # 创建环境
    if args.course_num > 0:
        env = TrafficEnvWithCurriculum(cfg=cfg, render_mode="rgb_array" if args.video else None)
    else:
        env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)

    return env

from learning.test_traffic import evaluate_model, get_latest_checkpoint_models




def main():
    parser = argparse.ArgumentParser(description="Train Traffic Environment with SB3 PPO")

    # learning params, num_envs, num_mini_batch, n_steps, learning_rate
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--num_mini_batch", type=int, default=32, help="Number of mini batches")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of steps")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    # total timesteps
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps")


    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for saving")
    parser.add_argument("--course_num", type=int, default=0, help="Number of courses")
    # 添加wandb相关参数
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="omni_drones_traffic", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")

    # success reward, collision reward
    parser.add_argument("--rew_success", type=float, default=None, help="Success reward")
    parser.add_argument("--rew_collision", type=float, default=None, help="Collision reward")


    # add args, drones_num and evtols_num, drone_future_penalty and evtol_future_penalty
    parser.add_argument("--drones_num", type=int, default=10, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=0, help="Number of evtols")
    parser.add_argument("--evtol_radius", type=float, default=10.0, help="Evtol radius")    
    parser.add_argument("--drone_future_penalty", type=float, default=0.0, help="Drone future penalty")
    parser.add_argument("--evtol_future_penalty", type=float, default=0.0, help="Evtol future penalty")
    parser.add_argument("--drones_threshold_factor", type=float, default=None, help="Drones threshold factor")
    parser.add_argument("--drones_decay_factor", type=float, default=None, help="Drones decay factor")
    parser.add_argument("--evtols_threshold_factor", type=float, default=None, help="Evtols threshold factor")
    parser.add_argument("--evtols_decay_factor", type=float, default=None, help="Evtols decay factor")

    parser.add_argument("--rew_action_penalty", type=float, default=None, help="Action penalty")
    
    parser.add_argument("--predict_steps", type=int, default=5, help="Predict steps")

    # whether reward normalize
    parser.add_argument("--norm_reward", action="store_true", default=True, help="Reward normalize")
    parser.add_argument("--norm_obs", action="store_true", default=False, help="Reward normalize")

    parser.add_argument("--use_global_path", action="store_true", default=True, help="Use global path")
    parser.add_argument("--rew_cross_track_coeff", type=float, default=0.0, help="Cross track coeff")
    parser.add_argument("--rew_cross_track_alpha", type=float, default=1.0, help="Cross track alpha")

    # action space type
    parser.add_argument("--action_space_type", type=str, default="beta", help="Action space type")
    parser.add_argument("--action_space_num_per_dim", type=int, default=7, help="Action space num per dim")
    parser.add_argument("--action_mode", type=str, default="velocity_components", help="Action mode")

    parser.add_argument("--init_gain", type=float, default=0.01, help="Init gain")

    parser.add_argument("--use_rnn", action="store_true", default=True, help="Use RNN-based recurrent policy")

    # 添加评估相关参数
    parser.add_argument("--eval_after_training", action="store_true", default=True, help="Evaluate models after training")

    # video recording
    parser.add_argument("--video", action="store_true", default=False, help="Record video")
    parser.add_argument("--video_interval", type=int, default=1000, help="Video interval")
    parser.add_argument("--video_length", type=int, default=250, help="Video length")

    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置headless模式
    args.headless = True  # 强制使用headless模式进行训练

    if args.video:
        args.enable_cameras = True
    else:
        args.enable_cameras = False
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    algo_args.num_mini_batch = args.num_mini_batch
    algo_args.n_steps = args.n_steps
    algo_args.lr = args.learning_rate
    algo_args.seq_length = args.n_steps
    algo_args.use_rnn = args.use_rnn
    if args.total_timesteps is not None:
        algo_args.num_env_steps = args.total_timesteps



    # 导入环境配置（必须在AppLauncher之后）
    # 导入环境配置（必须在AppLauncher之后）
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg, TrafficEnvWithCurriculumCfg
    
    # 创建环境配置
    if args.course_num > 0:
        cfg = TrafficEnvWithCurriculumCfg()
    else:
        cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)

    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    cfg.seed = algo_args.seed
    if args.video:
        cfg.debug_vis = True

    if args.course_num > 0:
        from isaac_lab_envs.direct.traffic_env import TrafficCurriculumCfg
        course_list = [
                    TrafficCurriculumCfg(drones_num=6, evtol_num=1, evtol_radius=2.0),
                    TrafficCurriculumCfg(drones_num=6, evtol_num=1, evtol_radius=4.0),
                    TrafficCurriculumCfg(drones_num=6, evtol_num=1, evtol_radius=6.0),
                    TrafficCurriculumCfg(drones_num=6, evtol_num=1, evtol_radius=8.0),
                ]
        args.course_num = len(course_list)
        cfg.curriculum_list = course_list
        cfg.curriculum_learning = True
        cfg.traffic_sim.num_drones = course_list[-1].drones_num
        cfg.traffic_sim.num_evtols = course_list[-1].evtol_num
        cfg.traffic_sim.evtol.safety_radius = course_list[0].evtol_radius
    else:
        cfg.traffic_sim.num_drones = args.drones_num
        cfg.traffic_sim.num_evtols = args.evtols_num
        cfg.traffic_sim.evtol.safety_radius = args.evtol_radius

    cfg.traffic_sim.evtol.random_safety_radius = True
    if args.rew_success is not None:
        cfg.rew_success = args.rew_success
    if args.rew_collision is not None:
        cfg.rew_collision = args.rew_collision
        
    if args.rew_action_penalty is not None:
        cfg.rew_action_penalty = args.rew_action_penalty
        cfg.rew_action_penalty = -abs(args.rew_action_penalty)
    # future reward 
    cfg.rew_drone_future_penalty = -abs(args.drone_future_penalty)
    cfg.rew_evtol_future_penalty = -abs(args.evtol_future_penalty)

    if args.drones_threshold_factor is not None:
        cfg.rew_drones_threshold_factor = args.drones_threshold_factor
    if args.drones_decay_factor is not None:
        cfg.rew_drones_decay_factor = args.drones_decay_factor
    if args.evtols_threshold_factor is not None:
        cfg.rew_evtols_threshold_factor = args.evtols_threshold_factor
    if args.evtols_decay_factor is not None:
        cfg.rew_evtols_decay_factor = args.evtols_decay_factor

    cfg.use_global_path = args.use_global_path
    cfg.rew_cross_track_coeff = args.rew_cross_track_coeff
    cfg.rew_cross_track_alpha = args.rew_cross_track_alpha
    
    cfg.predict_steps = args.predict_steps

    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) 
    algo_args.human_human_edge_input_size = algo_args.human_human_edge_input_size + 1

    cfg.action_space_type = args.action_space_type
    cfg.action_mode = args.action_mode
    cfg.action_space_num_per_dim = args.action_space_num_per_dim

    algo_args.init_gain = args.init_gain
    algo_args.action_space_type = cfg.action_space_type
    # 设置实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"traffic_ppo_{timestamp}"
    else:
        # add timestamp to experiment name
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.experiment_name = f"{args.experiment_name}_{timestamp}"

    # 创建保存目录
    save_dir = f"runs/traffic/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 初始化wandb
    if args.use_wandb and WANDB_AVAILABLE:
        # 设置wandb配置
        wandb_config = {
            "num_envs": args.num_envs,
            "num_mini_batch": args.num_mini_batch,
            "experiment_name": args.experiment_name,
            "traffic_drones": cfg.traffic_sim.num_drones,
            "traffic_evtols": cfg.traffic_sim.num_evtols,
            "predict_steps": cfg.predict_steps,
            "rew_evtol_future_penalty": cfg.rew_evtol_future_penalty,
            "rew_drone_future_penalty": cfg.rew_drone_future_penalty,
            "learning_rate": algo_args.lr,
            "gamma": algo_args.gamma,
            "entropy_coef": algo_args.entropy_coef,
            "ppo_epochs": algo_args.ppo_epoch,
            "n_steps": algo_args.n_steps,
        }
        
        # 确定run name
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        else:
            run_name = f"traffic_ppo_{args.experiment_name}"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            sync_tensorboard=True,  # 自动同步tensorboard日志
            dir=save_dir
        )
        print(f"Wandb initialized: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb logging.")

    from omni.isaac.lab.utils.io import dump_yaml
    dump_yaml(os.path.join(save_dir, "env.yaml"), cfg)

    # Save training arguments as YAML for human readability
    args_dict = vars(args)
    # Convert any non-serializable objects to strings
    serializable_args = {}
    for key, value in args_dict.items():
        try:
            yaml.dump({key: value})  # Test if serializable
            serializable_args[key] = value
        except:
            serializable_args[key] = str(value)  # Convert to string if not serializable
    
    with open(os.path.join(save_dir, "training_args.yaml"), 'w') as f:
        yaml.dump(serializable_args, f, default_flow_style=False, indent=2)
    # create env
    base_env = create_env(cfg, args=args)
    env = base_env

    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    env = Sb3VecEnvWrapper(env)
    
    env.seed(seed=algo_args.seed)
    norm_obs_keys = ['robot_node', 'spatial_edges', 'temporal_edges']
    env = VecNormalize(env, 
                        norm_obs=args.norm_obs, 
                        norm_reward=args.norm_reward, 
                        training=True,
                        clip_obs=10.0, 
                        clip_reward=10.0,
                        gamma=algo_args.gamma,
                        norm_obs_keys=norm_obs_keys,
                        shared_across_agents = True
                        )


    # change robot_node_input_size
    algo_args.robot_node_input_size = env.observation_space['robot_node'].shape[1] + env.observation_space['temporal_edges'].shape[1]
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ortho_init=True,
        squash_output=False,
    )# 这些其实都没用到
    policy_kwargs['optimizer_class'] = torch.optim.Adam
    policy_kwargs['optimizer_kwargs'] = {"eps": algo_args.eps}  # 这里设置和你原来一样的eps
    # create model
    if algo_args.use_linear_lr_decay:
        lr_schedule = linear_schedule_with_min(algo_args.lr, algo_args.min_lr)
    else:
        lr_schedule = algo_args.lr
    callbacks = []
    if args.course_num > 0:
        total_timesteps_per_course = algo_args.num_env_steps/args.course_num
        SR_check_callback = CourseWithSuccessRateCallback(check_freq=getattr(algo_args, 'log_interval', 10),
                                                            save_path=os.path.join(save_dir, 'checkpoints'),
                                                            queue_size=args.num_envs,
                                                            success_rate_threshold=1.1,
                                                            min_episodes_for_curriculum=50,
                                                            initial_lr=algo_args.lr,
                                                            min_lr=algo_args.min_lr,
                                                            total_timesteps_per_course=total_timesteps_per_course,
                                                            warmup_steps=total_timesteps_per_course/10,
                                                            warmup_start_lr_factor=0.1,
                                                            course_num=args.course_num,
                                                            name_prefix='SR')
        callbacks.append(SR_check_callback)
    else:
        SR_check_callback = SucessRateCallback(check_freq=getattr(algo_args, 'log_interval', 10),
                                        save_path=os.path.join(save_dir, 'checkpoints'),
                                        queue_size=args.num_envs,
                                        name_prefix='SR')
        callbacks.append(SR_check_callback)
    model = CustomPPO(
        CustomSelfAttnPolicy,
        env,
        args=algo_args,
        n_steps=algo_args.n_steps,
        batch_size=algo_args.n_steps * algo_args.num_processes // algo_args.num_mini_batch,
        learning_rate=lr_schedule,
        gamma=algo_args.gamma,
        ent_coef=algo_args.entropy_coef,
        vf_coef=algo_args.value_loss_coef,
        max_grad_norm=algo_args.max_grad_norm,
        clip_range=algo_args.clip_param,
        clip_range_vf=algo_args.clip_param,
        policy_kwargs=policy_kwargs,
        n_epochs=algo_args.ppo_epoch,
        verbose=1,
        seed=algo_args.seed,
        tensorboard_log=f"{save_dir}/tensorboard/",
        )

    new_logger = configure(os.path.join(save_dir, 'logs'), ["stdout","tensorboard", "log"])
    model.set_logger(new_logger)
    
    # print args
    model.logger.info(f"drones_num: {cfg.traffic_sim.num_drones}")
    model.logger.info(f"evtols_num: {cfg.traffic_sim.num_evtols}")
    model.logger.info(f"drone_future_penalty: {cfg.rew_drone_future_penalty}")
    model.logger.info(f"evtol_future_penalty: {cfg.rew_evtol_future_penalty}")
    model.logger.info(f"rew_action_penalty: {cfg.rew_action_penalty}")
    if cfg.use_global_path:
        model.logger.info(f"use_global_path: {cfg.use_global_path}")
        model.logger.info(f"rew_cross_track_coeff: {cfg.rew_cross_track_coeff}")
        model.logger.info(f"rew_cross_track_alpha: {cfg.rew_cross_track_alpha}")
    model.logger.info(f"init_gain: {algo_args.init_gain}")
    model.logger.info(f"lr: {algo_args.lr}")
    model.logger.info(f"gamma: {algo_args.gamma}")
    model.logger.info(f"entropy_coef: {algo_args.entropy_coef}")
    model.logger.info(f"value_loss_coef: {algo_args.value_loss_coef}")
    model.logger.info(f"ppo_epoch: {algo_args.ppo_epoch}")
    model.logger.info(f"max_grad_norm: {algo_args.max_grad_norm}")
    model.logger.info(f"clip_param: {algo_args.clip_param}")
    model.logger.info(f"n_steps: {algo_args.n_steps}")
    model.logger.info(f"num_envs: {algo_args.num_processes}")
    model.logger.info(f"num_mini_batch: {algo_args.num_mini_batch}")
    model.logger.info(f"num_env_steps: {algo_args.num_env_steps}")
    model.logger.info(f"Starting training for {algo_args.num_env_steps} timesteps...")
    model.learn(total_timesteps=int(algo_args.num_env_steps), callback=callbacks, log_interval=getattr(algo_args, 'log_interval', 10))
    
    # 保存模型
    model_path = os.path.join(save_dir, "final_model")
    model.save(model_path)
    model.get_vec_normalize_env().save(os.path.join(save_dir, "final_model_vecnormalize.pkl"))
    print(f"Model saved to: {model_path}")

    # 完成wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        # 记录最终模型路径
        wandb.save(model_path)
        wandb.finish()
        print("Wandb run completed and synced")



    # ===============================
    # 训练完成后进行模型评估
    # ===============================
    if args.eval_after_training:
        model.logger.info("="*50)
        model.logger.info("Starting post-training evaluation...")
        model.logger.info("="*50)
        
        # 准备要评估的模型列表
        models_to_evaluate = []
        
        # 1. 添加final_model
        final_model_path = os.path.join(save_dir, "final_model.zip")
        final_vecnorm_path = os.path.join(save_dir, "final_model_vecnormalize.pkl")
        if os.path.exists(final_model_path):
            models_to_evaluate.append(("final_model", final_model_path, final_vecnorm_path))
        
        # 2. 添加最新的两个checkpoint模型
        checkpoints_dir = os.path.join(save_dir, 'checkpoints')
        latest_checkpoints = get_latest_checkpoint_models(checkpoints_dir, num_models=2)
        for checkpoint_path in latest_checkpoints:
            checkpoint_name = os.path.basename(checkpoint_path).replace('.zip', '')
            vecnorm_path = checkpoint_path.replace('.zip', '_vecnormalize.pkl')
            models_to_evaluate.append((checkpoint_name, checkpoint_path, vecnorm_path))
        
        if not models_to_evaluate:
            model.logger.warning("No models found for evaluation!")
        else:
            
            # 创建测试环境
            test_env = base_env
            
            # 为每个模型进行评估
            evaluation_results = {}
            for model_name, model_file, vecnorm_file in models_to_evaluate:
                model.logger.info("-"*30)
                model.logger.info(f"Evaluating {model_name}...")
                model.logger.info(f"Model path: {model_file}")
                
                try:
                    # 创建测试环境的副本
                    current_test_env = test_env
                    if args.video:
                        video_kwargs = {
                            "video_folder": os.path.join(save_dir, "test_videos", model_name),
                            "step_trigger": lambda step: step % args.video_interval == 0,
                            "video_length": args.video_length,
                            "disable_logger": True,
                        }
                        current_test_env = gym.wrappers.RecordVideo(current_test_env, **video_kwargs)
                    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
                    current_test_env = Sb3VecEnvWrapper(current_test_env)
                    # 如果存在vecnormalize文件，应用归一化
                    if os.path.exists(vecnorm_file):
                        model.logger.info(f"Loading VecNormalize: {vecnorm_file}")
                        current_test_env = VecNormalize.load(vecnorm_file, current_test_env)
                        # 测试时不更新归一化统计
                        current_test_env.training = False
                        current_test_env.norm_reward = False
                    current_test_env.seed(seed=algo_args.seed)
                    # 加载模型
                    eval_model = CustomPPO.load(model_file, env=current_test_env, args=algo_args)
                    
                    # 进行评估
                    eval_envs = args.num_envs
                    eval_episodes = max(eval_envs*5, 100)
                    evaluate_results = evaluate_model(
                        eval_model, current_test_env, eval_envs, eval_episodes, model.logger
                    )
                    
                    # 记录结果
                    evaluation_results[model_name] = evaluate_results.copy()
                                        
                except Exception as e:
                    model.logger.error(f"Error evaluating {model_name}: {str(e)}")
                    evaluation_results[model_name] = {}
            
            
            # 保存评估结果到文件
            import json
            eval_results_file = os.path.join(save_dir, "evaluation_results.json")
            with open(eval_results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            model.logger.info(f"Evaluation results saved to: {eval_results_file}")


    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()