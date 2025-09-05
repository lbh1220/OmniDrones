#!/usr/bin/env python3

import argparse
import os
import time
from datetime import datetime
from dataclasses import replace
import torch
import torch.nn as nn
from rl.sb3.config import ArgsConfig

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
from stable_baselines3.common.logger import configure
from rl.sb3.custom_callback import SucessRateCallback, CourseWithSuccessRateCallback
from rl.sb3.custom_ppo import CustomPPO
from rl.sb3.custom_policy import CustomSelfAttnPolicy
from rl.sb3.network_utils import linear_schedule_with_min
from rl.sb3.custom_callback import SucessRateCallback
from rl.sb3.vec_normalize import VecNormalize

# 添加wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

def create_env(cfg, headless=True):
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
    parser = argparse.ArgumentParser(description="Train Traffic Environment with SB3 PPO")

    # learning params, num_envs, num_mini_batch, num_steps, learning_rate
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
    parser.add_argument("--num_mini_batch", type=int, default=16, help="Number of mini batches")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")


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
    parser.add_argument("--drones_num", type=int, default=0, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=1, help="Number of evtols")
    parser.add_argument("--drone_future_penalty", type=float, default=0.0, help="Drone future penalty")
    parser.add_argument("--evtol_future_penalty", type=float, default=0.0, help="Evtol future penalty")
    parser.add_argument("--drones_threshold_factor", type=float, default=None, help="Drones threshold factor")
    parser.add_argument("--drones_decay_factor", type=float, default=None, help="Drones decay factor")
    parser.add_argument("--evtols_threshold_factor", type=float, default=None, help="Evtols threshold factor")
    parser.add_argument("--evtols_decay_factor", type=float, default=None, help="Evtols decay factor")

    # whether reward normalize
    parser.add_argument("--reward_normalize", action="store_true", help="Reward normalize")

    # 添加AppLauncher参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # 设置headless模式
    args.headless = True  # 强制使用headless模式进行训练
    
    # 启动Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    algo_args.num_mini_batch = args.num_mini_batch
    algo_args.num_steps = args.num_steps
    algo_args.lr = args.learning_rate
    algo_args.seq_length = args.num_steps



    # 导入环境配置（必须在AppLauncher之后）
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # 创建环境配置
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    if args.course_num > 0:
        from isaac_lab_envs.direct.traffic_env import TrafficCurriculumCfg
        course_list = [
                    TrafficCurriculumCfg(drones_num=2, evtol_num=1),
                    TrafficCurriculumCfg(drones_num=5, evtol_num=1),
                    TrafficCurriculumCfg(drones_num=5, evtol_num=2),
                    TrafficCurriculumCfg(drones_num=5, evtol_num=3),
                ]
        args.course_num = len(course_list)
        cfg.curriculum_list = course_list
        cfg.curriculum_learning = True
        cfg.traffic_sim.num_drones = course_list[-1].drones_num
        cfg.traffic_sim.num_evtols = course_list[-1].evtol_num
    else:
        cfg.traffic_sim.num_drones = args.drones_num
        cfg.traffic_sim.num_evtols = args.evtols_num
        
    if args.rew_success is not None:
        cfg.rew_success = args.rew_success
    if args.rew_collision is not None:
        cfg.rew_collision = args.rew_collision

    # future reward 
    cfg.rew_drone_future_penalty = args.drone_future_penalty
    cfg.rew_evtol_future_penalty = args.evtol_future_penalty

    if args.drones_threshold_factor is not None:
        cfg.rew_drones_threshold_factor = args.drones_threshold_factor
    if args.drones_decay_factor is not None:
        cfg.rew_drones_decay_factor = args.drones_decay_factor
    if args.evtols_threshold_factor is not None:
        cfg.rew_evtols_threshold_factor = args.evtols_threshold_factor
    if args.evtols_decay_factor is not None:
        cfg.rew_evtols_decay_factor = args.evtols_decay_factor


    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) 
    algo_args.human_human_edge_input_size = algo_args.human_human_edge_input_size + 1
    # 设置实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"traffic_ppo_{timestamp}"

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
            "num_steps": algo_args.num_steps,
        }
        
        # 确定run name
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        else:
            run_name = f"traffic_ppo_{args.experiment_name}"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
            sync_tensorboard=True,  # 自动同步tensorboard日志
            dir=save_dir
        )
        print(f"Wandb initialized: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb logging.")

    # create env
    env = create_env(cfg, headless=True)
    if hasattr(args, 'reward_normalize'):
        norm_reward = True
    else:
        norm_reward = False
    norm_obs_keys = ['robot_node', 'spatial_edges', 'temporal_edges']
    env = VecNormalize(env, 
                        norm_obs=True, 
                        norm_reward=norm_reward, 
                        training=True,
                        clip_obs=10.0, 
                        clip_reward=10.0,
                        gamma=algo_args.gamma,
                        norm_obs_keys=norm_obs_keys,
                        shared_across_agents = True
                        )

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
                                                            success_rate_threshold=0.8,
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
        n_steps=algo_args.num_steps,
        batch_size=algo_args.num_steps * algo_args.num_processes // algo_args.num_mini_batch,
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
    model.logger.info(f"lr: {algo_args.lr}")
    model.logger.info(f"gamma: {algo_args.gamma}")
    model.logger.info(f"entropy_coef: {algo_args.entropy_coef}")
    model.logger.info(f"value_loss_coef: {algo_args.value_loss_coef}")
    model.logger.info(f"ppo_epoch: {algo_args.ppo_epoch}")
    model.logger.info(f"max_grad_norm: {algo_args.max_grad_norm}")
    model.logger.info(f"clip_param: {algo_args.clip_param}")
    model.logger.info(f"num_steps: {algo_args.num_steps}")
    model.logger.info(f"num_envs: {algo_args.num_processes}")
    model.logger.info(f"num_mini_batch: {algo_args.num_mini_batch}")
    model.logger.info(f"num_env_steps: {algo_args.num_env_steps}")
    model.logger.info(f"Starting training for {algo_args.num_env_steps} timesteps...")
    model.learn(total_timesteps=int(algo_args.num_env_steps), callback=callbacks, log_interval=getattr(algo_args, 'log_interval', 10))
    
    # 保存模型
    model_path = os.path.join(save_dir, "ppo_traffic_demo")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # 完成wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        # 记录最终模型路径
        wandb.save(model_path)
        wandb.finish()
        print("Wandb run completed and synced")


if __name__ == "__main__":
    main()