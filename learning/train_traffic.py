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
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure
from rl.sb3.custom_callback import RewardCallback, CustomCheckpointCallback, SucessRateCallback, CourseWithSuccessRateCallback
from rl.sb3.custom_ppo import CustomPPO
from rl.sb3.custom_policy import CustomSelfAttnPolicy
from rl.sb3.network_utils import linear_schedule_with_min

def create_env(cfg, headless=True):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.traffic_env import TrafficEnv
    # SB3包装器
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    # 创建环境
    env = TrafficEnv(cfg=cfg)
    
    # 使用SB3包装器包装
    env = Sb3VecEnvWrapper(env)
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Environment with SB3 PPO")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--num_mini_batch", type=int, default=4, help="Number of mini batches")
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

    algo_args = ArgsConfig()
    algo_args.num_processes = args.num_envs
    algo_args.num_mini_batch = args.num_mini_batch



    # 导入环境配置（必须在AppLauncher之后）
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
    
    # 创建环境配置
    cfg = TrafficEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
    cfg.traffic_sim.num_drones = 5
    cfg.traffic_sim.num_evtols = 0


    algo_args.human_human_edge_input_size = int(2*(cfg.predict_steps+1)) 
    algo_args.human_human_edge_input_size = algo_args.human_human_edge_input_size + 1
    # 设置实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"traffic_ppo_{timestamp}"

    # 创建保存目录
    save_dir = f"runs/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)


    # create env
    env = create_env(cfg, headless=True)
    
    env = VecNormalize(env, 
                        norm_obs=False, 
                        norm_reward=True, 
                        training=True,
                        clip_obs=10.0, 
                        clip_reward=20.0)

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
    model.learn(total_timesteps=int(algo_args.num_env_steps), callback=callbacks, log_interval=getattr(algo_args, 'log_interval', 10))
    model.save(os.path.join(save_dir, "ppo_traffic_demo"))


if __name__ == "__main__":
    main()