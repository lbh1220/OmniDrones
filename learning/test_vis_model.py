#!/usr/bin/env python3

"""Test script for trained SB3 models
加载训练好的模型并进行可视化测试，支持视频录制保存
"""

import argparse
import os
import time
import numpy as np
from dataclasses import replace

from tensordict import base
from learning.rl.sb3.config import ArgsConfig
import torch
from stable_baselines3.common.logger import configure
from learning.rl.sb3.custom_ppo import CustomPPO
from stable_baselines3.ppo import PPO
from learning.rl.sb3.vec_normalize import VecNormalize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dataclasses import dataclass, field
# 移除evaluation导入，Isaac Lab SB3包装器不支持evaluate_policy

# 导入Isaac Lab
from omni.isaac.lab.app import AppLauncher
import math
from typing import Tuple

import gymnasium as gym


def create_test_env(cfg, args=None):
    """创建并包装环境"""
    # 设置headless模式
    
    from isaac_lab_envs.direct.traffic_env import TrafficEnv, TrafficEnvWithCurriculum

    # 创建环境
    env = TrafficEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)

    
    return env


def get_latest_checkpoint_models(checkpoints_dir, num_models=2):
    """获取最新的checkpoint模型文件"""
    import glob
    if not os.path.exists(checkpoints_dir):
        return []
    
    # 查找所有.zip文件
    model_files = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    if not model_files:
        return []
    
    # 按修改时间排序，取最新的num_models个
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[:num_models]

from rl.sb3.evaluate_policy import evaluate_policy


def build_grid(xmin: float, xmax: float, ymin: float, ymax: float, res: float):
    xs = np.arange(xmin, xmax + 1e-6, res, dtype=np.float32)
    ys = np.arange(ymin, ymax + 1e-6, res, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    pos = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    return pos  # [N, 2]


def build_focus_positions(
    traffic_pos: torch.Tensor,
    traffic_radii: torch.Tensor,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    samples_per_traffic: int,
    r_min_factor: float,
    r_max: float,
    dedup_resolution: float,
    seed: int | None = None,
):
    """Sample positions around each traffic in an annulus [r_min, r_max],
    excluding inside safety radii to save memory. Returns np.ndarray [N,2].

    - r_min = r_min_factor * radius_i
    - r_max: absolute cap for outer radius (meters)
    - dedup_resolution: merge near-duplicates by rounding to a grid
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    tp = traffic_pos.detach().cpu().numpy()
    tr = traffic_radii.detach().cpu().numpy()

    samples = []
    for i in range(tp.shape[0]):
        cx, cy = float(tp[i, 0]), float(tp[i, 1])
        r_min = max(0.0, r_min_factor * float(tr[i]))
        r_max_i = float(r_max)
        if r_max_i <= r_min:
            continue
        # Proper radial sampling in annulus: r = sqrt(u*(R^2 - r^2) + r^2)
        u = rng.random(samples_per_traffic)
        r = np.sqrt(u * (r_max_i * r_max_i - r_min * r_min) + r_min * r_min)
        theta = rng.random(samples_per_traffic) * 2.0 * np.pi
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        pts = np.stack([xs, ys], axis=-1)
        # clip to area bounds (reject outside)
        mask = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        pts = pts[mask]
        samples.append(pts)

    if len(samples) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts_all = np.concatenate(samples, axis=0)
    if pts_all.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Deduplicate by rounding to a grid
    if dedup_resolution > 0:
        key = np.round(pts_all / dedup_resolution).astype(np.int64)
        # unique rows
        _, idx = np.unique(key, axis=0, return_index=True)
        pts_all = pts_all[np.sort(idx)]

    return pts_all.astype(np.float32)


def generate_random_traffic(
    cfg, device: torch.device,
):
    xmin, xmax, ymin, ymax = cfg.area_bounds.xmin, cfg.area_bounds.xmax, cfg.area_bounds.ymin, cfg.area_bounds.ymax
    total = cfg.traffic_sim.num_drones + cfg.traffic_sim.num_evtols
    if total == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, device=device),
        )
    xs = torch.rand(total, device=device) * (xmax - xmin) + xmin
    ys = torch.rand(total, device=device) * (ymax - ymin) + ymin
    zs = torch.full((total,), cfg.flight_height, device=device)
    positions = torch.stack([xs, ys, zs], dim=-1)

    headings = 2 * math.pi * torch.rand(total, device=device)
    speeds = torch.cat([
        torch.full((cfg.traffic_sim.num_drones,), cfg.traffic_sim.drone.v_pref, device=device),
        torch.full((cfg.traffic_sim.num_evtols,), cfg.traffic_sim.evtol.v_pref, device=device),
    ], dim=0)
    vx = speeds * torch.cos(headings)
    vy = speeds * torch.sin(headings)
    vz = torch.zeros_like(vx)
    velocities = torch.stack([vx, vy, vz], dim=-1)

    types = torch.cat([
        torch.full((cfg.traffic_sim.num_drones,), 1, dtype=torch.long, device=device),
        torch.full((cfg.traffic_sim.num_evtols,), 2, dtype=torch.long, device=device),
    ], dim=0)
    radii = torch.cat([
        torch.full((cfg.traffic_sim.num_drones,), cfg.traffic_sim.drone.safety_radius, device=device),
        torch.full((cfg.traffic_sim.num_evtols,), cfg.traffic_sim.evtol.safety_radius, device=device),
    ], dim=0)
    return positions, velocities, types, radii

def generate_fix_traffic(
    cfg, device: torch.device,
):
        # evtol在（0, 15) 速度（2，0），两个drones在（-15，-15）和（15，15），速度（1，0）
    flight_height = cfg.flight_height
    positions = torch.tensor([[0.0, 15.0, flight_height], [15.0, -15.0, flight_height], [-15.0, -15.0, flight_height]], device=device)
    velocities = torch.tensor([[2.0, 0.0, 0], [0.0, 1.0, 0], [1.0, 0.0, 0]], device=device)
    types = torch.tensor([2, 1, 1], dtype=torch.long, device=device)
    radii = torch.tensor([cfg.traffic_sim.evtol.safety_radius, cfg.traffic_sim.drone.safety_radius, cfg.traffic_sim.drone.safety_radius], device=device)
    return positions, velocities, types, radii

def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Test trained SB3 model with vector-field visualization")
    parser.add_argument("--model_dir", type=str, 
                        default="runs/traffic_standard/drlvo/u10e0_r8.0_1103_205401",
                       help="Path to the trained model directory")
    parser.add_argument("--model_name", type=str, default="final_model.zip", help="Model name")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes for evaluation")
    

    # add args, drones_num and evtols_num, drone_future_penalty and evtol_future_penalty
    parser.add_argument("--drones_num", type=int, default=2, help="Number of drones")
    parser.add_argument("--evtols_num", type=int, default=1, help="Number of evtols")
    parser.add_argument("--use_rnn", action="store_true", default=True, help="Use RNN-based recurrent policy")
    parser.add_argument("--video", action="store_true", default=True, help="Record video")
    parser.add_argument("--video_interval", type=int, default=1000, help="Video interval")
    parser.add_argument("--video_length", type=int, default=500, help="Video length")
    # 添加AppLauncher参数
    # Grid and area args for vector field
    parser.add_argument("--grid_res", type=float, default=1.0, help="Grid resolution in meters")
    parser.add_argument("--focus_near_traffic", action="store_true", default=False, help="Sample near traffic instead of uniform grid")
    parser.add_argument("--samples_per_traffic", type=int, default=300, help="Number of samples per traffic aircraft")
    parser.add_argument("--r_min_factor", type=float, default=1.1, help="Inner radius = factor * safety_radius")
    parser.add_argument("--r_max", type=float, default=30.0, help="Outer radius (meters) for sampling around traffic")
    parser.add_argument("--dedup_resolution", type=float, default=2.0, help="Rounding step when deduping samples")
    parser.add_argument("--seed", type=int, default=55, help="Random seed")

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


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

        # 导入环境配置（必须在AppLauncher之后）
    from omni.isaac.lab.utils.io import load_yaml
    from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg, TrafficCurriculumCfg
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
    

    cfg_yaml = os.path.join(args.model_dir, "env.yaml")



    yaml_cfg = load_yaml(cfg_yaml)
    yaml_cfg = TrafficEnvCfg(**yaml_cfg)



    
    # 创建环境配置
    cfg = TrafficEnvCfg()


    cfg.scene = replace(cfg.scene, num_envs=1)

    cfg.traffic_sim.num_drones = args.drones_num
    cfg.traffic_sim.num_evtols = args.evtols_num
    # 从保存的配置中加载关键参数
    if os.path.exists(cfg_yaml):
        loaded_keys = ["action_space_type", "action_space_num_per_dim", "action_mode", "predict_steps", "use_angle_distance_obs", "use_global_path"]
        for key in loaded_keys:
            if hasattr(yaml_cfg, key):
                setattr(cfg, key, getattr(yaml_cfg, key))
    if "use_drl_vo" in yaml_cfg.__dict__:
        cfg.use_drl_vo = yaml_cfg.use_drl_vo
    # traffic：所有 env 共享同一组 traffic（位置/速度/类型/半径）
    traffic_pos, traffic_vel, traffic_types, traffic_radii = generate_fix_traffic(
        cfg,
        torch.device("cuda"),
    )




    xmin = cfg.area_bounds.xmin
    xmax = cfg.area_bounds.xmax
    ymin = cfg.area_bounds.ymin
    ymax = cfg.area_bounds.ymax
    # Build positions to evaluate actions
    if args.focus_near_traffic and (traffic_pos is not None) and (traffic_pos.numel() > 0):
        grid_xy = build_focus_positions(
            traffic_pos,
            traffic_radii,
            xmin, xmax, ymin, ymax,
            samples_per_traffic=args.samples_per_traffic,
            r_min_factor=args.r_min_factor,
            r_max=args.r_max,
            dedup_resolution=args.dedup_resolution,
            seed=args.seed,
        )
        if grid_xy.shape[0] == 0:
            grid_xy = build_grid(xmin, xmax, ymin, ymax, args.grid_res)
    else:
        # Fallback to uniform grid
        grid_xy = build_grid(xmin, xmax, ymin, ymax, args.grid_res)  # [N,2]
    # 测试30，30这个点
    # grid_xy = np.array([[30.0, 30.0]], dtype=np.float32)
    num_envs = grid_xy.shape[0]







    algo_args = ArgsConfig()
    algo_args.num_processes = num_envs
    algo_args.use_rnn = args.use_rnn
    


    algo_args.seed = args.seed
    cfg.seed = args.seed
    




    cfg.debug_vis = True



    algo_args.action_space_type = cfg.action_space_type

      

    from omni.isaac.lab.envs.common import ViewerCfg
    cfg.viewer = ViewerCfg(
        resolution=(1920, 1080),
        eye=(125, 0., 125),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    )
    print(f"创建测试环境...")
    output_dir = os.path.join(args.model_dir, 'test_results', args.model_name.replace(".zip", "_") + f"{args.num_episodes}episodes" + time.strftime("%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试环境
    base_env = create_test_env(cfg, args=args)
    env = base_env

    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    env = Sb3VecEnvWrapper(env)
    # 如果存在VecNormalize文件，加载归一化参数
    if os.path.exists(vecnormalize_file):
        env = VecNormalize.load(vecnormalize_file, env)
        # 测试时不更新归一化统计
        env.training = False
        env.norm_reward = False
    if cfg.use_drl_vo:
        pass
    else:
        algo_args.robot_node_input_size = base_env.observation_space['robot_node'].shape[-1] + base_env.observation_space['temporal_edges'].shape[-1]

        algo_args.human_human_edge_input_size = base_env.observation_space['spatial_edges'].shape[-1] 

    # model = CustomPPO.load(model_file, env=env, args=algo_args)
    model = PPO.load(model_file, env=env)



    base_env.seed(seed=algo_args.seed)











    # state = base_env.state
    from isaac_lab_envs.direct.mdp.state import EnvState
    base_env.state = EnvState(device=base_env.device, num_envs=num_envs)
    base_env.state.initialize_basic_tensors()
    base_env.state.collision.safety_radius = base_env.cfg.safety_radius
    base_env.state.init_traffic_namespace(predict_steps=base_env.cfg.predict_steps, pred_timestep=base_env.cfg.pred_timestep)
    state = base_env.state
    start, goal, waypoints = base_env._generate_crossing_task_with_waypoints(1, flight_height=base_env.cfg.flight_height)
    # 起点（xmin,10), 终点(xmax,-10），waypoints中间添加一个(0,0)
    flight_height = base_env.cfg.flight_height
    start = torch.tensor([[xmin, 10.0, flight_height]], device=base_env.device).unsqueeze(0)
    goal = torch.tensor([[xmax, -10.0, flight_height]], device=base_env.device).unsqueeze(0)
    waypoints = torch.tensor([[xmin, 10.0, flight_height], [0.0, 0.0, flight_height], [xmax, -10.0, flight_height]], device=base_env.device).unsqueeze(0)
    # 共享同一task：repeat到 num_envs
    state.navigation.start_positions = start.repeat(num_envs, 1, 1)
    state.navigation.target_positions = goal.repeat(num_envs, 1, 1)
    state.navigation.waypoints = waypoints.repeat(num_envs, 1, 1)
    state.navigation.waypoint_lengths = torch.full((num_envs,), waypoints.shape[1], dtype=torch.long, device=base_env.device)
    state.navigation.current_waypoint_indices = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)




    # 构造批量 drone_state，位置取网格，速度指向 target，模长=max_speed
    drone_state = torch.zeros(num_envs, 1, 13, device=base_env.device)
    drone_state[:, 0, 0] = torch.from_numpy(grid_xy[:, 0]).to(base_env.device)
    drone_state[:, 0, 1] = torch.from_numpy(grid_xy[:, 1]).to(base_env.device)
    drone_state[:, 0, 2] = base_env.cfg.flight_height
    # 计算指向目标的单位方向并赋速度
    if cfg.use_global_path:
        target_xy = state.navigation.local_goals[:, 0, :2]
    else:
        target_xy = state.navigation.target_positions[:, 0, :2]
    pos_xy = drone_state[:, 0, :2]  # [N,2]
    dir_xy = target_xy - pos_xy
    eps = 1e-6
    norm = torch.norm(dir_xy, dim=-1, keepdim=True)
    unit = torch.where(norm > eps, dir_xy / norm, torch.zeros_like(dir_xy))
    vel_xy_init = unit * float(base_env.cfg.max_speed)
    drone_state[:, 0, 7:9] = vel_xy_init
    # drone_state[:, 0, 7:9] = 0.0
    drone_state[:, 0, 9] = 0.0
    

    state.update_ego_drone_state(drone_state)
    state.update_navigation_distances()
    state.update_reached_target_mask(base_env.cfg.arrival_threshold)
    if base_env.cfg.use_global_path:
        state.update_navigation_state_vectorized(base_env.cfg.lookahead_distance)
    state.traffic.traffic_positions = traffic_pos
    state.traffic.traffic_velocities = traffic_vel
    state.traffic.traffic_types = traffic_types
    state.traffic.traffic_safety_radius = traffic_radii

    

    # # 直接写入观测处理器缓存
    if hasattr(base_env.obs_processor, "predict_traffic_trajectory"):
        base_env.obs_processor.predict_traffic_trajectory(traffic_pos, traffic_vel, traffic_types, traffic_radii)

    # 生成观测（取内层 'policy'）
    observations = base_env.obs_processor.process_observation(state)["policy"]
    obs_np = {k: v.detach().cpu().numpy() for k, v in observations.items()}


    # 模型预测（无env闭环，仅单步）
    episode_starts = np.ones((num_envs,), dtype=bool)
    rnn_states = None
    action, rnn_states = model.predict(obs_np, 
                                       state=rnn_states, 
                                       episode_start=episode_starts, 
                                       deterministic=True)
    # 解码为连续速度并通过 env 的预处理规范速度范围
    # 将 numpy -> torch
    if isinstance(action, np.ndarray):
        action_tensor = torch.from_numpy(action).to(base_env.device)
    else:
        action_tensor = action
    base_env._pre_physics_step(action_tensor)

    # 取批量速度向量
    vel_xy = state.navigation.velocity_commands[:, 0, :2].detach().cpu().numpy()

    # 可视化：向量场 + traffic + 路径
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title("Policy velocity field and traffic")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # 自动抽样网格，避免过密
    max_arrows = 4000
    stride = max(1, int(np.sqrt(max(1, grid_xy.shape[0] // max_arrows))))
    ax.quiver(
        grid_xy[::stride, 0], grid_xy[::stride, 1], vel_xy[::stride, 0], vel_xy[::stride, 1],
        angles="xy", scale_units="xy", scale=1.0 / max(1e-6, cfg.max_speed),
        width=0.003, color="tab:blue", alpha=0.8,
    )

    # traffic circles + 速度箭头
    if traffic_pos.numel() > 0:
        tp = traffic_pos.detach().cpu().numpy()
        tr = traffic_radii.detach().cpu().numpy()
        tt = traffic_types.detach().cpu().numpy()
        tv = traffic_vel.detach().cpu().numpy()[:, :2]

        # 绘制圆与位置标记
        for i in range(tp.shape[0]):
            color = "tab:red" if int(tt[i]) == 1 else "tab:orange"
            circ = plt.Circle((tp[i, 0], tp[i, 1]), tr[i], color=color, fill=False, lw=1.2, alpha=0.9)
            ax.add_patch(circ)
            ax.plot(tp[i, 0], tp[i, 1], marker="x", color=color, ms=4)

        # 分类型画速度箭头，按各自首选速度缩放
        drone_mask = (tt == 1)
        evtol_mask = (tt == 2)
        # 防止除零
        drone_scale = 1.0 / max(1e-6, float(cfg.traffic_sim.drone.v_pref))
        evtol_scale = 1.0 / max(1e-6, float(cfg.traffic_sim.evtol.v_pref))

        if np.any(drone_mask):
            ax.quiver(
                tp[drone_mask, 0], tp[drone_mask, 1], tv[drone_mask, 0], tv[drone_mask, 1],
                angles="xy", scale_units="xy", scale=drone_scale,
                width=0.004, color="tab:red", alpha=0.9,
            )
        if np.any(evtol_mask):
            ax.quiver(
                tp[evtol_mask, 0], tp[evtol_mask, 1], tv[evtol_mask, 0], tv[evtol_mask, 1],
                angles="xy", scale_units="xy", scale=evtol_scale,
                width=0.004, color="tab:orange", alpha=0.9,
            )

    # global path (共享同一路径，绘制一次)
    wp = waypoints[0].detach().cpu().numpy()
    ax.plot(wp[:, 0], wp[:, 1], "-", color="tab:green", lw=2.0, label="path")
    ax.plot(wp[0, 0], wp[0, 1], "o", color="tab:green", ms=6, label="start")
    ax.plot(wp[-1, 0], wp[-1, 1], "*", color="tab:purple", ms=10, label="goal")

    # 自定义图例（quiver默认不自动进图例）
    legend_handles = [
        mlines.Line2D([], [], color="tab:blue", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="policy velocity"),
        mlines.Line2D([], [], color="tab:red", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="drone traffic v"),
        mlines.Line2D([], [], color="tab:orange", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="eVTOL traffic v"),
        mlines.Line2D([], [], color="tab:green", linestyle='-', label="path"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    
    fig.tight_layout()
    png_path = os.path.join(output_dir, "policy_velocity_field_and_traffic.png")
    svg_path = os.path.join(output_dir, "policy_velocity_field_and_traffic.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

    plt.show()  # 如需交互，可保留；否则可去掉
    plt.close(fig)  # 释放内存，避免后续空白/叠图
    env.close()

    simulation_app.close()


if __name__ == "__main__":
    main()