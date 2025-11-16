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

import torch
from skrl.utils.spaces.torch import flatten_tensorized_space
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dataclasses import dataclass, field
# 移除evaluation导入，Isaac Lab SB3包装器不支持evaluate_policy

# 导入Isaac Lab

import math
from typing import Tuple

import gymnasium as gym

def build_grid(xmin: float, xmax: float, ymin: float, ymax: float, res: float):
    xs = np.arange(xmin, xmax + 1e-6, res, dtype=np.float32)
    ys = np.arange(ymin, ymax + 1e-6, res, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    pos = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    return pos  # [N, 2]

def predict_traffic_trajectory(
    traffic_positions: torch.Tensor, 
    traffic_velocities: torch.Tensor, 
    traffic_types: torch.Tensor, 
    traffic_safety_radius: torch.Tensor,
    predict_steps: int,
    pred_timestep: float,
):
    """预计算traffic轨迹预测，供后续观测和奖励计算使用
    
    Args:
        traffic_positions: [total_traffic, 3] traffic位置
        traffic_velocities: [total_traffic, 3] traffic速度
        traffic_types: [total_traffic] traffic类型（1=drone, 2=evtol；0=dummy）
    """
    if traffic_positions.numel() == 0:
        traffic_future_traj = torch.empty(0, predict_steps + 1, 3, device=traffic_positions.device)
        return
        
    # 使用恒速模型预测轨迹
    total_traffic = traffic_positions.shape[0]

    traffic_future_traj = torch.zeros(total_traffic, predict_steps + 1, 3, device=traffic_positions.device)
    
    # 第0步是当前位置
    traffic_future_traj[:, 0] = traffic_positions
    
    # 恒速预测后续位置
    for step in range(1, predict_steps + 1):
        traffic_future_traj[:, step] = (
            traffic_positions + traffic_velocities * step * pred_timestep
        )

    return traffic_future_traj

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
        # 另外一种方式是让evtol在path上
    flight_height = cfg.flight_height
    # positions = torch.tensor([[0.0, 15.0, flight_height], [-15.0, -15.0, flight_height], [15.0, -15.0, flight_height]], device=device)
    # velocities = torch.tensor([[2.0, 0.0, 0], [-1.0, 0.0, 0], [0.707, 0.707, 0]], device=device)
    positions = torch.tensor([[0.0, 20.0, flight_height], [-15.0, -15.0, flight_height], [15.0, -15.0, flight_height]], device=device)
    velocities = torch.tensor([[0.0, -2.0, 0], [-1.0, 0.0, 0], [0.707, 0.707, 0]], device=device)
    types = torch.tensor([2, 1, 1], dtype=torch.long, device=device)
    radii = torch.tensor([cfg.traffic_sim.evtol.safety_radius, cfg.traffic_sim.drone.safety_radius, cfg.traffic_sim.drone.safety_radius], device=device)
    return positions, velocities, types, radii
def generate_one_traffic(
    cfg, device: torch.device,
):
        # evtol在（0, 15) 速度（2，0），两个drones在（-15，-15）和（15，15），速度（1，0）
        # 另外一种方式是让evtol在path上
    flight_height = cfg.flight_height
    # positions = torch.tensor([[0.0, 15.0, flight_height], [-15.0, -15.0, flight_height], [15.0, -15.0, flight_height]], device=device)
    # velocities = torch.tensor([[2.0, 0.0, 0], [-1.0, 0.0, 0], [0.707, 0.707, 0]], device=device)
    positions = torch.tensor([[0.0, 20.0, flight_height]], device=device)
    velocities = torch.tensor([[0.0, -2.0, 0]], device=device)
    types = torch.tensor([2], dtype=torch.long, device=device)
    radii = torch.tensor([cfg.traffic_sim.evtol.safety_radius], device=device)
    return positions, velocities, types, radii
def visualize_model_velocity(model, cfg, base_env, output_dir, grid_res=1.0):
    """主函数"""

    # traffic：所有 env 共享同一组 traffic（位置/速度/类型/半径）
    traffic_pos, traffic_vel, traffic_types, traffic_radii = generate_one_traffic(
        cfg,
        torch.device("cuda"),
    )




    xmin = cfg.area_bounds.xmin
    xmax = cfg.area_bounds.xmax
    ymin = cfg.area_bounds.ymin
    ymax = cfg.area_bounds.ymax

    grid_xy = build_grid(xmin, xmax, ymin, ymax, grid_res)  # [N,2]
    # grid_xy = np.array([[20, 40]])
    # filter out grid points that are inside any traffic safety radius
    if traffic_pos.numel() > 0:
        tp = traffic_pos.detach().cpu().numpy()[:, :2]  # [T,2]
        tr = traffic_radii.detach().cpu().numpy()       # [T]
        tr = tr + 1.0 # add a ego radius
        # compute squared distance from each grid point to each traffic
        diff = grid_xy[:, None, :] - tp[None, :, :]     # [N,T,2]
        dist2 = np.sum(diff * diff, axis=-1)            # [N,T]
        inside_any = dist2 <= (tr[None, :] ** 2)        # [N,T]
        keep_mask = ~np.any(inside_any, axis=1)         # [N]
        grid_xy = grid_xy[keep_mask]
    num_envs = grid_xy.shape[0]

    if hasattr(model.policy, "num_envs"):
        model.policy.num_envs = num_envs
        model.init()
        model.set_running_mode("eval")
        # model.policy.sequence_length = rollouts
    
    # state = base_env.state
    from isaac_lab_envs.direct.mdp.state import EnvState
    base_env.state = EnvState(device=base_env.device, num_envs=num_envs)
    base_env.state.initialize_basic_tensors()
    base_env.state.collision.safety_radius = base_env.cfg.safety_radius
    base_env.state.init_traffic_namespace(predict_steps=base_env.cfg.predict_steps, pred_timestep=base_env.cfg.pred_timestep)
    state = base_env.state



    # 构造批量 drone_state，位置取网格，速度指向 target，模长=max_speed
    drone_state = torch.zeros(num_envs, 1, 14, device=base_env.device)
    drone_state[:, 0, 0] = torch.from_numpy(grid_xy[:, 0]).to(base_env.device)
    drone_state[:, 0, 1] = torch.from_numpy(grid_xy[:, 1]).to(base_env.device)
    drone_state[:, 0, 2] = base_env.cfg.flight_height
    drone_state[:, 0, 3] = 1.0 # yaw = 0


    # 构造task
    # 起点（xmin,10), 终点(xmax,-10），waypoints中间添加一个(0,0)
    flight_height = base_env.cfg.flight_height
    start = torch.tensor([[xmin, 0.0, flight_height]], device=base_env.device).unsqueeze(0)
    goal = torch.tensor([[xmax-5, -10.0, flight_height]], device=base_env.device).unsqueeze(0)
    inter_points = torch.tensor([[0.0, 0.0, flight_height]], device=base_env.device).unsqueeze(0)
    waypoints = torch.cat([start, inter_points, goal], dim=1)
    # waypoints = torch.cat([start, goal], dim=1)
    # waypoints = torch.tensor([[xmin, 0.0, flight_height], [0.0, 0.0, flight_height], [xmax, 0.0, flight_height]], device=base_env.device).unsqueeze(0)
    # 共享同一task：repeat到 num_envs
    # share one global path
    state.navigation.start_positions = start.repeat(num_envs, 1, 1)
    state.navigation.target_positions = goal.repeat(num_envs, 1, 1)
    state.navigation.waypoints = waypoints.repeat(num_envs, 1, 1)
    state.navigation.waypoint_lengths = torch.full((num_envs,), waypoints.shape[1], dtype=torch.long, device=base_env.device)
    state.navigation.current_waypoint_indices = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)


    point_to_point_path = True
    if point_to_point_path:
        # do not share global path, use point to point path
        state.navigation.start_positions = drone_state[:, :, :3]
        state.navigation.waypoints = torch.cat([state.navigation.start_positions, state.navigation.target_positions], dim=1)
        state.navigation.waypoint_lengths = torch.full((num_envs,), 2, dtype=torch.long, device=base_env.device)
        state.navigation.current_waypoint_indices = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)


    

    state.update_ego_drone_state(drone_state)
    state.update_navigation_distances()
    state.update_reached_target_mask(base_env.cfg.arrival_threshold)
    if base_env.cfg.use_global_path:
        state.update_navigation_state_vectorized(base_env.cfg.global_path_planner_cfg.lookahead_distance)
    state.traffic.traffic_positions = traffic_pos
    state.traffic.traffic_velocities = traffic_vel
    state.traffic.traffic_types = traffic_types
    state.traffic.traffic_safety_radius = traffic_radii
    

    # 直接写入观测处理器缓存
    traffic_future_traj = predict_traffic_trajectory(traffic_pos, traffic_vel, traffic_types, traffic_radii, base_env.cfg.predict_steps, base_env.cfg.pred_timestep)
    state.traffic.traffic_future_traj = traffic_future_traj
    
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
    vel_xy = unit * float(base_env.cfg.max_speed)
    # vel_xy = torch.zeros_like(vel_xy)
    drone_state[:, 0, 7:9] = vel_xy
    drone_state[:, 0, 9] = 0.0


    with torch.inference_mode():
        for i in range(3):
            drone_state[:, 0, 7:9] = vel_xy
            state.update_ego_drone_state(drone_state)
            # 生成观测（取内层 'policy'）
            observations = base_env.obs_processor.process_observation(state)["policy"]
            # 可能需要把observation flatten
            
            obs_flat = flatten_tensorized_space(observations)


            # 模型预测（无env闭环，仅单步）
            # with torch.inference_mode():
            actions = model.act(obs_flat, timestep=0, timesteps=0)[0]

            base_env._pre_physics_step(actions)
            # 取批量速度向量
            vel_xy = state.navigation.velocity_commands[:, 0, :2].detach()
    #         if i == 0:
    #             vel_xy0 = vel_xy.clone()

    # mask_y15 = (grid_xy[:, 1] < 15.0) & (grid_xy[:, 0] < -8.5)
    # vel_xy[mask_y15] = vel_xy0[mask_y15]

    vel_xy = vel_xy.cpu().numpy()
    

    # 可视化：向量场 + traffic + 路径
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    ax.set_ylim([-30,30])
    # ax.set_title("Policy velocity field and traffic")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

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

    # global path（仅在非 point-to-point 模式下绘制一次）
    show_global_path = not point_to_point_path
    if show_global_path:
        wp = waypoints[0].detach().cpu().numpy()
        ax.plot(wp[:, 0], wp[:, 1], "-", color="tab:green", lw=2.0, label="path")
        ax.plot(wp[0, 0], wp[0, 1], "o", color="tab:green", ms=6, label="start")
        # 目标单独绘制（不再用这里的 star）

    # 单独绘制共享 target（所有 agent 仅画一个）
    tgt = state.navigation.target_positions[0, 0, :2].detach().cpu().numpy()
    ax.scatter(
        tgt[0], tgt[1],
        s=140, marker="X",
        color="tab:purple", edgecolors="white", linewidths=1.0,
        zorder=5,
    )

    # 自定义图例（quiver默认不自动进图例）
    legend_handles = [
        mlines.Line2D([], [], color="tab:blue", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="policy velocity"),
        mlines.Line2D([], [], color="tab:red", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="drone traffic v"),
        mlines.Line2D([], [], color="tab:orange", marker=r'$\rightarrow$', linestyle='None', markersize=10, label="eVTOL traffic v"),
    ]
    if show_global_path:
        legend_handles.append(mlines.Line2D([], [], color="tab:green", linestyle='-', label="path"))
    legend_handles.append(mlines.Line2D([], [], color="tab:purple", marker='X', linestyle='None', markersize=10, label="target"))
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    
    fig.tight_layout()
    png_path = os.path.join(output_dir, "policy_velocity_field_and_traffic.png")
    svg_path = os.path.join(output_dir, "policy_velocity_field_and_traffic.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

    plt.show()  # 如需交互，可保留；否则可去掉
    plt.close(fig)  # 释放内存，避免后续空白/叠图
