#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns


from omni.isaac.lab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description="Test Forest Environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)



from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessor
from isaac_lab_envs.direct.mdp.rewards import TrafficRewardCalculator

# 设置device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化配置和处理器
cfg = TrafficEnvCfg()
obs_processor = TrafficObservationProcessor(cfg, device)
reward_calculator = TrafficRewardCalculator(cfg, device)

# 确保初始化势能缓存为None（在_compute_potential_reward中会自动初始化）
reward_calculator.previous_potential = None

# 定义traffic飞机数据
traffic_positions = torch.tensor([[0.0, 0.0, 20.0], [15, 15, 20.0]], 
                                device=device, dtype=torch.float32)
traffic_velocities = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
                                 device=device, dtype=torch.float32)
traffic_types = torch.tensor([0, 1], device=device)  # 1=drone, 0=evtol
traffic_safety_radius = torch.tensor([10.0, 1.0], device=device, dtype=torch.float32)

# 预计算轨迹
obs_processor.predict_traffic_trajectory(traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius)

print("Traffic future trajectory shape:", obs_processor.traffic_future_traj.shape)
print("Traffic future trajectory:\n", obs_processor.traffic_future_traj)

def visualize_traffic_and_penalty():
    """可视化traffic飞机和future collision penalty分布"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # =================== 左图：Traffic飞机可视化 ===================
    ax1.set_title("Traffic Aircraft Visualization", fontsize=14, fontweight='bold')
    
    # 获取数据
    positions = traffic_positions.cpu().numpy()
    velocities = traffic_velocities.cpu().numpy()
    radii = traffic_safety_radius.cpu().numpy()
    types = traffic_types.cpu().numpy()
    future_traj = obs_processor.traffic_future_traj.cpu().numpy()
    
    # 定义颜色
    colors = ['red', 'blue']  # evtol=red, drone=blue
    type_names = ['eVTOL', 'Drone']
    
    for i, (pos, vel, radius, aircraft_type) in enumerate(zip(positions, velocities, radii, types)):
        color = colors[aircraft_type]
        
        # 画安全半径圆圈
        circle = patches.Circle((pos[0], pos[1]), radius, 
                              fill=False, edgecolor=color, linewidth=2, alpha=0.7)
        ax1.add_patch(circle)
        
        # 画当前位置点
        ax1.scatter(pos[0], pos[1], color=color, s=100, zorder=5, 
                   label=f'{type_names[aircraft_type]} {i+1}' if i < 3 else "")
        
        # 画速度向量
        ax1.arrow(pos[0], pos[1], vel[0]*2, vel[1]*2, 
                 head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=0.8)
        
        # 画未来轨迹（虚线）
        future_x = future_traj[i, :, 0]
        future_y = future_traj[i, :, 1]
        ax1.plot(future_x, future_y, '--', color=color, alpha=0.6, linewidth=2)
        
        # 标记未来位置点
        ax1.scatter(future_x[1:], future_y[1:], color=color, s=30, alpha=0.6, zorder=3)
        
        # 添加标签
        ax1.annotate(f'{type_names[aircraft_type]} {i+1}', 
                    (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # =================== 右图：Future Collision Penalty热力图 ===================
    ax2.set_title("Future Collision Penalty Distribution", fontsize=14, fontweight='bold')
    
    # 创建网格用于计算penalty
    x_range = np.linspace(-30, 30, 50)
    y_range = np.linspace(-30, 30, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # 创建用于计算的drone_state和target_pos
    grid_points = torch.stack([
        torch.tensor(X.flatten(), device=device, dtype=torch.float32),
        torch.tensor(Y.flatten(), device=device, dtype=torch.float32),
        torch.full((X.size,), 20.0, device=device, dtype=torch.float32)  # 固定高度
    ], dim=1)  # [num_points, 3]
    
    num_points = grid_points.shape[0]
    
    # 构造drone_state [num_points, 1, 13]
    drone_state = torch.zeros(num_points, 1, 13, device=device, dtype=torch.float32)
    drone_state[:, 0, :3] = grid_points  # 位置
    drone_state[:, 0, 3:7] = torch.tensor([1., 0., 0., 0.], device=device)  # 四元数
    drone_state[:, 0, 7:10] = 0.0  # 速度为0
    
    # 构造target_pos（这里不影响future penalty计算，可以随意设置）
    target_pos = torch.zeros(num_points, 1, 3, device=device, dtype=torch.float32)
    target_pos[:, 0, :] = grid_points + 5.0  # 目标位置
    
    # 计算future collision penalty
    print("Computing future collision penalties...")
    with torch.no_grad():
        penalties = reward_calculator._compute_future_collision_penalty_refactored(
            drone_state,
            obs_processor.traffic_future_traj,
            traffic_safety_radius,
            traffic_types
        )
    
    # 将结果转换为numpy并reshape
    penalty_grid = penalties.cpu().numpy().reshape(X.shape)
    
    # 创建热力图
    # 使用负值的对数来更好地显示分布（因为penalty都是负数）
    penalty_for_plot = -penalty_grid  # 转为正数
    penalty_for_plot = np.maximum(penalty_for_plot, 1e-6)  # 避免log(0)
    
    # 绘制热力图
    im = ax2.contourf(X, Y, penalty_for_plot, levels=20, cmap='hot', alpha=0.8)
    contours = ax2.contour(X, Y, penalty_for_plot, levels=10, colors='black', linewidths=0.5, alpha=0.6)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Collision Risk (-penalty)', fontsize=12)
    
    # 叠加traffic飞机位置
    for i, (pos, radius, aircraft_type) in enumerate(zip(positions, radii, types)):
        color = colors[aircraft_type]
        
        # 画安全半径圆圈
        circle = patches.Circle((pos[0], pos[1]), radius, 
                              fill=False, edgecolor=color, linewidth=2)
        ax2.add_patch(circle)
        
        # 画当前位置点
        ax2.scatter(pos[0], pos[1], color=color, s=100, zorder=5, 
                   edgecolors='white', linewidth=2)
        
        # 画未来轨迹
        future_x = future_traj[i, :, 0]
        future_y = future_traj[i, :, 1]
        ax2.plot(future_x, future_y, '--', color=color, linewidth=2, alpha=0.9)
    
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 设置相同的坐标范围
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些统计信息
    print(f"\nPenalty Statistics:")
    print(f"Min penalty: {penalties.min().item():.4f}")
    print(f"Max penalty: {penalties.max().item():.4f}")
    print(f"Mean penalty: {penalties.mean().item():.4f}")
    print(f"Std penalty: {penalties.std().item():.4f}")
    
    # 找出风险最高的位置
    max_risk_idx = penalties.argmin()  # 最负的penalty = 最高风险
    max_risk_pos = grid_points[max_risk_idx].cpu().numpy()
    print(f"Highest risk position: ({max_risk_pos[0]:.2f}, {max_risk_pos[1]:.2f}) with penalty: {penalties[max_risk_idx].item():.4f}")

if __name__ == "__main__":
    print("Starting visualization...")
    visualize_traffic_and_penalty()
    print("Visualization complete!")