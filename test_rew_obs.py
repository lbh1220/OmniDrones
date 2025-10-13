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

def check_collision(
        traffic_positions: torch.Tensor,
        traffic_safety_radius: torch.Tensor,
        external_positions: torch.Tensor, # 形状: (env_num, m, 3) 或 (m, 3)
        external_safety_radii: torch.Tensor, # 形状: (env_num, m) 或 (m)
    ) -> torch.Tensor:
        """
        检查外部无人机与交通无人机之间是否存在潜在碰撞。
        此函数可以处理2D (m, ...) 或 3D (env_num, m, ...) 的输入。
        """
        
        
        if traffic_positions.shape[0] == 0:
            return torch.zeros_like(external_safety_radii, dtype=torch.bool)
        
        # --- 新增的保障层：检查输入维度 ---
        # 记录原始形状，以便最后恢复
        original_shape = external_safety_radii.shape
        
        # 如果输入是2D的 (m, 3)，我们给它增加一个批处理维度，变成 (1, m, 3)
        if external_positions.ndim == 2:
            external_positions = external_positions.unsqueeze(0)
            external_safety_radii = external_safety_radii.unsqueeze(0)
        # ------------------------------------

        # 1. 维度重塑，便于批处理计算
        # 将输入的 env_num * m 架无人机展平为一个维度
        env_num, m, _ = external_positions.shape
        flat_external_positions = external_positions.view(-1, 3)     # 形状变为: (env_num * m, 3)
        flat_external_radii = external_safety_radii.view(-1)         # 形状变为: (env_num * m)

        # 2. 计算距离矩阵
        # 计算每一架外部无人机到每一架交通无人机的距离
        # distances 形状: (env_num * m, N)
        distances = torch.cdist(flat_external_positions, traffic_positions)

        # 3. 计算阈值矩阵 (核心改动)
        # 我们需要一个和 distances 形状相同的阈值矩阵，
        # 其中每个元素 (i, j) 的值是第 i 架外部无人机和第 j 架交通无人机的安全半径之和。
        traffic_radii = traffic_safety_radius # 形状: (N)

        # 利用广播机制：(env_num * m, 1) + (N,) -> (env_num * m, N)
        # unsqueeze(-1) 将 flat_external_radii 变为列向量
        thresholds = flat_external_radii.unsqueeze(-1) + traffic_radii

        # 4. 执行碰撞判断
        # 逐元素比较距离是否小于对应的阈值
        collision_matrix = distances < thresholds # 形状: (env_num * m, N)
        
        # 检查每架外部无人机是否与 *任何* 一架交通无人机发生了碰撞
        collision_risk = collision_matrix.any(dim=1) # 形状: (env_num * m)

        # 5. 恢复原始形状并返回
        # collision_risk 的形状是 (env_num * m)，我们将其恢复为输入的原始批处理形状
        return collision_risk.view(original_shape)

from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessor
from isaac_lab_envs.direct.mdp.rewards import TrafficRewardCalculator
from isaac_lab_envs.direct.mdp.state import EnvState

# 设置device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化配置和处理器
cfg = TrafficEnvCfg()
cfg.rew_drone_future_penalty = -1.0
cfg.rew_evtol_future_penalty = -1.0
cfg.rew_evtols_decay_factor = 1.0
cfg.rew_evtols_threshold_factor = 1.5

obs_processor = TrafficObservationProcessor(cfg, device)
reward_calculator = TrafficRewardCalculator(cfg, device)

# 确保初始化势能缓存为None（在_compute_potential_reward中会自动初始化）
reward_calculator.previous_potential = None

# 创建并初始化环境状态对象
num_envs = 2
state = EnvState(device=device, num_envs=num_envs)
state.initialize_basic_tensors(state_dim=13)
state.init_traffic_namespace(predict_steps=cfg.predict_steps, pred_timestep=cfg.pred_timestep)

# 定义traffic飞机数据
traffic_positions = torch.tensor([[0.0, 0.0, 20.0], [15, 15, 20.0], [22, 15, 20.0]], 
                                device=device, dtype=torch.float32)
traffic_velocities = torch.tensor([[-2, -0.0, 0.0], [0.0, 1.0, 0.0], [0, 1, 0.0]], 
                                 device=device, dtype=torch.float32)
traffic_types = torch.tensor([0, 1, 0], device=device)  # 0=evtol, 1=drone
traffic_safety_radius = torch.tensor([10.0, 1.0, 1.0], device=device, dtype=torch.float32)

# 定义ego drone状态数据 [num_envs, 1, 13]
drone_state = torch.tensor([[[0.0, 0.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0]],
                            [[10.0, 0.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0]]], 
                            device=device, dtype=torch.float32)
target_pos = torch.tensor([[[10.0, 10.0, 20.0]],
                           [[10.0, 10.0, 20.0]]], device=device, dtype=torch.float32)

# 填充state对象
state.update_ego_drone_state(drone_state)
state.navigation.target_positions = target_pos
state.navigation.start_positions = drone_state[:, :, :3].clone()

# 更新traffic数据到state
state.traffic.traffic_positions = traffic_positions
state.traffic.traffic_velocities = traffic_velocities
state.traffic.traffic_types = traffic_types
state.traffic.traffic_safety_radius = traffic_safety_radius

# 预计算轨迹（同时更新state和obs_processor）
obs_processor.predict_traffic_trajectory(traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius)
state.traffic.traffic_future_traj = obs_processor.traffic_future_traj.clone()

# 更新导航距离
state.update_navigation_distances()

print("Traffic future trajectory shape:", obs_processor.traffic_future_traj.shape)
print("Traffic future trajectory:\n", obs_processor.traffic_future_traj)

print("\n" + "="*60)
print("ENV STATE SUMMARY")
print("="*60)
state_summary = state.get_state_summary()
for key, value in state_summary.items():
    print(f"{key}: {value}")

# 计算并打印观测数据
print("\n" + "="*60)
print("OBSERVATION COMPUTATION CHECK")
print("="*60)
print(f"Drone state shape: {drone_state.shape}")
print(f"Target pos shape: {target_pos.shape}")
print(f"State ego drone positions shape: {state.ego_drone.positions.shape}")
print(f"State navigation target positions shape: {state.navigation.target_positions.shape}")
print(f"State traffic positions shape: {state.traffic.traffic_positions.shape}")
print(f"State traffic future traj shape: {state.traffic.traffic_future_traj.shape}")

# 计算观测（使用state对象）
observations = obs_processor.process_observation(state)

print(f"Observation keys: {list(observations.keys())}")
print(f"Policy observation keys: {list(observations['policy'].keys())}")

# 详细打印每个观测组件
policy_obs = observations['policy']

print("\n--- Robot Node ---")
robot_node = policy_obs['robot_node']
print(f"Shape: {robot_node.shape}")
print(f"Content:")
for i in range(robot_node.shape[0]):
    print(f"  - Relative goal X: {robot_node[i, 0, 0].item():.4f}")
    print(f"  - Relative goal Y: {robot_node[i, 0, 1].item():.4f}")
    print(f"  - Robot radius: {robot_node[i, 0, 2].item():.4f}")
    print(f"  - Robot max speed: {robot_node[i, 0, 3].item():.4f}")
    print(f"  - Robot yaw: {robot_node[i, 0, 4].item():.4f}")

    print("\n--- Temporal Edges ---")
    temporal_edges = policy_obs['temporal_edges']
    print(f"Shape: {temporal_edges.shape}")
    print(f"Content:")
    print(f"  - Velocity X: {temporal_edges[i, 0, 0].item():.4f}")
    print(f"  - Velocity Y: {temporal_edges[i, 0, 1].item():.4f}")

    print("\n--- Spatial Edges ---")
    spatial_edges = policy_obs['spatial_edges']
    print(f"Shape: {spatial_edges.shape}")
    print(f"Content:")
    for j in range(spatial_edges.shape[1]):  # 遍历每个traffic aircraft
        print(f"  Traffic {j+1}:")
        for k in range(spatial_edges.shape[2]):  # 只显示前5个特征
            print(f"    Feature {k}: {spatial_edges[i, j, k].item():.4f}")

    print("\n--- Detected Human Num ---")
    detected_human_num = policy_obs['detected_human_num']
    print(f"Shape: {detected_human_num.shape}")
    print(f"Content: {detected_human_num[i, 0].item()}")

print("\n--- Raw Data Check ---")
print(f"Drone state shape: {drone_state.shape}")
print(f"Target pos shape: {target_pos.shape}")
print(f"Traffic positions shape: {traffic_positions.shape}")
print(f"Traffic velocities shape: {traffic_velocities.shape}")
print(f"Traffic types: {traffic_types.cpu().numpy()}")
print(f"Traffic safety radius: {traffic_safety_radius.cpu().numpy()}")

print("="*60)

# 测试奖励计算
print("\n" + "="*60)
print("REWARD COMPUTATION CHECK")
print("="*60)

# 设置碰撞和到达目标的mask（用于测试）
state.collision.collision_mask = torch.tensor([False, False], device=device)
state.navigation.reached_target_mask = torch.tensor([False, False], device=device)

# 计算奖励
rewards = reward_calculator.compute_reward(state)
print(f"Rewards shape: {rewards.shape}")
print(f"Rewards: {rewards}")

print("="*60)

def visualize_traffic_and_penalty():
    """可视化traffic飞机和future collision penalty分布"""
    
    # 创建图形：三联图（Traffic、Penalty、Collision）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    
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
    grid_drone_state = torch.zeros(num_points, 1, 13, device=device, dtype=torch.float32)
    grid_drone_state[:, 0, :3] = grid_points  # 位置
    grid_drone_state[:, 0, 3:7] = torch.tensor([1., 0., 0., 0.], device=device)  # 四元数
    grid_drone_state[:, 0, 7:10] = 0.0  # 速度为0
    
    # 构造target_pos（这里不影响future penalty计算，可以随意设置）
    grid_target_pos = torch.zeros(num_points, 1, 3, device=device, dtype=torch.float32)
    grid_target_pos[:, 0, :] = grid_points + 5.0  # 目标位置
    
    # 计算future collision penalty（使用直接调用，因为这是批量评估）
    print("Computing future collision penalties...")
    with torch.no_grad():
        penalties = reward_calculator._compute_future_collision_penalty_refactored(
            grid_drone_state,
            state.traffic.traffic_future_traj,
            state.traffic.traffic_safety_radius,
            state.traffic.traffic_types
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
    
    # =================== 第三图：碰撞区域布尔热力图 ===================
    ax3.set_title("Collision Region (ego radius=1.0)", fontsize=14, fontweight='bold')

    # 复用同一网格，构造外部位置与半径
    ego_positions = grid_points.view(Y.shape[0], Y.shape[1], 3)  # [H, W, 3]
    # external_positions 期望形状：(env_num, m, 3) 或 (m, 3)，我们这里做批到每行一组
    # 为了简单，按 (H*W, 1, 3) 调用，让函数自行兼容
    external_positions = grid_points.view(-1, 1, 3)
    external_safety_radii = torch.ones(external_positions.shape[0], 1, device=device) * 1.0

    with torch.no_grad():
        collision_mask_flat = check_collision(
            traffic_positions=traffic_positions,
            traffic_safety_radius=traffic_safety_radius,
            external_positions=external_positions,
            external_safety_radii=external_safety_radii
        )  # 形状: (H*W, 1)

    collision_mask = collision_mask_flat.view(Y.shape[0], Y.shape[1]).cpu().numpy()

    # 用布尔掩码绘制：True 碰撞区域显示为高亮
    ax3.imshow(collision_mask.astype(float), extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', cmap='Reds', alpha=0.6, vmin=0.0, vmax=1.0)

    # 叠加traffic飞机位置与安全半径
    for i, (pos, radius, aircraft_type) in enumerate(zip(positions, radii, types)):
        color = colors[aircraft_type]
        circle = patches.Circle((pos[0], pos[1]), radius, fill=False, edgecolor=color, linewidth=2)
        ax3.add_patch(circle)
        # 未来轨迹
        future_x = future_traj[i, :, 0]
        future_y = future_traj[i, :, 1]
        ax3.plot(future_x, future_y, '--', color=color, linewidth=2, alpha=0.9)

    ax3.set_xlabel('X Position (m)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-30, 30)
    ax3.set_ylim(-30, 30)

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