from isaac_lab_envs.direct.mdp.state import EnvState
from omni.isaac.lab.utils import configclass
from abc import ABC, abstractmethod
import torch
import gymnasium as gym
import numpy as np
from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)


def encode_relative_xy(relative_xy: torch.Tensor, use_angle_distance_obs: bool, circle_radius: float, observation_norm_scale: float) -> torch.Tensor:
    """将相对位置编码为所需表示。
    
    - 旧模式: 直接对 (dx, dy) 做归一化
    - 新模式: 输出 (sin(theta), cos(theta), 1/(d+1))
    
    Args:
        relative_xy: [..., 2]
    Returns:
        编码后的张量: [..., 2] 或 [..., 3]
    """
    if not use_angle_distance_obs:
        return relative_xy / (2 * circle_radius) * observation_norm_scale
    # 新模式
    dx = relative_xy[..., 0]
    dy = relative_xy[..., 1]
    theta = torch.atan2(dy, dx)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    dist = torch.norm(relative_xy, dim=-1)
    inv_dist = 1.0 / (dist + 1.0)
    return torch.stack([sin_theta, cos_theta, inv_dist], dim=-1)

def world_to_body(rel_xy, cy, sy):
    """
    将世界坐标系下的相对位移转换为机器人本体坐标系（只考虑Yaw旋转）。
    支持任意前导批次维度。

    Args:
        rel_xy: 相对位移张量，形状为 [..., 2]。
        cy: 机器人Yaw角的余弦，形状需要能广播到 rel_xy 的前导维度。
            通常为 [..., 1]。
        sy: 机器人Yaw角的正弦，形状需要能广播到 rel_xy 的前导维度。
            通常为 [..., 1]。

    Returns:
        转换到本体坐标系的相对位移，形状与 rel_xy 相同 [..., 2]。
    """
    x = rel_xy[..., 0]
    y = rel_xy[..., 1]
    
    # --- 核心修改 ---
    # 确保 cy 和 sy 的形状能够广播到 x 和 y
    # 如果 cy, sy 来自 [N, 1]，而 x, y 是 [N, K, T]，
    # 我们需要将 cy, sy unsqueeze 成 [N, 1, 1]
    # 通用做法：获取 x 的维度数，然后 unsqueeze cy/sy 直到维度数匹配
    num_extra_dims = x.dim() - cy.dim()
    if num_extra_dims > 0:
        # 在 cy 和 sy 的末尾添加所需的 singleton 维度
        cy = cy.view(cy.shape + (1,) * num_extra_dims)
        sy = sy.view(sy.shape + (1,) * num_extra_dims)

    # --- 旋转计算 (现在广播可以正常工作) ---
    bx = x * cy + y * sy
    by = -x * sy + y * cy
    
    return torch.stack([bx, by], dim=-1)


class ObservationModule(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda"

    def process_observation(self, state: EnvState) -> dict:
        raise NotImplementedError

    def get_observation_space(self) -> dict:
        raise NotImplementedError

class RobotNodeObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def process_observation(self, state: EnvState) -> dict:
        return robot_node_observation(self.cfg, state)

    def get_observation_space(self) -> dict:
        return robot_node_dim(self.cfg)

class TrafficSpatialEdgesObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def process_observation(self, state: EnvState) -> dict:
        return traffic_spatial_edges_observation(self.cfg, state)

    def get_observation_space(self) -> dict:
        return traffic_spatial_edges_dim(self.cfg)

class LidarObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def process_observation(self, state: EnvState) -> dict:
        return lidar_observation(self.cfg, state)

    def get_observation_space(self) -> dict:
        return lidar_dim(self.cfg)


def robot_node_dim(cfg):
    use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
    use_global_path = getattr(cfg, 'use_global_path', False)
    robot_node_dim = 6
    if use_angle_distance_obs:
        robot_node_dim += 1 # 1 dim for relative target goal
    if use_global_path:
        robot_node_dim += 4 # 4 dim for relative local goal and projection point
        if use_angle_distance_obs:
            robot_node_dim += 2 # 2 dim for relative local goal and projection point

    return {'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, robot_node_dim), dtype=np.float32)}

def robot_node_observation(cfg, state: EnvState) -> dict:

    use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
    observation_norm_scale = getattr(cfg, 'observation_norm_scale', 10.0)
    bounds = getattr(cfg, 'area_bounds', None)
    if bounds is None:
        raise ValueError("area_bounds is not set")
    area_size = max(
        bounds['xmax'] - bounds['xmin'],
        bounds['ymax'] - bounds['ymin']
    )
    circle_radius = area_size / 2.0 * 1.4142135623730951

    device = state.device
    drone_state = state.ego_drone.drone_state
    target_pos = state.navigation.target_positions

    robot_pos = drone_state[:, :, :2]  # [N,1,2]
    robot_vel_world = drone_state[:, :, 7:9]  # [N,1,2]
    robot_quat = drone_state[:, :, 3:7]
    robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
    cy = torch.cos(robot_yaw)
    sy = torch.sin(robot_yaw)
    encoded_robot_yaw = torch.stack([cy, sy], dim=-1)
    # helper: rotate world -> body (yaw-only)

    goal_pos = target_pos[:, :, :2]
    rel_goal_world = goal_pos - robot_pos
    rel_goal_body = world_to_body(rel_goal_world, cy, sy)
    encoded_rel_goal = encode_relative_xy(rel_goal_body, 
                        use_angle_distance_obs, circle_radius, observation_norm_scale)
    vel_body = world_to_body(robot_vel_world, cy, sy)

    robot_radius = torch.full((drone_state.shape[0], 1, 1), cfg.safety_radius, device=device)
    robot_v_pref = torch.full((drone_state.shape[0], 1, 1), cfg.v_pref, device=device)

    # [rel_goal(2/3), radius(1), v_pref(1), yaw(1), vel_body(2)]
    robot_node = torch.cat([
        encoded_rel_goal,
        robot_radius,
        robot_v_pref,
        encoded_robot_yaw,
        vel_body,
    ], dim=-1)

    if cfg.use_global_path:
        local_goals = state.navigation.local_goals
        projection_points = state.navigation.projection_points
        if local_goals is not None:
            rel_local_goal_body = world_to_body(local_goals[:, :, :2] - robot_pos, cy, sy)
        else:
            rel_local_goal_body = torch.zeros_like(rel_goal_body)
        enc_rel_local = encode_relative_xy(rel_local_goal_body, 
                use_angle_distance_obs, circle_radius, observation_norm_scale)

        if projection_points is not None:
            rel_proj_body = world_to_body(projection_points[:, :, :2] - robot_pos, cy, sy)
        else:
            rel_proj_body = torch.zeros_like(rel_goal_body)
        enc_rel_proj = encode_relative_xy(rel_proj_body, 
                use_angle_distance_obs, circle_radius, observation_norm_scale)
        robot_node = torch.cat([
            robot_node,
            enc_rel_local,
            enc_rel_proj,
        ], dim=-1)

    return {'robot_node': robot_node}




def lidar_dim(cfg):
    lidar_resolution = getattr(cfg, 'lidar_resolution', (36, 4))
    return {'lidar': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, lidar_resolution[0], lidar_resolution[1]), dtype=np.float32)}

def lidar_observation(cfg, state: EnvState) -> dict:
    lidar_range = getattr(cfg, 'lidar_range', 10.0)
    lidar_resolution = getattr(cfg, 'lidar_resolution', (36, 4))
    if state.perception.lidar_scan is not None:
        lidar_scan = state.perception.lidar_scan / lidar_range
    else:
        h, w = lidar_resolution
        lidar_scan = torch.zeros(state.ego_drone.drone_state.shape[0], 1, h, w, device=state.device)
    return {'lidar': lidar_scan}


def traffic_spatial_edges_dim(cfg):
    predict_steps = getattr(cfg, 'predict_steps', 5)
    use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)

    spatial_point_dim = 3 if use_angle_distance_obs else 2 
    spatial_dim_output = spatial_point_dim * (predict_steps + 1) + 1 # add 1 dim for safety radius
    
    drone_num = cfg.traffic_sim.num_drones
    evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
    total_traffic_num = max(drone_num + evtol_num, 20)
    return {
            'spatial_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, spatial_dim_output), dtype=np.float32),
            'visible_masks': gym.spaces.Box(low=0.0, high=1.0, shape=(total_traffic_num,), dtype=np.float32),
            'spatial_types': gym.spaces.Box(low=0, high=2, shape=(total_traffic_num,), dtype=np.int64),
            }

def traffic_spatial_edges_observation(cfg, state: EnvState) -> dict:

    """使用缓存的轨迹数据计算空间边观测
    
    Args:
        robot_pos: [num_envs, 1, 2] ego位置
        robot_vel: [num_envs, 1, 2] ego速度
        
    Returns:
        spatial_edges: [num_envs, total_traffic_num, spatial_dim]
        visible_masks: [num_envs, total_traffic_num]
        spatial_types: [num_envs, total_traffic_num] (1=drone, 2=evtol；0=dummy)
    """
    predict_steps = getattr(cfg, 'predict_steps', 5)
    use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
    
    drone_num = cfg.traffic_sim.num_drones
    evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
    total_traffic_num = max(drone_num + evtol_num, 20)
    sensor_range = getattr(cfg, 'observation_radius', 1000)
    sensor_range = 1000
    bounds = cfg.traffic_sim.area_bounds
    area_size = max(
        bounds['xmax'] - bounds['xmin'],
        bounds['ymax'] - bounds['ymin']
    )
    circle_radius = area_size / 2.0 * 1.4142135623730951
    observation_norm_scale = getattr(cfg, 'observation_norm_scale', 10.0)

    num_envs = state.num_envs
    device = state.device

    drone_state = state.ego_drone.drone_state
    robot_pos = drone_state[:, :, :2] 
    robot_vel = drone_state[:, :, 7:9]
    traffic_future_traj = state.traffic.traffic_future_traj
    robot_quat = drone_state[:, :, 3:7]
    robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
    cy = torch.cos(robot_yaw)
    sy = torch.sin(robot_yaw)

    # helper: rotate world -> body (yaw-only)
    spatial_point_dim = 3 if use_angle_distance_obs else 2 
    spatial_dim_xy = 2 * (predict_steps + 1) 
    spatial_dim_encoded = spatial_point_dim * (predict_steps + 1)
    spatial_dim_output = spatial_point_dim * (predict_steps + 1) + 1 # add 1 dim for safety radius
    
    if traffic_future_traj is None or traffic_future_traj.numel() == 0:
        spatial_edges = torch.zeros(
            (num_envs, total_traffic_num, spatial_dim_output), device=device
        )
        visible_masks = torch.zeros((num_envs, total_traffic_num), device=device)
        spatial_types = torch.zeros((num_envs, total_traffic_num), dtype=torch.long, device=device)
        return spatial_edges, visible_masks, spatial_types
    
    # 计算相对位置 [num_envs, total_traffic, predict_steps+1, 2]
    traffic_pos_2d = traffic_future_traj[:, :, :2]  # [total_traffic, predict_steps+1, 2]
    robot_pos_expanded = robot_pos.unsqueeze(1)  # [num_envs, 1, 1, 2]
    
    relative_pos = traffic_pos_2d.unsqueeze(0) - robot_pos_expanded  # [num_envs, total_traffic, predict_steps+1, 2]
    relative_pos = world_to_body(relative_pos, cy, sy) # [num_envs, total_traffic, predict_steps+1, 2]
    
    # 计算距离用于感知范围过滤
    current_distances = torch.norm(relative_pos[:, :, 0], dim=-1)  # [num_envs, total_traffic]
    in_range_mask = current_distances <= sensor_range
    # 假设都能看到        
    # 展平预测位置为spatial edges格式
    if use_angle_distance_obs:
        # 编码为 (sin, cos, 1/(d+1)) 并展平
        encoded = encode_relative_xy(relative_pos, use_angle_distance_obs, circle_radius, observation_norm_scale)  # [num_envs, total_traffic, predict_steps+1, 3]
        predicted_flat = encoded.reshape(num_envs, -1, spatial_dim_encoded) # # [num_envs, total_traffic, (predict_steps+1)*3]
    else:
        predicted_flat = relative_pos.reshape(num_envs, -1, spatial_dim_xy) # [num_envs, total_traffic, (predict_steps+1)*2]
        predicted_flat = predicted_flat / (2 * circle_radius) * observation_norm_scale

    # 添加safety radius
    radius_with_feature_dim = state.traffic.traffic_safety_radius.unsqueeze(-1) #  [total_traffic, 1]
    expanded_radius = radius_with_feature_dim.expand(num_envs, -1, -1) # [num_envs, total_traffic, 1]
    predicted_flat = torch.cat([predicted_flat, expanded_radius], dim=-1) # [num_envs, total_traffic, spatial_dim+1]
    
    # 1. 创建一个包含所有有效数据的基础张量
    #    注意：我们不再需要预先用 'inf' 填充 spatial_edges
    base_spatial_edges = predicted_flat

    # 2. 准备掩码用于广播
    #    in_range_mask 的形状是 [num_envs, total_traffic_num]
    #    我们需要让它能作用于形状为 [num_envs, total_traffic_num, spatial_dim+1] 的张量
    #    所以，我们在最后增加一个维度
    mask_expanded = in_range_mask.unsqueeze(-1) # 形状变为: [num_envs, total_traffic_num, 1]

    # 3. 使用 torch.where() 进行优雅的条件赋值
    spatial_edges = torch.where(
        mask_expanded,
        base_spatial_edges,
        torch.zeros_like(base_spatial_edges)
    )
    # spatial_edges的维度是 [num_envs, current_traffic_num, spatial_dim+1]
    # 但是可能小于total_traffic_num，所以需要cat
    pad_count = total_traffic_num - spatial_edges.shape[1]
    if pad_count > 0:
        fill_spatial_edges = torch.zeros((num_envs, pad_count, spatial_dim_output), device=device)
        spatial_edges = torch.cat([spatial_edges, fill_spatial_edges], dim=1)

    # 生成spatial_types并pad（1=drone, 2=evtol；0=dummy）
    base_types = state.traffic.traffic_types  # [total_traffic]
    if base_types is None or base_types.numel() == 0:
        spatial_types = torch.zeros((num_envs, total_traffic_num), dtype=torch.long, device=device)
    else:
        expanded_types = base_types.view(1, -1).expand(num_envs, -1).to(device=device)
        if pad_count > 0:
            fill_types = torch.zeros((num_envs, pad_count), dtype=torch.long, device=device)
            spatial_types = torch.cat([expanded_types, fill_types], dim=1)
        else:
            spatial_types = expanded_types
    
    # 生成visible_masks并pad
    visible_masks = in_range_mask
    if pad_count > 0:
        fill_masks = torch.zeros((num_envs, pad_count), dtype=visible_masks.dtype, device=device)
        visible_masks = torch.cat([visible_masks, fill_masks], dim=1)
    return {'spatial_edges': spatial_edges, 'visible_masks': visible_masks, 'spatial_types': spatial_types}