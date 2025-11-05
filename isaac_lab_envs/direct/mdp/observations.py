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

def body_to_world(body_xy, cy, sy):
    x = body_xy[..., 0]
    y = body_xy[..., 1]
    num_extra_dims = x.dim() - cy.dim()
    if num_extra_dims > 0:
        cy = cy.view(cy.shape + (1,) * num_extra_dims)
        sy = sy.view(sy.shape + (1,) * num_extra_dims)
    world_x = x * cy - y * sy
    world_y = x * sy + y * cy
    return torch.stack([world_x, world_y], dim=-1)


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
        cfg = self.cfg
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
        observation_norm_scale = getattr(cfg, 'observation_norm_scale', 10.0)
        bounds = getattr(cfg, 'area_bounds', None)
        if bounds is None:
            raise ValueError("area_bounds is not set")
        area_size = max(
            bounds.xmax - bounds.xmin,
            bounds.ymax - bounds.ymin
        )
        circle_radius = area_size / 2.0 * 1.4142135623730951

        device = state.device
        target_pos = state.navigation.target_positions

        robot_pos = state.ego_drone.positions[:, :, :2]  # [N,1,2]
        robot_vel_world = state.ego_drone.velocities[:, :, :2]  # [N,1,2]
        robot_quat = state.ego_drone.rotations # [N,1,4]
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

        robot_radius = torch.full((state.ego_drone.positions.shape[0], 1, 1), cfg.safety_radius, device=device)
        robot_v_pref = torch.full((state.ego_drone.positions.shape[0], 1, 1), cfg.v_pref, device=device)

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

    def get_observation_space(self) -> dict:
        cfg = self.cfg
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
        use_global_path = getattr(cfg, 'use_global_path', False)
        robot_node_dim = 8
        if use_angle_distance_obs:
            robot_node_dim += 1 # 1 dim for relative target goal
        if use_global_path:
            robot_node_dim += 4 # 4 dim for relative local goal and projection point
            if use_angle_distance_obs:
                robot_node_dim += 2 # 2 dim for relative local goal and projection point

        return {'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, robot_node_dim), dtype=np.float32)}


class TrafficStateObservationModule(ObservationModule):
    # output base traffic state, relative position, velocity, safety radius, type
    def __init__(self, cfg):
        super().__init__(cfg)
        drone_num = getattr(cfg.traffic_sim, 'num_drones', 0)
        evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
        self.total_traffic_num = max(drone_num + evtol_num, 20)
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
        state_output_dim = 5
        if use_angle_distance_obs:
            state_output_dim += 1 # 1 dim for relative position
        self.state_output_dim = state_output_dim
    def process_observation(self, state: EnvState) -> dict:
        cfg = self.cfg
        device = state.device
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
        sensor_range = getattr(cfg, 'observation_radius', 1000.0)

        # Ego
        robot_pos = state.ego_drone.positions[:, :, :2]          # [N,1,2]
        # Traffic
        total_traffic_num = self.total_traffic_num
        state_output_dim = self.state_output_dim
        if state.traffic is None:
            # empty outputs

            zeros_states = torch.zeros((state.num_envs, total_traffic_num, state_output_dim), device=device)
            zeros_mask = torch.zeros((state.num_envs, total_traffic_num), device=device)
            zeros_types = torch.zeros((state.num_envs, total_traffic_num), dtype=torch.long, device=device)
            return {'traffic_states': zeros_states, 'visible_masks': zeros_mask, 'traffic_types': zeros_types}

        traffic_pos = state.traffic.traffic_positions  # [T,3]
        traffic_vel = state.traffic.traffic_velocities # [T,3]
        traffic_rad = state.traffic.traffic_safety_radius  # [T]
        traffic_types = state.traffic.traffic_types        # [T]

        if (traffic_pos is None or traffic_pos.numel() == 0 or
            traffic_vel is None or traffic_vel.numel() == 0 or
            traffic_rad is None or traffic_rad.numel() == 0 or
            traffic_types is None or traffic_types.numel() == 0):

            zeros_states = torch.zeros((state.num_envs, total_traffic_num, state_output_dim), device=device)
            zeros_mask = torch.zeros((state.num_envs, total_traffic_num), device=device)
            zeros_types = torch.zeros((state.num_envs, total_traffic_num), dtype=torch.long, device=device)
            return {'traffic_states': zeros_states, 'visible_masks': zeros_mask, 'traffic_types': zeros_types}

        # Shapes
        T = traffic_pos.shape[0]
        N = state.num_envs
        traffic_pos_2d = traffic_pos[:, :2]  # [T,2]
        traffic_vel_2d = traffic_vel[:, :2]  # [T,2]

        # Relative to each env ego (broadcast)
        rel_pos = traffic_pos_2d.unsqueeze(0) - robot_pos  # [N, T, 2]
        robot_quat = state.ego_drone.rotations # [N,1,4]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)
        rel_pos = world_to_body(rel_pos, cy, sy)

        # visible mask by range
        dists = torch.norm(rel_pos, dim=-1)  # [N, T]
        visible_masks = (dists <= sensor_range)

        # Assemble traffic_states per agent: [rel_x, rel_y, vel_x, vel_y, radius] (+ inv_dist optional)
        radii = traffic_rad.view(1, -1, 1).expand(N, T, 1)  # [N,T,1]
        base = torch.cat([rel_pos, traffic_vel_2d.unsqueeze(0).expand(N, -1, -1), radii], dim=-1)  # [N,T,5]
        if use_angle_distance_obs:
            inv_dist = (1.0 / (dists + 1.0)).unsqueeze(-1)  # [N,T,1]
            traffic_states = torch.cat([base, inv_dist], dim=-1)  # [N,T,6]
        else:
            traffic_states = base  # [N,T,5]


        cur_T = traffic_states.shape[1]
        if cur_T < total_traffic_num:
            pad_states = torch.zeros((N, total_traffic_num - cur_T, traffic_states.shape[-1]), device=device)
            traffic_states = torch.cat([traffic_states, pad_states], dim=1)
            pad_mask = torch.zeros((N, total_traffic_num - cur_T), dtype=visible_masks.dtype, device=device)
            visible_masks = torch.cat([visible_masks, pad_mask], dim=1)
            types_expanded = traffic_types.view(1, -1).expand(N, -1)
            pad_types = torch.zeros((N, total_traffic_num - cur_T), dtype=torch.long, device=device)
            traffic_types_full = torch.cat([types_expanded, pad_types], dim=1)
        else:
            traffic_states = traffic_states[:, :total_traffic_num]
            visible_masks = visible_masks[:, :total_traffic_num]
            traffic_types_full = traffic_types.view(1, -1).expand(N, -1)[:, :total_traffic_num]

        return {
            'traffic_states': traffic_states,
            'visible_masks': visible_masks,
            'traffic_types': traffic_types_full.to(dtype=torch.long)
        }

    def get_observation_space(self) -> dict:
        cfg = self.cfg

        state_output_dim = self.state_output_dim
        total_traffic_num = self.total_traffic_num
        return {
                'traffic_states': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_traffic_num, state_output_dim), dtype=np.float32),
                'visible_masks': gym.spaces.Box(low=0.0, high=1.0, shape=(total_traffic_num,), dtype=np.float32),
                'traffic_types': gym.spaces.Box(low=0, high=2, shape=(total_traffic_num,), dtype=np.int64),
                }


class TrafficSpatialEdgesObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def process_observation(self, state: EnvState) -> dict:

        """使用缓存的轨迹数据计算空间边观测
        Returns:
            traffic_states: [num_envs, total_traffic_num, spatial_dim]
            visible_masks: [num_envs, total_traffic_num]
            traffic_types: [num_envs, total_traffic_num] (1=drone, 2=evtol；0=dummy)
        """
        cfg = self.cfg
        predict_steps = getattr(cfg, 'predict_steps', 5)
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)
        
        drone_num = cfg.traffic_sim.num_drones
        evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
        total_traffic_num = max(drone_num + evtol_num, 20)
        sensor_range = getattr(cfg, 'observation_radius', 1000)
        sensor_range = 1000
        bounds = cfg.traffic_sim.area_bounds
        area_size = max(
            bounds.xmax - bounds.xmin,
            bounds.ymax - bounds.ymin
        )
        circle_radius = area_size / 2.0 * 1.4142135623730951
        observation_norm_scale = getattr(cfg, 'observation_norm_scale', 10.0)

        num_envs = state.num_envs
        device = state.device

        robot_pos = state.ego_drone.positions[:, :, :2] 
        robot_vel = state.ego_drone.velocities[:, :, :2]  # [N,1,2]
        traffic_future_traj = state.traffic.traffic_future_traj
        robot_quat = state.ego_drone.rotations # [N,1,4]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)

        # helper: rotate world -> body (yaw-only)
        spatial_point_dim = 3 if use_angle_distance_obs else 2 
        spatial_dim_xy = 2 * (predict_steps + 1) 
        spatial_dim_encoded = spatial_point_dim * (predict_steps + 1)
        spatial_dim_output = spatial_point_dim * (predict_steps + 1) + 1 # add 1 dim for safety radius
        
        if traffic_future_traj is None or traffic_future_traj.numel() == 0:
            traffic_states = torch.zeros(
                (num_envs, total_traffic_num, spatial_dim_output), device=device
            )
            visible_masks = torch.zeros((num_envs, total_traffic_num), device=device)
            traffic_types = torch.zeros((num_envs, total_traffic_num), dtype=torch.long, device=device)
            return {'traffic_states': traffic_states, 'visible_masks': visible_masks, 'traffic_types': traffic_types}
        
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
        #    注意：我们不再需要预先用 'inf' 填充 traffic_states
        base_traffic_states = predicted_flat

        # 2. 准备掩码用于广播
        #    in_range_mask 的形状是 [num_envs, total_traffic_num]
        #    我们需要让它能作用于形状为 [num_envs, total_traffic_num, spatial_dim+1] 的张量
        #    所以，我们在最后增加一个维度
        mask_expanded = in_range_mask.unsqueeze(-1) # 形状变为: [num_envs, total_traffic_num, 1]

        # 3. 使用 torch.where() 进行优雅的条件赋值
        traffic_states = torch.where(
            mask_expanded,
            base_traffic_states,
            torch.zeros_like(base_traffic_states)
        )
        # traffic_states的维度是 [num_envs, current_traffic_num, spatial_dim+1]
        # 但是可能小于total_traffic_num，所以需要cat
        pad_count = total_traffic_num - traffic_states.shape[1]
        if pad_count > 0:
            fill_traffic_states = torch.zeros((num_envs, pad_count, spatial_dim_output), device=device)
            traffic_states = torch.cat([traffic_states, fill_traffic_states], dim=1)

        # 生成traffic_types并pad（1=drone, 2=evtol；0=dummy）
        base_types = state.traffic.traffic_types  # [total_traffic]
        if base_types is None or base_types.numel() == 0:
            traffic_types = torch.zeros((num_envs, total_traffic_num), dtype=torch.long, device=device)
        else:
            expanded_types = base_types.view(1, -1).expand(num_envs, -1).to(device=device)
            if pad_count > 0:
                fill_types = torch.zeros((num_envs, pad_count), dtype=torch.long, device=device)
                traffic_types = torch.cat([expanded_types, fill_types], dim=1)
            else:
                traffic_types = expanded_types
        
        # 生成visible_masks并pad
        visible_masks = in_range_mask
        if pad_count > 0:
            fill_masks = torch.zeros((num_envs, pad_count), dtype=visible_masks.dtype, device=device)
            visible_masks = torch.cat([visible_masks, fill_masks], dim=1)
        return {'traffic_states': traffic_states, 'visible_masks': visible_masks, 'traffic_types': traffic_types}

    def get_observation_space(self) -> dict:
        cfg = self.cfg
        predict_steps = getattr(cfg, 'predict_steps', 5)
        use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', True)

        spatial_point_dim = 3 if use_angle_distance_obs else 2 
        spatial_dim_output = spatial_point_dim * (predict_steps + 1) + 1 # add 1 dim for safety radius
        
        drone_num = cfg.traffic_sim.num_drones
        evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
        total_traffic_num = max(drone_num + evtol_num, 20)
        return {
                'traffic_states': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_traffic_num, spatial_dim_output), dtype=np.float32),
                'visible_masks': gym.spaces.Box(low=0.0, high=1.0, shape=(total_traffic_num,), dtype=np.float32),
                'traffic_types': gym.spaces.Box(low=0, high=2, shape=(total_traffic_num,), dtype=np.int64),
                }

class LidarObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def process_observation(self, state: EnvState) -> dict:
        cfg = self.cfg
        lidar_range = getattr(cfg, 'lidar_range', 10.0)
        lidar_resolution = getattr(cfg, 'lidar_resolution', (36, 4))
        if state.perception.lidar_scan is not None:
            lidar_scan = state.perception.lidar_scan.clamp(min=0.0, max=lidar_range) / lidar_range
        else:
            h, w = lidar_resolution
            lidar_scan = torch.zeros(state.num_envs, 1, h, w, device=state.device)
        return {'lidar': lidar_scan}


    def get_observation_space(self) -> dict:
        cfg = self.cfg
        lidar_resolution = getattr(cfg, 'lidar_resolution', (36, 4))
        return {'lidar': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, lidar_resolution[0], lidar_resolution[1]), dtype=np.float32)}



class DynamicObstacleObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dynamic_obstacle_dim = 6
        self.dynamic_obstacle_num = getattr(cfg, 'dynamic_obstacle_num', 5)
        # 固定使用角度+距离编码，输出维度恒为6: [sin(theta), cos(theta), 1/(d+1), vel_x, vel_y, radius]
    def process_observation(self, state: EnvState) -> dict:
        device = state.device
        K = int(self.dynamic_obstacle_num)

        # Ego
        robot_pos = state.ego_drone.positions[:, :, :2]          # [N,1,2]
        robot_quat = state.ego_drone.rotations # [N,1,4]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)

        # Traffic empty -> zeros
        if state.traffic is None:
            return {'traffic_states': torch.zeros((state.num_envs, 1, K, self.dynamic_obstacle_dim), device=device)}

        traffic_pos = state.traffic.traffic_positions  # [T,3]
        traffic_vel = state.traffic.traffic_velocities # [T,3]
        traffic_rad = state.traffic.traffic_safety_radius  # [T]
        if (traffic_pos is None or traffic_pos.numel() == 0 or
            traffic_vel is None or traffic_vel.numel() == 0 or
            traffic_rad is None or traffic_rad.numel() == 0):
            return {'traffic_states': torch.zeros((state.num_envs, 1, K, self.dynamic_obstacle_dim), device=device)}

        # Shapes and relative/body-frame features
        N = state.num_envs
        traffic_pos_2d = traffic_pos[:, :2]  # [T,2]
        traffic_vel_2d = traffic_vel[:, :2]  # [T,2]
        rel_pos_world = traffic_pos_2d.unsqueeze(0) - robot_pos  # [N,T,2]
        rel_pos_body = world_to_body(rel_pos_world, cy, sy)      # [N,T,2]
        dists = torch.norm(rel_pos_body, dim=-1)                 # [N,T]

        # Encode relative pos -> (sin, cos, 1/(d+1))
        theta = torch.atan2(rel_pos_body[..., 1], rel_pos_body[..., 0])
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)
        inv_dist = (1.0 / (dists + 1.0)).unsqueeze(-1)

        vel_exp = traffic_vel_2d.unsqueeze(0).expand(N, -1, -1)  # [N,T,2]
        rad_exp = traffic_rad.view(1, -1, 1).expand(N, -1, 1)    # [N,T,1]
        feats = torch.cat([sin_theta, cos_theta, inv_dist, vel_exp, rad_exp], dim=-1)  # [N,T,6]

        # Sort by distance and pick top-K
        if feats.shape[1] > 0:
            idx = torch.argsort(dists, dim=1)  # [N,T]
            if idx.shape[1] >= K:
                idx_k = idx[:, :K]
            else:
                idx_k = idx
            gathered = torch.gather(feats, 1, idx_k.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))  # [N,K',6]
            if gathered.shape[1] < K:
                pad = torch.zeros((N, K - gathered.shape[1], feats.shape[-1]), device=device)
                gathered = torch.cat([gathered, pad], dim=1)
        else:
            gathered = torch.zeros((N, K, feats.shape[-1]), device=device)

        dynamic_obs = gathered.unsqueeze(1)  # [N,1,K,6]
        return {'traffic_states': dynamic_obs}

    def get_observation_space(self) -> dict:

        return {
                'traffic_states': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.dynamic_obstacle_num, self.dynamic_obstacle_dim), dtype=np.float32),
                }

class DrlvoObservationModule(ObservationModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs

        # DRL-VO 网格参数
        self.grid_size = (20, 20)
        self.grid_res = 4.0  # meters / cell
        self.grid_shape_m = (80.0, 80.0)  # (x_range, y_range)

        # DRL-VO 激光雷达历史参数 [cite: 205, 217]
        self.lidar_history_len = 10  # 10 帧 @ 20Hz = 0.5s
        
        # 假设激光雷达点数
        # 论文 [cite: 445] 提到 1080 个点 (UTM-30LX)。
        # `cnn_data_pub.py` 使用 720 个点。
        # `custom_cnn_full.py` 期望 6400 个值 (80*80)。
        
        # 论文中最合理的解释是 [cite: 212] "min+avg pooling" 
        # 我们假设有 3200 个激光点。
        # (Min(scans_over_time) [3200] + Avg(scans_over_time) [3200]) = 6400 -> 80x80
        self.lidar_points = getattr(cfg, "lidar_points", 3200)
        
        # 初始化激光雷达历史缓冲区
        self.lidar_history_buffer = torch.zeros(
            (self.num_envs, self.lidar_history_len, self.lidar_points),
            device=self.device
        )
        
        # 用于行人网格“splatting”的坐标（处理半径）
        # 我们将为每个行人创建一个小的核，而不是循环
        self.max_radius_cells = int(np.ceil(cfg.traffic_sim.evtol.safety_radius / self.grid_res))
        splat_range = torch.arange(-self.max_radius_cells, self.max_radius_cells + 1, device=self.device)
        splat_xx, splat_yy = torch.meshgrid(splat_range, splat_range, indexing="ij")
        self.splat_kernel_indices = torch.stack([splat_xx, splat_yy], dim=-1).view(1, 1, -1, 2)  # [1, 1, K*K, 2]
        self.splat_kernel_size = self.splat_kernel_indices.shape[2]

    def get_observation_space(self) -> dict:
        policy_space_dict = {
            # 论文中 CNN 的输入是 3 通道 (2 ped + 1 lidar) [cite: 192, 175, 178]
            "ped_map": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            )
        }        
        return policy_space_dict

    def process_observation(self, state: EnvState) -> dict:
        # 1. 计算行人/交通图 (ped_map)
        ped_map = self._compute_ped_map(state)  # [N, 2, 80, 80]
        # visualize_ped_map(ped_map=ped_map, vis_idx=5065)
        
        # 2. 计算激光雷达图 (scan_map)
        # scan_map = self._compute_scan_map(state)  # [N, 1, 80, 80]
        
        # 3. 计算向量输入 (subgoal, robot_vel), #由robot node module解决吧
        
        # 4. 组合 CNN 输入
        # 论文 [cite: 192] (图2) 和代码 确认
        # 3个通道被拼接 (cat) 在一起
        # cnn_input = torch.cat([ped_map, scan_map], dim=1)  # [N, 3, 80, 80]
        
        policy_obs = {
            "ped_map": ped_map,
        }
        
        return policy_obs

    def _compute_ped_map(self, state: EnvState) -> torch.Tensor:
        """
        并行计算行人速度图 ( $2 \times 80 \times 80$ )。
        """
        num_envs = state.num_envs
        # 假设 traffic.positions 是全局的 [total_traffic, 3]
        # 并且 EnvState 提供了所有 traffic 的信息
        if state.traffic.traffic_positions.numel() == 0:
            return torch.zeros((num_envs, 2, self.grid_size[0], self.grid_size[1]), device=self.device)

        # 提取 ego 状态 (世界坐标系)
        robot_pos = state.ego_drone.drone_state[:, :, :2]  # [N, 1, 2]
        robot_quat = state.ego_drone.rotations # [N,1,4]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)
        
        # 提取 traffic 状态 (世界坐标系)
        # 扩展为 [1, M, 2] 以便与 [N, 1, 2] 广播
        traffic_pos = state.traffic.traffic_positions[:, :2].unsqueeze(0)    # [1, M, 2]
        traffic_vel = state.traffic.traffic_velocities[:, :2].unsqueeze(0)  # [1, M, 2]
        traffic_radius = state.traffic.traffic_safety_radius.unsqueeze(0) # [1, M]

        # 1. 转换到机器人局部坐标系
        # 1.1 相对位置 (世界系)
        rel_pos_world = traffic_pos - robot_pos  # [N, M, 2]
        # 1.2 相对位置 (机器人系)
        rel_pos_robot = world_to_body(rel_pos_world, cy, sy) # [N, M, 2]
        # 1.3 绝对速度 (机器人系)
        vel_robot = world_to_body(traffic_vel, cy, sy)  # [N, M, 2]

        # 2. 过滤在 DRL-VO 观测区外的 traffic
        x_local = rel_pos_robot[..., 0]
        y_local = rel_pos_robot[..., 1]
        
        mask = (x_local >= 0) & (x_local < self.grid_shape_m[0]) & \
               (y_local >= -self.grid_shape_m[1] / 2) & (y_local < self.grid_shape_m[1] / 2)
        # mask shape: [N, M]

        # 3. 计算网格索引
        # (r, c) 是网格坐标 (row, col)
        # r 对应 x (前方), c 对应 y (侧方)
        #
        r_center = (x_local / self.grid_res).floor()  # [N, M]
        c_center = (-(y_local - (self.grid_shape_m[1] / 2)) / self.grid_res).floor() # [N, M]
        
        # --- 处理行人半径 (你要求的功能) ---
        # 我们将“splat” (扩展) 每个行人以覆盖其半径
        radius_cells = (traffic_radius / self.grid_res).ceil().long() # [1, M]
        
        # 确保 radius_cells 不超过我们的 splatting 核大小
        radius_cells = torch.clamp(radius_cells, 0, self.max_radius_cells) # [1, M]
        
        # 为每个行人创建其 splatting 索引
        # [N, M, 1, 2] + [1, 1, K*K, 2] -> [N, M, K*K, 2]
        center_indices = torch.stack([r_center, c_center], dim=-1).unsqueeze(2)
        splat_indices = center_indices + self.splat_kernel_indices
        
        # [N, M, K*K, 2]
        
        # 过滤掉核中超出半径的单元
        # [1, 1, K*K]
        kernel_dist_sq = self.splat_kernel_indices[..., 0]**2 + self.splat_kernel_indices[..., 1]**2
        # [1, M, 1]
        radius_cells_sq = radius_cells.float().square().unsqueeze(-1)
        
        # [N, M, K*K]
        radius_mask = (kernel_dist_sq <= radius_cells_sq).expand(num_envs, -1, -1)

        # 4. 展平以便使用 scatter (或 index_put_)
        
        # 结合观测区域掩码和半径掩码
        final_mask = mask.unsqueeze(-1) & radius_mask # [N, M, K*K]

        # 展平所有东西
        
        # 批次索引 [N] -> [N, 1, 1] -> [N, M, K*K]
        batch_idx = torch.arange(num_envs, device=self.device).view(num_envs, 1, 1).expand(-1, mask.shape[1], self.splat_kernel_size)
        
        flat_batch = batch_idx[final_mask] # [Num_Valid_Cells]
        flat_r = splat_indices[..., 0][final_mask].long()
        flat_c = splat_indices[..., 1][final_mask].long()

        # 裁剪索引到网格边界 [0, 79]
        flat_r = torch.clamp(flat_r, 0, self.grid_size[0] - 1)
        flat_c = torch.clamp(flat_c, 0, self.grid_size[1] - 1)

        # 准备速度值
        vx_vals = vel_robot[..., 0].unsqueeze(-1).expand(-1, -1, self.splat_kernel_size) # [N, M, K*K]
        vy_vals = vel_robot[..., 1].unsqueeze(-1).expand(-1, -1, self.splat_kernel_size) # [N, M, K*K]
        
        flat_vx = vx_vals[final_mask]
        flat_vy = vy_vals[final_mask]

        # 5. 写入网格 (使用 index_put_ 以便并行)
        # 这种方法用最后一个写入的值覆盖 (与 DRL-VO 的原始实现行为一致)
        ped_map_vx = torch.zeros((num_envs, self.grid_size[0], self.grid_size[1]), device=self.device)
        ped_map_vy = torch.zeros((num_envs, self.grid_size[0], self.grid_size[1]), device=self.device)
        
        ped_map_vx[flat_batch, flat_r, flat_c] = flat_vx
        ped_map_vy[flat_batch, flat_r, flat_c] = flat_vy

        return torch.stack([ped_map_vx, ped_map_vy], dim=1) # [N, 2, 80, 80]



# -------------------- Modular Observation manager --------------------
@configclass
class ObservationManagerCfg:
    modules: list[str] = ("robot_node",)


OBSERVATION_MODULES: dict[str, type[ObservationModule]] = {
    "robot_node": RobotNodeObservationModule,
    "traffic_state": TrafficStateObservationModule,
    "traffic_spatial_state": TrafficSpatialEdgesObservationModule,
    "lidar": LidarObservationModule,
    "dynamic_obstacle": DynamicObstacleObservationModule,
    "drlvo": DrlvoObservationModule,
}


class ObservationManager:
    def __init__(self, env_cfg, manager_cfg: ObservationManagerCfg | None = None, device: str = "cuda"):
        self.env_cfg = env_cfg
        self.device = device
        self.manager_cfg = manager_cfg or ObservationManagerCfg()
        self.modules: list[ObservationModule] = []
        for name in self.manager_cfg.modules:
            mod_cls = OBSERVATION_MODULES.get(name)
            if mod_cls is None:
                continue

            mod = mod_cls(env_cfg)
            mod.device = device
            self.modules.append(mod)

    def process_observation(self, state: EnvState) -> dict:
        obs = {}
        for mod in self.modules:

            out = mod.process_observation(state)
            if out is not None:
                obs.update(out)

        return {"policy": obs}

    def generate_policy_obs_dict(self) -> dict:
        spaces = {}
        for mod in self.modules:
            sp = mod.get_observation_space()
            if sp is not None:
                spaces.update(sp)

        return spaces



