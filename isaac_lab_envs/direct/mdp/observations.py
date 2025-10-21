import torch
from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
import numpy as np
import gymnasium as gym
from isaac_lab_envs.direct.mdp.state import EnvState

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)

class NavObservationProcessor:
    """基础导航环境的观测处理器"""
    
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        
    def generate_policy_obs_dict(self):
        """生成观测空间字典"""
        policy_space_dict = {
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32),
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32)
        }
        return policy_space_dict
    
    def process_observation(self, state: EnvState) -> dict:
        """处理基础导航观测
        
        Args:
            state: 环境状态对象
            
        Returns:
            观测字典
        """
        # 从状态对象提取数据
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        
        # 提取位置和速度
        robot_pos = drone_state[:, :, :2]  # [num_envs, 1, 2] (x, y)
        robot_vel = drone_state[:, :, 7:9]  # [num_envs, 1, 2] (vx, vy)
        
        # 计算相对目标位置
        goal_pos = target_pos[:, :, :2]  # [num_envs, 1, 2] 只取x,y
        relative_goal_pos = goal_pos - robot_pos  # [num_envs, 1, 2]
        
        # 计算速度方向yaw
        robot_yaw = torch.atan2(robot_vel[:, :, 1], robot_vel[:, :, 0])  # [num_envs, 1]
        
        # 机器人参数
        robot_radius = torch.full((drone_state.shape[0], 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((drone_state.shape[0], 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建robot_node: [rel_goal_x, rel_goal_y, robot_radius, robot_v_pref, robot_yaw]
        robot_node = torch.cat([
            relative_goal_pos,  # [num_envs, 1, 2]
            robot_radius,       # [num_envs, 1, 1]  
            robot_v_pref,       # [num_envs, 1, 1]
            robot_yaw.unsqueeze(-1)  # [num_envs, 1, 1]
        ], dim=-1)  # [num_envs, 1, 5]

        policy_obs = {
            'robot_node': robot_node,
            'temporal_edges': robot_vel
        }
        
        return {"policy": policy_obs}

class NavObservationProcessorWithPath(NavObservationProcessor):
    """支持局部目标的导航观测处理器"""
    
    def generate_policy_obs_dict(self):
        """生成观测空间字典 - 增加local goal和projection point维度"""
        policy_space_dict = {
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 9), dtype=np.float32),  # 5+2+2=9
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32)
        }
        return policy_space_dict
    
    def process_observation(self, state: EnvState) -> dict:
        """处理包含局部目标的观测
        
        Args:
            state: 环境状态对象
            
        Returns:
            观测字典
        """
        # 从状态对象提取数据
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        local_goals = state.navigation.local_goals
        projection_points = state.navigation.projection_points
        
        # 提取位置和速度
        robot_pos = drone_state[:, :, :2]  # [num_envs, 1, 2] (x, y)
        robot_vel = drone_state[:, :, 7:9]  # [num_envs, 1, 2] (vx, vy)
        
        # 计算相对目标位置
        goal_pos = target_pos[:, :, :2]  # [num_envs, 1, 2] 只取x,y
        relative_goal_pos = goal_pos - robot_pos  # [num_envs, 1, 2]
        
        # 计算相对局部目标位置
        local_goal_pos = local_goals[:, :, :2] if local_goals is not None else goal_pos  # [num_envs, 1, 2]
        relative_local_goal_pos = local_goal_pos - robot_pos  # [num_envs, 1, 2]
        
        # 计算相对投影点位置
        projection_pos = projection_points[:, :, :2] if projection_points is not None else robot_pos  # [num_envs, 1, 2]
        relative_projection_pos = projection_pos - robot_pos  # [num_envs, 1, 2]
        
        # 计算速度方向yaw
        robot_yaw = torch.atan2(robot_vel[:, :, 1], robot_vel[:, :, 0])  # [num_envs, 1]
        
        # 机器人参数
        robot_radius = torch.full((drone_state.shape[0], 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((drone_state.shape[0], 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建扩展的robot_node: [rel_goal_x, rel_goal_y, robot_radius, robot_v_pref, robot_yaw, rel_local_goal_x, rel_local_goal_y, rel_proj_x, rel_proj_y]
        robot_node = torch.cat([
            relative_goal_pos,           # [num_envs, 1, 2] - 前2个维度
            robot_radius,                # [num_envs, 1, 1] - 第3个维度
            robot_v_pref,                # [num_envs, 1, 1] - 第4个维度
            robot_yaw.unsqueeze(-1),     # [num_envs, 1, 1] - 第5个维度
            relative_local_goal_pos,     # [num_envs, 1, 2] - 第6-7个维度
            relative_projection_pos,     # [num_envs, 1, 2] - 第8-9个维度
        ], dim=-1)  # [num_envs, 1, 9]

        policy_obs = {
            'robot_node': robot_node,
            'temporal_edges': robot_vel
        }
        
        return {"policy": policy_obs}


class TrafficObservationProcessor:
    """Traffic环境的观测处理器，基于Isaac Lab tensor操作优化"""
    
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.predict_steps = cfg.predict_steps
        self.pred_timestep = cfg.pred_timestep
        self.observation_norm_scale = cfg.observation_norm_scale
        # 是否使用角度+距离编码: (sin(theta), cos(theta), 1/(d+1))
        self.use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', False)
        # Traffic aircraft数量
        # self.total_traffic_num = cfg.traffic_sim.num_drones + getattr(cfg.traffic_sim, 'num_evtols', 0)
        self.drone_num = cfg.traffic_sim.num_drones
        self.evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
        self.total_traffic_num = max(self.drone_num+self.evtol_num, 20)
        if (self.drone_num == 0):
            self.total_traffic_num = self.drone_num + self.evtol_num
        
        # 传感器感知范围（使用NavEnv的observation_radius）
        self.sensor_range = cfg.observation_radius
        # 单步空间特征维度：旧模式为2，新模式为3
        self.spatial_point_dim = 3 if self.use_angle_distance_obs else 2
        self.spatial_dim = self.spatial_point_dim * (self.predict_steps + 1) + 1 

        self.robot_node_dim = 6
        if self.use_angle_distance_obs:
            self.robot_node_dim += 1
        
        self.temporal_edges_dim = 2
        
        # 用于归一化的参数  
        if hasattr(cfg.traffic_sim.area_bounds, 'xmax'):
            self.area_size = max(
                cfg.traffic_sim.area_bounds.xmax - cfg.traffic_sim.area_bounds.xmin,
                cfg.traffic_sim.area_bounds.ymax - cfg.traffic_sim.area_bounds.ymin
            )
        else:
            # 如果是字典格式
            bounds = cfg.traffic_sim.area_bounds
            self.area_size = max(
                bounds['xmax'] - bounds['xmin'],
                bounds['ymax'] - bounds['ymin']
            )
        self.circle_radius = self.area_size / 2.0 * 1.4142135623730951 # sqrt(2)
        # 预测轨迹缓存
        self.traffic_future_traj = None
        self.traffic_positions = None
        self.traffic_velocities = None
        self.traffic_types = None
        self.traffic_safety_radius = None

    def _encode_relative_xy(self, relative_xy: torch.Tensor) -> torch.Tensor:
        """将相对位置编码为所需表示。
        
        - 旧模式: 直接对 (dx, dy) 做归一化
        - 新模式: 输出 (sin(theta), cos(theta), 1/(d+1))
        
        Args:
            relative_xy: [..., 2]
        Returns:
            编码后的张量: [..., 2] 或 [..., 3]
        """
        if not self.use_angle_distance_obs:
            return relative_xy / (2 * self.circle_radius) * self.observation_norm_scale
        # 新模式
        dx = relative_xy[..., 0]
        dy = relative_xy[..., 1]
        theta = torch.atan2(dy, dx)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        dist = torch.norm(relative_xy, dim=-1)
        inv_dist = 1.0 / (dist + 1.0)
        return torch.stack([sin_theta, cos_theta, inv_dist], dim=-1)

    def predict_traffic_trajectory(self, traffic_positions: torch.Tensor, traffic_velocities: torch.Tensor, traffic_types: torch.Tensor, traffic_safety_radius: torch.Tensor):
        """预计算traffic轨迹预测，供后续观测和奖励计算使用
        
        Args:
            traffic_positions: [total_traffic, 3] traffic位置
            traffic_velocities: [total_traffic, 3] traffic速度
            traffic_types: [total_traffic] traffic类型（1=drone, 2=evtol；0=dummy）
        """
        self.traffic_positions = traffic_positions
        self.traffic_velocities = traffic_velocities
        self.traffic_types = traffic_types
        self.traffic_safety_radius = traffic_safety_radius
        if traffic_positions.numel() == 0:
            self.traffic_future_traj = torch.empty(0, self.predict_steps + 1, 3, device=self.device)
            return
            
        # 使用恒速模型预测轨迹
        total_traffic = traffic_positions.shape[0]

        self.traffic_future_traj = torch.zeros(total_traffic, self.predict_steps + 1, 3, device=self.device)
        
        # 第0步是当前位置
        self.traffic_future_traj[:, 0] = traffic_positions
        
        # 恒速预测后续位置
        for step in range(1, self.predict_steps + 1):
            self.traffic_future_traj[:, step] = (
                traffic_positions + traffic_velocities * step * self.pred_timestep
            )

    def generate_policy_obs_dict(self):
        # 计算traffic数量配置
        total_traffic_num = self.total_traffic_num
        
        # 1. 创建内层字典 "policy" 的内容
        policy_space_dict = {
            # robot_node维度: 相对目标(2或3) + 安全半径1 + v_pref 1 + yaw 1
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.robot_node_dim), dtype=np.float32),
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.temporal_edges_dim), dtype=np.float32),
            'spatial_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_traffic_num, self.spatial_dim), dtype=np.float32),
            'visible_masks': gym.spaces.Box(low=0.0, high=1.0, shape=(total_traffic_num,), dtype=np.float32),
            'spatial_types': gym.spaces.Box(low=0, high=2, shape=(total_traffic_num,), dtype=np.int64),
        }
        return policy_space_dict

    def process_observation(self, state: EnvState) -> dict:
        """处理观测数据（使用预计算的轨迹）
        
        Args:
            state: 环境状态对象
            
        Returns:
            观测字典
        """
        # 从状态对象提取数据
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        num_envs = drone_state.shape[0]
        
        # 提取ego drone信息
        robot_pos = drone_state[:, :, :2]  # [num_envs, 1, 2] (x, y)
        robot_vel = drone_state[:, :, 7:9]  # [num_envs, 1, 2] (vx, vy)
        # norm_robot_vel = robot_vel
        
        # 计算相对目标位置并编码
        goal_pos = target_pos[:, :, :2]  # [num_envs, 1, 2] 只取x,y
        relative_goal_pos = goal_pos - robot_pos  # [num_envs, 1, 2]
        encoded_goal = self._encode_relative_xy(relative_goal_pos)
        
        # 计算速度方向yaw
        robot_yaw = torch.atan2(robot_vel[:, :, 1], robot_vel[:, :, 0])  # [num_envs, 1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)
        
        # 机器人参数
        robot_radius = torch.full((num_envs, 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((num_envs, 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建robot_node: [rel_goal(2/3), robot_radius, robot_v_pref, robot_yaw]
        robot_node = torch.cat([
            encoded_goal,       # [num_envs, 1, 2/3]
            robot_radius,       # [num_envs, 1, 1]  
            robot_v_pref,       # [num_envs, 1, 1]
            cy.unsqueeze(-1),  # [num_envs, 1, 1]
            sy.unsqueeze(-1)  # [num_envs, 1, 1]
        ], dim=-1)
        
        # 计算空间边观测（使用预计算的轨迹）
        spatial_edges, visible_masks, spatial_types = self._compute_spatial_edges_from_cache(robot_pos, robot_vel)
        
        
        # 构建观测字典
        policy_obs = {
            'robot_node': robot_node,
            'temporal_edges': robot_vel,  # [num_envs, 1, 2]
            'spatial_edges': spatial_edges,
            'visible_masks': visible_masks,
            'spatial_types': spatial_types,
        }
        
        return {"policy": policy_obs}

    def _compute_spatial_edges_from_cache(self, robot_pos: torch.Tensor, robot_vel: torch.Tensor) -> torch.Tensor:
        """使用缓存的轨迹数据计算空间边观测
        
        Args:
            robot_pos: [num_envs, 1, 2] ego位置
            robot_vel: [num_envs, 1, 2] ego速度
            
        Returns:
            spatial_edges: [num_envs, total_traffic_num, spatial_dim]
            visible_masks: [num_envs, total_traffic_num]
            spatial_types: [num_envs, total_traffic_num] (1=drone, 2=evtol；0=dummy)
        """
        num_envs = robot_pos.shape[0]
        spatial_dim_xy = 2 * (self.predict_steps + 1)
        spatial_dim_encoded = self.spatial_point_dim * (self.predict_steps + 1)
        
        if self.traffic_future_traj is None or self.traffic_future_traj.numel() == 0:
            spatial_edges = torch.zeros(
                (num_envs, self.total_traffic_num, self.spatial_dim), device=self.device
            )
            visible_masks = torch.zeros((num_envs, self.total_traffic_num), device=self.device)
            spatial_types = torch.zeros((num_envs, self.total_traffic_num), dtype=torch.long, device=self.device)
            return spatial_edges, visible_masks, spatial_types
        
        # 计算相对位置 [num_envs, total_traffic, predict_steps+1, 2]
        traffic_pos_2d = self.traffic_future_traj[:, :, :2]  # [total_traffic, predict_steps+1, 2]
        robot_pos_expanded = robot_pos.unsqueeze(1)  # [num_envs, 1, 1, 2]
        
        relative_pos = traffic_pos_2d.unsqueeze(0) - robot_pos_expanded  # [num_envs, total_traffic, predict_steps+1, 2]
        
        # 计算距离用于感知范围过滤
        current_distances = torch.norm(relative_pos[:, :, 0], dim=-1)  # [num_envs, total_traffic]
        self.sensor_range = 1000
        in_range_mask = current_distances <= self.sensor_range
        # 假设都能看到        
        # 展平预测位置为spatial edges格式
        if self.use_angle_distance_obs:
            # 编码为 (sin, cos, 1/(d+1)) 并展平
            encoded = self._encode_relative_xy(relative_pos)  # [num_envs, total_traffic, predict_steps+1, 3]
            predicted_flat = encoded.reshape(num_envs, -1, spatial_dim_encoded)
        else:
            predicted_flat = relative_pos.reshape(num_envs, -1, spatial_dim_xy)
            predicted_flat = predicted_flat / (2 * self.circle_radius) * self.observation_norm_scale

        # 添加safety radius
        radius_with_feature_dim = self.traffic_safety_radius.unsqueeze(-1) #  [total_traffic, 1]
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
        pad_count = self.total_traffic_num - spatial_edges.shape[1]
        if pad_count > 0:
            fill_spatial_edges = torch.zeros((num_envs, pad_count, self.spatial_dim), device=self.device)
            spatial_edges = torch.cat([spatial_edges, fill_spatial_edges], dim=1)

        # 生成spatial_types并pad（1=drone, 2=evtol；0=dummy）
        base_types = self.traffic_types  # [total_traffic]
        if base_types is None or base_types.numel() == 0:
            spatial_types = torch.zeros((num_envs, self.total_traffic_num), dtype=torch.long, device=self.device)
        else:
            expanded_types = base_types.view(1, -1).expand(num_envs, -1).to(device=self.device)
            if pad_count > 0:
                fill_types = torch.zeros((num_envs, pad_count), dtype=torch.long, device=self.device)
                spatial_types = torch.cat([expanded_types, fill_types], dim=1)
            else:
                spatial_types = expanded_types
        
        # 生成visible_masks并pad
        visible_masks = in_range_mask
        if pad_count > 0:
            fill_masks = torch.zeros((num_envs, pad_count), dtype=visible_masks.dtype, device=self.device)
            visible_masks = torch.cat([visible_masks, fill_masks], dim=1)

        return spatial_edges, visible_masks.float(), spatial_types.long()
    




class TrafficObservationProcessorWithPath(TrafficObservationProcessor):
    """支持局部目标的Traffic环境观测处理器"""
    
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        super().__init__(cfg, device)
        # self.robot_node_dim = 3 if self.use_angle_distance_obs else 2 + 3 + (3 if self.use_angle_distance_obs else 2) + (3 if self.use_angle_distance_obs else 2)
        self.robot_node_dim = 10
        if self.use_angle_distance_obs:
            self.robot_node_dim += 3
        self.temporal_edges_dim = 2



    def process_observation(self, state: EnvState) -> dict:
        """处理包含局部目标的观测数据（使用预计算的轨迹）
        
        Args:
            state: 环境状态对象
            
        Returns:
            观测字典
        """
        # 从状态对象提取数据
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        local_goals = state.navigation.local_goals
        projection_points = state.navigation.projection_points
        num_envs = drone_state.shape[0]
        
        # 提取ego drone信息
        robot_pos = drone_state[:, :, :2]  # [num_envs, 1, 2] (x, y)
        robot_vel = drone_state[:, :, 7:9]  # [num_envs, 1, 2] (vx, vy)
        
        # 计算相对目标位置并编码
        goal_pos = target_pos[:, :, :2]  # [num_envs, 1, 2] 只取x,y
        relative_goal_pos = goal_pos - robot_pos  # [num_envs, 1, 2]
        encoded_goal = self._encode_relative_xy(relative_goal_pos)
        
        # 计算相对局部目标位置并编码
        local_goal_pos = local_goals[:, :, :2] if local_goals is not None else goal_pos  # [num_envs, 1, 2]
        relative_local_goal_pos = local_goal_pos - robot_pos  # [num_envs, 1, 2]
        encoded_local_goal = self._encode_relative_xy(relative_local_goal_pos)
        
        # 计算相对投影点位置并编码
        projection_pos = projection_points[:, :, :2] if projection_points is not None else robot_pos  # [num_envs, 1, 2]
        relative_projection_pos = projection_pos - robot_pos  # [num_envs, 1, 2]
        encoded_projection = self._encode_relative_xy(relative_projection_pos)
        
        # 计算速度方向yaw
        robot_yaw = torch.atan2(robot_vel[:, :, 1], robot_vel[:, :, 0])  # [num_envs, 1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)
        
        # 机器人参数
        robot_radius = torch.full((num_envs, 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((num_envs, 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建扩展的robot_node: [goal(2/3), radius, v_pref, yaw, local_goal(2/3), proj(2/3)]
        robot_node = torch.cat([
            encoded_goal,                # [num_envs, 1, 2/3]
            robot_radius,                # [num_envs, 1, 1]  
            robot_v_pref,                # [num_envs, 1, 1]
            cy.unsqueeze(-1),            # [num_envs, 1, 1]
            sy.unsqueeze(-1),            # [num_envs, 1, 1]
            encoded_local_goal,          # [num_envs, 1, 2/3]
            encoded_projection,          # [num_envs, 1, 2/3]
        ], dim=-1)  # [num_envs, 1, 9]
        
        # 计算空间边观测（使用预计算的轨迹）
        spatial_edges, visible_masks, spatial_types = self._compute_spatial_edges_from_cache(robot_pos, robot_vel)
        
        
        # 构建观测字典
        policy_obs = {
            'robot_node': robot_node,
            'temporal_edges': robot_vel,  # [num_envs, 1, 2]
            'spatial_edges': spatial_edges,
            'visible_masks': visible_masks,
            'spatial_types': spatial_types,
        }
        
        
        return {"policy": policy_obs}






class CityNavObservationProcessor:
    """Observation processor for city nav with lidar.

    Outputs a policy dict with:
    - robot_node: [1, 5]
    - temporal_edges: [1, 2]
    - lidar: [1, 36, 4]
    """

    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        # angle+distance encoding switch (sin, cos, 1/(d+1))
        self.use_angle_distance_obs = getattr(cfg, 'use_angle_distance_obs', False)
        self.robot_node_dim = 8
        if self.use_angle_distance_obs:
            self.robot_node_dim += 1

    def _encode_relative_xy(self, relative_xy: torch.Tensor) -> torch.Tensor:
        """Encode relative XY either as normalized (dx, dy) or (sin, cos, 1/(d+1))."""
        if not self.use_angle_distance_obs:
            # normalize by lidar range to [~ -1, 1]
            return relative_xy / max(1e-6, float(self.cfg.lidar_range))
        dx = relative_xy[..., 0]
        dy = relative_xy[..., 1]
        theta = torch.atan2(dy, dx)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        dist = torch.norm(relative_xy, dim=-1)
        inv_dist = 1.0 / (dist + 1.0)
        return torch.stack([sin_theta, cos_theta, inv_dist], dim=-1)

    def generate_policy_obs_dict(self):
        policy_space_dict = {
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.robot_node_dim), dtype=np.float32),
            'lidar': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 36, 4), dtype=np.float32),
        }
        return policy_space_dict

    def process_observation(self, state: EnvState) -> dict:
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
        def world_to_body(rel_xy):
            x = rel_xy[..., 0]
            y = rel_xy[..., 1]
            bx = x * cy + y * sy
            by = -x * sy + y * cy
            return torch.stack([bx, by], dim=-1)

        goal_pos = target_pos[:, :, :2]
        rel_goal_world = goal_pos - robot_pos
        rel_goal_body = world_to_body(rel_goal_world)
        encoded_rel_goal = self._encode_relative_xy(rel_goal_body)
        vel_body = world_to_body(robot_vel_world)

        robot_radius = torch.full((drone_state.shape[0], 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((drone_state.shape[0], 1, 1), self.cfg.v_pref, device=self.device)

        # [rel_goal(2/3), radius(1), v_pref(1), yaw(1), vel_body(2)]
        robot_node = torch.cat([
            encoded_rel_goal,
            robot_radius,
            robot_v_pref,
            encoded_robot_yaw,
            vel_body,
        ], dim=-1)

        # lidar
        if state.perception.lidar_scan is not None:
            lidar_scan = state.perception.lidar_scan/self.cfg.lidar_range
        else:
            lidar_scan = torch.zeros(drone_state.shape[0], 1, 36, 4, device=self.device)

        policy_obs = {
            'robot_node': robot_node,
            'lidar': lidar_scan,
        }
        return {"policy": policy_obs}



class CityNavObservationProcessorWithPath(CityNavObservationProcessor):
    """Observation processor for city nav with lidar and path.

    Outputs a policy dict with:
    - robot_node: [1, 10]
    - lidar: [1, 36, 4]
    - path: [1, 3, 3]
    """

    def __init__(self, cfg, device: str = "cuda"):
        super().__init__(cfg, device)
        self.robot_node_dim = 8+4
        if self.use_angle_distance_obs:
            self.robot_node_dim += 3

    def process_observation(self, state: EnvState) -> dict:
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        local_goals = state.navigation.local_goals
        projection_points = state.navigation.projection_points

        robot_pos = drone_state[:, :, :2]
        robot_vel_world = drone_state[:, :, 7:9]
        robot_quat = drone_state[:, :, 3:7]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]
        cy = torch.cos(robot_yaw)
        sy = torch.sin(robot_yaw)
        encoded_robot_yaw = torch.stack([cy, sy], dim=-1)

        def world_to_body(rel_xy):
            x = rel_xy[..., 0]
            y = rel_xy[..., 1]
            bx = x * cy + y * sy
            by = -x * sy + y * cy
            return torch.stack([bx, by], dim=-1)

        goal_pos = target_pos[:, :, :2]
        rel_goal_body = world_to_body(goal_pos - robot_pos)
        enc_rel_goal = self._encode_relative_xy(rel_goal_body)

        if local_goals is not None:
            rel_local_goal_body = world_to_body(local_goals[:, :, :2] - robot_pos)
        else:
            rel_local_goal_body = torch.zeros_like(rel_goal_body)
        enc_rel_local = self._encode_relative_xy(rel_local_goal_body)

        if projection_points is not None:
            rel_proj_body = world_to_body(projection_points[:, :, :2] - robot_pos)
        else:
            rel_proj_body = torch.zeros_like(rel_goal_body)
        enc_rel_proj = self._encode_relative_xy(rel_proj_body)

        vel_body = world_to_body(robot_vel_world)

        robot_radius = torch.full((drone_state.shape[0], 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((drone_state.shape[0], 1, 1), self.cfg.v_pref, device=self.device)

        robot_node = torch.cat([
            enc_rel_goal,
            robot_radius,
            robot_v_pref,
            encoded_robot_yaw,
            vel_body,
            enc_rel_local,
            enc_rel_proj,
        ], dim=-1)

        # lidar
        if state.perception.lidar_scan is not None:
            lidar_scan = state.perception.lidar_scan / self.cfg.lidar_range
        else:
            lidar_scan = torch.zeros(drone_state.shape[0], 1, 36, 4, device=self.device)

        policy_obs = {
            'robot_node': robot_node,
            'lidar': lidar_scan,
        }
        return {"policy": policy_obs}