import torch
from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
import numpy as np
import gymnasium as gym
from isaac_lab_envs.direct.mdp.state import EnvState



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

        self.robot_node_dim = 5
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
        
        # 机器人参数
        robot_radius = torch.full((num_envs, 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((num_envs, 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建robot_node: [rel_goal(2/3), robot_radius, robot_v_pref, robot_yaw]
        robot_node = torch.cat([
            encoded_goal,       # [num_envs, 1, 2/3]
            robot_radius,       # [num_envs, 1, 1]  
            robot_v_pref,       # [num_envs, 1, 1]
            robot_yaw.unsqueeze(-1)  # [num_envs, 1, 1]
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
        self.robot_node_dim = 9
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
        
        # 机器人参数
        robot_radius = torch.full((num_envs, 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((num_envs, 1, 1), self.cfg.v_pref, device=self.device)
        
        # 构建扩展的robot_node: [goal(2/3), radius, v_pref, yaw, local_goal(2/3), proj(2/3)]
        robot_node = torch.cat([
            encoded_goal,                # [num_envs, 1, 2/3]
            robot_radius,                # [num_envs, 1, 1]  
            robot_v_pref,                # [num_envs, 1, 1]
            robot_yaw.unsqueeze(-1),     # [num_envs, 1, 1]
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



def transform_to_robot_frame_batch(
    world_vectors: torch.Tensor,
    robot_pos_world: torch.Tensor,
    robot_yaw_world: torch.Tensor,
) -> torch.Tensor:
    """
    将世界坐标系下的一批向量 (N, ..., 2) 转换到机器人局部坐标系。
    
    Args:
        world_vectors: 世界坐标系下的向量 [N, ..., 2]。可以是相对位置或绝对速度。
        robot_pos_world: 机器人在世界坐标系下的位置 [N, 1, 2] 或 [N, 2]。
        robot_yaw_world: 机器人在世界坐标系下的朝向 (rad) [N, 1] 或 [N]。
    """
    if robot_pos_world.dim() > world_vectors.dim():
        robot_pos_world = robot_pos_world.squeeze(1)
    if robot_yaw_world.dim() > world_vectors.dim():
        robot_yaw_world = robot_yaw_world.squeeze(1)

    # 1. 计算相对位置（如果输入是绝对位置）
    #    在 DRL-VO 中，我们通常传入的已经是相对位置 (rel_pos) 或绝对速度 (vel)
    #    对于速度，我们只旋转它。对于相对位置，我们也只旋转它。
    #    因此，这个函数假设 world_vectors 要么是 (traffic_pos - robot_pos)，要么是 traffic_vel。
    
    # 2. 旋转
    # 我们要旋转到机器人坐标系，所以使用 -yaw
    yaw = -robot_yaw_world
    
    # 扩展 yaw 以匹配 world_vectors 的维度
    while yaw.dim() < world_vectors.dim():
        yaw = yaw.unsqueeze(-1)
        
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    wx = world_vectors[..., 0]
    wy = world_vectors[..., 1]
    
    rx = cos_yaw * wx - sin_yaw * wy
    ry = sin_yaw * wx + cos_yaw * wy
    
    return torch.stack([rx, ry], dim=-1)


class DrlVoObservationProcessor:
    """
    为 DRL-VO 算法生成观测数据的处理器。
    遵循 Isaac Lab 的并行化（tensor-based）设计。
    生成与 DRL-VO 论文  一致的观测数据。
    """
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs

        # DRL-VO 网格参数
        self.grid_size = (80, 80)
        self.grid_res = 1.0  # meters / cell
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
        self.max_radius_cells = int(np.ceil(cfg.traffic_sim.max_safety_radius / self.grid_res))
        splat_range = torch.arange(-self.max_radius_cells, self.max_radius_cells + 1, device=self.device)
        splat_xx, splat_yy = torch.meshgrid(splat_range, splat_range, indexing="ij")
        self.splat_kernel_indices = torch.stack([splat_xx, splat_yy], dim=-1).view(1, 1, -1, 2)  # [1, 1, K*K, 2]
        self.splat_kernel_size = self.splat_kernel_indices.shape[2]

    def generate_policy_obs_dict(self) -> dict:
        """
        生成 DRL-VO 策略的观测空间字典。
        我们将 `ped_map` 和 `scan_map` 组合成一个 `cnn_input`。
        并将 `subgoal` 和 `robot_vel` 组合成一个 `vector_input`。
        """
        policy_space_dict = {
            # 论文中 CNN 的输入是 3 通道 (2 ped + 1 lidar) [cite: 192, 175, 178]
            "cnn_input": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, self.grid_size[0], self.grid_size[1]), dtype=np.float32
            ),
            # (rel_subgoal_x, rel_subgoal_y) + (robot_vx, robot_vy)
            "vector_input": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
        }
        return policy_space_dict

    def process_observation(self, state: EnvState) -> dict:
        """
        并行处理 num_envs 个环境的观测数据。
        """
        # 1. 计算行人/交通图 (ped_map)
        ped_map = self._compute_ped_map(state)  # [N, 2, 80, 80]
        
        # 2. 计算激光雷达图 (scan_map)
        scan_map = self._compute_scan_map(state)  # [N, 1, 80, 80]
        
        # 3. 计算向量输入 (subgoal, robot_vel)
        vector_input = self._compute_vector_input(state)  # [N, 4]
        
        # 4. 组合 CNN 输入
        # 论文 [cite: 192] (图2) 和代码 确认
        # 3个通道被拼接 (cat) 在一起
        cnn_input = torch.cat([ped_map, scan_map], dim=1)  # [N, 3, 80, 80]
        
        policy_obs = {
            "cnn_input": cnn_input,
            "vector_input": vector_input,
        }
        
        return {"policy": policy_obs}

    def _compute_ped_map(self, state: EnvState) -> torch.Tensor:
        """
        并行计算行人速度图 ( $2 \times 80 \times 80$ )。
        """
        num_envs = self.num_envs
        # 假设 traffic.positions 是全局的 [total_traffic, 3]
        # 并且 EnvState 提供了所有 traffic 的信息
        if state.traffic.positions.numel() == 0:
            return torch.zeros((num_envs, 2, 80, 80), device=self.device)

        # 提取 ego 状态 (世界坐标系)
        robot_pos = state.ego_drone.drone_state[:, :, :2]  # [N, 1, 2]
        robot_yaw = state.ego_drone.drone_state[:, :, 6].unsqueeze(-1)  # [N, 1] (假设第6维是yaw)
        
        # 提取 traffic 状态 (世界坐标系)
        # 扩展为 [1, M, 2] 以便与 [N, 1, 2] 广播
        traffic_pos = state.traffic.positions[:, :2].unsqueeze(0)    # [1, M, 2]
        traffic_vel = state.traffic.velocities[:, :2].unsqueeze(0)  # [1, M, 2]
        traffic_radius = state.traffic.safety_radius.unsqueeze(0) # [1, M]

        # 1. 转换到机器人局部坐标系
        # 1.1 相对位置 (世界系)
        rel_pos_world = traffic_pos - robot_pos  # [N, M, 2]
        # 1.2 相对位置 (机器人系)
        rel_pos_robot = transform_to_robot_frame_batch(rel_pos_world, 
                                                       torch.zeros_like(robot_pos), 
                                                       robot_yaw)  # [N, M, 2]
        # 1.3 绝对速度 (机器人系)
        vel_robot = transform_to_robot_frame_batch(traffic_vel, 
                                                   torch.zeros_like(robot_pos), 
                                                   robot_yaw)  # [N, M, 2]

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
        ped_map_vx = torch.zeros((num_envs, 80, 80), device=self.device)
        ped_map_vy = torch.zeros((num_envs, 80, 80), device=self.device)
        
        ped_map_vx[flat_batch, flat_r, flat_c] = flat_vx
        ped_map_vy[flat_batch, flat_r, flat_c] = flat_vy

        return torch.stack([ped_map_vx, ped_map_vy], dim=1) # [N, 2, 80, 80]


    def _compute_scan_map(self, state: EnvState) -> torch.Tensor:
        """
        并行计算激光雷达历史图 ( $1 \times 80 \times 80$ )。
        
        注意：DRL-VO 论文/代码在如何从 10 帧扫描 [cite: 205] 得到 1x80x80 [cite: 178] 
        方面存在矛盾。`custom_cnn_full.py` 期望 6400 个值，
        而 `cnn_data_pub.py` 似乎发布了 7200 个值。
        
        我们将采用论文中 Fig. 3a [cite: 239-247] 和 文本 [cite: 212] 描述的最合乎逻辑的
        高性能解释：
        1. 随时间取 Min pooling
        2. 随时间取 Avg pooling
        3. 拼接 (Concatenate) 两个结果
        4. Reshape 为 1x80x80
        
        这假设总点数 (self.lidar_points) 是 3200。
        (3200_min + 3200_avg = 6400 = 80*80)
        """
        # 假设 state.lidar_data.scan 提供了当前帧 [N, num_points]
        # (如果 state.lidar_data.scan_buffer 存在，可以直接使用)
        current_scan = state.lidar_data.scan # [N, self.lidar_points]
        
        # 滚动缓冲区
        self.lidar_history_buffer = torch.roll(self.lidar_history_buffer, shifts=-1, dims=1)
        self.lidar_history_buffer[:, -1] = current_scan

        # 1. 随时间 Min pooling [cite: 212]
        min_scan, _ = torch.min(self.lidar_history_buffer, dim=1) # [N, 3200]
        
        # 2. 随时间 Avg pooling [cite: 212]
        avg_scan = torch.mean(self.lidar_history_buffer, dim=1) # [N, 3200]
        
        # 3. 拼接
        scan_features = torch.cat([min_scan, avg_scan], dim=1) # [N, 6400]
        
        # 4. Reshape
        scan_map = scan_features.view(self.num_envs, 1, 80, 80) # [N, 1, 80, 80]
        
        return scan_map

    def _compute_vector_input(self, state: EnvState) -> torch.Tensor:
        """
        计算非 CNN 的向量输入：(相对子目标, 机器人局部速度)
        """
        # 提取 ego 状态 (世界坐标系)
        robot_pos = state.ego_drone.drone_state[:, :, :2].squeeze(1)  # [N, 2]
        robot_vel = state.ego_drone.drone_state[:, :, 7:9].squeeze(1) # [N, 2] (假设 7,8 是 vx, vy)
        robot_yaw = state.ego_drone.drone_state[:, :, 6]            # [N] (假设第6维是yaw)

        # 1. 机器人局部速度
        # 注意：DRL-VO 的动作空间是局部速度 [cite: 289]。
        # 你的 `observations.py` > `TrafficObservationProcessor` 也使用 'temporal_edges': robot_vel
        # 我们假设输入也应该是局部速度，以保持一致性。
        robot_vel_local = transform_to_robot_frame_batch(robot_vel,
                                                         torch.zeros_like(robot_pos),
                                                         robot_yaw) # [N, 2]
                                                         
        # 2. 相对子目标 (Subgoal)
        # DRL-VO 使用 "subgoal" [cite: 219]。
        # 你的 `NavObservationProcessorWithPath` 包含 `local_goals`，
        # 这在概念上是相同的。
        local_goals = state.navigation.local_goals # [N, 1, 3]
        if local_goals is None:
            # 如果没有 local_goal，回退到最终目标
            local_goals = state.navigation.target_positions # [N, 1, 3]
            
        local_goal_pos = local_goals[:, :, :2].squeeze(1) # [N, 2]
        
        # 2.1 相对子目标 (世界系)
        rel_goal_world = local_goal_pos - robot_pos # [N, 2]
        
        # 2.2 相对子目标 (机器人系)
        rel_goal_local = transform_to_robot_frame_batch(rel_goal_world,
                                                        torch.zeros_like(robot_pos),
                                                        robot_yaw) # [N, 2]
                                                        
        # 3. 拼接
        vector_input = torch.cat([rel_goal_local, robot_vel_local], dim=1) # [N, 4]
        
        return vector_input