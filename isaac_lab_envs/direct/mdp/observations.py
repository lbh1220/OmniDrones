import torch
from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
import numpy as np
import gymnasium as gym



class SimpleObservationProcessor:
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

        



class TrafficObservationProcessor:
    """Traffic环境的观测处理器，基于Isaac Lab tensor操作优化"""
    
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.predict_steps = cfg.predict_steps
        self.pred_timestep = cfg.pred_timestep
        self.observation_norm_scale = cfg.observation_norm_scale
        # Traffic aircraft数量
        self.total_traffic_num = cfg.traffic_sim.num_drones + getattr(cfg.traffic_sim, 'num_evtols', 0)
        self.drone_num = cfg.traffic_sim.num_drones
        self.evtol_num = getattr(cfg.traffic_sim, 'num_evtols', 0)
        
        # 传感器感知范围（使用NavEnv的observation_radius）
        self.sensor_range = cfg.observation_radius
        self.spatial_dim = 2 * (self.predict_steps + 1) + 1 
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

    def predict_traffic_trajectory(self, traffic_positions: torch.Tensor, traffic_velocities: torch.Tensor, traffic_types: torch.Tensor, traffic_safety_radius: torch.Tensor):
        """预计算traffic轨迹预测，供后续观测和奖励计算使用
        
        Args:
            traffic_positions: [total_traffic, 3] traffic位置
            traffic_velocities: [total_traffic, 3] traffic速度
            traffic_types: [total_traffic] traffic类型（0=evtol, 1=drone）
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
        total_traffic_num = self.cfg.traffic_sim.num_drones + self.cfg.traffic_sim.num_evtols
        
        # 1. 创建内层字典 "policy" 的内容
        policy_space_dict = {
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32),
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            'spatial_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_traffic_num, self.spatial_dim), dtype=np.float32),
            'detected_human_num': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        }
        return policy_space_dict

    def process_observation(self, drone_state: torch.Tensor, target_pos: torch.Tensor) -> dict:
        """处理观测数据（使用预计算的轨迹）
        
        Args:
            drone_state: [num_envs, 1, 13] ego drone状态
            target_pos: [num_envs, 1, 3] 目标位置
            
        Returns:
            观测字典
        """
        num_envs = drone_state.shape[0]
        
        # 提取ego drone信息
        robot_pos = drone_state[:, :, :2]  # [num_envs, 1, 2] (x, y)
        robot_vel = drone_state[:, :, 7:9]  # [num_envs, 1, 2] (vx, vy)
        # norm_robot_vel = robot_vel
        
        # 计算相对目标位置
        goal_pos = target_pos[:, :, :2]  # [num_envs, 1, 2] 只取x,y
        relative_goal_pos = goal_pos - robot_pos  # [num_envs, 1, 2]
        relative_goal_pos = relative_goal_pos / (2*self.circle_radius) * self.observation_norm_scale
        
        # 计算速度方向yaw
        robot_yaw = torch.atan2(robot_vel[:, :, 1], robot_vel[:, :, 0])  # [num_envs, 1]
        
        # 机器人参数
        robot_radius = torch.full((num_envs, 1, 1), self.cfg.safety_radius, device=self.device)
        robot_v_pref = torch.full((num_envs, 1, 1), self.cfg.max_speed, device=self.device)
        
        # 构建robot_node: [rel_goal_x, rel_goal_y, robot_radius, robot_v_pref, robot_yaw]
        robot_node = torch.cat([
            relative_goal_pos,  # [num_envs, 1, 2]
            robot_radius,       # [num_envs, 1, 1]  
            robot_v_pref,       # [num_envs, 1, 1]
            robot_yaw.unsqueeze(-1)  # [num_envs, 1, 1]
        ], dim=-1)  # [num_envs, 1, 5]
        
        # 计算空间边观测（使用预计算的轨迹）
        spatial_edges, detected_counts = self._compute_spatial_edges_from_cache(robot_pos, robot_vel)
        
        
        # 构建观测字典
        policy_obs = {
            'robot_node': robot_node,
            'temporal_edges': robot_vel,  # [num_envs, 1, 2]
            'spatial_edges': spatial_edges,
            'detected_human_num': detected_counts,
        }
        
        return {"policy": policy_obs}

    def _compute_spatial_edges_from_cache(self, robot_pos: torch.Tensor, robot_vel: torch.Tensor) -> torch.Tensor:
        """使用缓存的轨迹数据计算空间边观测
        
        Args:
            robot_pos: [num_envs, 1, 2] ego位置
            robot_vel: [num_envs, 1, 2] ego速度
            
        Returns:
            spatial_edges: [num_envs, total_traffic_num, spatial_dim]
        """
        num_envs = robot_pos.shape[0]
        spatial_dim = 2 * (self.predict_steps + 1)
        # self.spatial_dim = spatial_dim + 1
        
        if self.traffic_future_traj is None or self.traffic_future_traj.numel() == 0:
            spatial_edges = torch.full(
            (num_envs, self.total_traffic_num, self.spatial_dim),
            self.observation_norm_scale, device=self.device
            )
            detected_counts = torch.zeros((num_envs, 1), device=self.device)
            return spatial_edges, detected_counts
        
        # 计算相对位置 [num_envs, total_traffic, predict_steps+1, 2]
        traffic_pos_2d = self.traffic_future_traj[:, :, :2]  # [total_traffic, predict_steps+1, 2]
        robot_pos_expanded = robot_pos.unsqueeze(1)  # [num_envs, 1, 1, 2]
        
        relative_pos = traffic_pos_2d.unsqueeze(0) - robot_pos_expanded  # [num_envs, total_traffic, predict_steps+1, 2]
        
        # 计算距离用于感知范围过滤
        current_distances = torch.norm(relative_pos[:, :, 0], dim=-1)  # [num_envs, total_traffic]
        in_range_mask = current_distances <= self.sensor_range
        
        # 展平预测位置为spatial edges格式
        predicted_flat = relative_pos.view(num_envs, -1, spatial_dim)  # [num_envs, total_traffic, spatial_dim]
        predicted_flat = predicted_flat / (2*self.circle_radius) * self.observation_norm_scale

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
        #    torch.where(condition, value_if_true, value_if_false)
        #    这个操作是完全向量化的。
        #    - 如果掩码为 True (在范围内)，则保留 base_spatial_edges 的值。
        #    - 如果掩码为 False (不在范围内)，则赋值为您指定的大数值，例如 10.0。
        spatial_edges = torch.where(
            mask_expanded, 
            base_spatial_edges, 
            self.observation_norm_scale
        )
        # spatial_edges的维度是 [num_envs, current_traffic_num, spatial_dim+1]
        # 但是可能小于total_traffic_num，所以需要cat
        if spatial_edges.shape[1] < self.total_traffic_num:
            fill_spatial_edges = torch.full((num_envs, self.total_traffic_num - spatial_edges.shape[1], self.spatial_dim), self.observation_norm_scale, device=self.device)
            spatial_edges = torch.cat([spatial_edges, fill_spatial_edges], dim=1)

        detected_counts = torch.sum(in_range_mask, dim=1, dtype=torch.int32)  # [num_envs]
        detected_counts = torch.maximum(detected_counts, torch.ones_like(detected_counts))
        detected_counts = detected_counts.unsqueeze(-1) # [num_envs, 1]
        return spatial_edges, detected_counts
    
