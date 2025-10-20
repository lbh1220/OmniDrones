"""
Environment State Management
统一的环境状态管理类，支持多个命名空间
"""

import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass 
class EgoDroneNamespace:
    """自车无人机状态命名空间"""
    # 基础状态 [num_envs, 1, state_dim]
    drone_state: torch.Tensor = None         # 完整的无人机状态 [num_envs, 1, 13+]
    positions: torch.Tensor = None           # 位置 [num_envs, 1, 3]
    velocities: torch.Tensor = None          # 速度 [num_envs, 1, 3] 
    rotations: torch.Tensor = None           # 旋转四元数 [num_envs, 1, 4]
    angular_velocities: torch.Tensor = None  # 角速度 [num_envs, 1, 3]
    
    # 控制相关
    command_vel_xy: torch.Tensor = None      # 指令速度 [num_envs, 1, 2]
    previous_velocities: torch.Tensor = None  # 上一步的实际速度 [num_envs, 1, 3]
    

@dataclass
class NavigationNamespace:
    """导航相关状态命名空间"""
    # 目标和路径
    target_positions: torch.Tensor = None    # 目标位置 [num_envs, 1, 3]
    start_positions: torch.Tensor = None     # 起始位置 [num_envs, 1, 3]
    
    # 距离和状态
    current_dist_to_target: torch.Tensor = None  # 当前到目标距离 [num_envs]
    
    # 任务状态
    reached_target_mask: torch.Tensor = None     # 到达目标mask [num_envs]

    velocity_commands: torch.Tensor = None     # 速度指令 [num_envs, 1, 3]
    
    # 航路点信息
    waypoints: torch.Tensor = None           # 航路点 [num_envs, 20, 4]
    waypoint_lengths: torch.Tensor = None    # 航路点长度 [num_envs]
    current_waypoint_indices: torch.Tensor = None  # 当前航路点索引 [num_envs]

    # for local
    local_goals: torch.Tensor = None           # 局部目标 [num_envs, 1, 3]
    projection_points: torch.Tensor = None      # 投影点 [num_envs, 1, 3]
    cross_track_errors: torch.Tensor = None      # 横向误差 [num_envs]
    current_dist_along_path: torch.Tensor = None      # 投影点，到终点的距离 [num_envs]


@dataclass
class CollisionNamespace:
    """碰撞检测相关状态命名空间"""
    collision_mask: torch.Tensor = None      # 碰撞mask [num_envs]
    safety_radius: float = 1.0               # 安全半径
    collision_objects: List[Optional[str]] = field(default_factory=list)  # 碰撞对象列表


@dataclass
class PerceptionNamespace:
    """感知相关命名空间"""
    lidar_scan: torch.Tensor = None            # [num_envs, 1, W, H] 或 [num_envs, 1, 36, 4]


@dataclass
class MapNamespace:
    """地图/静态场景相关命名空间"""
    # 全局点云/高度图（以世界坐标网格表示）
    point_cloud_xy: torch.Tensor | None = None   # [N, 2]
    point_cloud_z: torch.Tensor | None = None    # [N]
    height_map: torch.Tensor | None = None       # [H, W]
    pc_shape_hw: tuple | None = None             # (H, W)
    pc_bounds: tuple | None = None               # (xmin, xmax, ymin, ymax)
    pc_resolution: float | None = None           # 采样分辨率（米）

    occupancy_grid: torch.Tensor | None = None   # [H, W] # occupancy grid可以看作感知的结果，比如用于observation
    grid_size: float | None = None              # 网格大小（米）
    grid_bounds: tuple | None = None            # (xmin, xmax, ymin, ymax)
    extended_occupancy_grid: torch.Tensor | None = None # [H, W] # extended occupancy grid就是用来判断是否碰撞的工具


@dataclass
class MissionNamespace:
    """任务相关状态命名空间"""
    mission_type: str = "navigation"         # 任务类型
    episode_progress: torch.Tensor = None   # 任务进度 [num_envs]
    success_mask: torch.Tensor = None       # 成功mask [num_envs]
    failure_mask: torch.Tensor = None       # 失败mask [num_envs]


@dataclass
class TrafficNamespace:
    """交通环境相关状态命名空间（可选）"""
    # 交通飞机状态
    traffic_positions: torch.Tensor = None     # [total_traffic, 3]
    traffic_velocities: torch.Tensor = None    # [total_traffic, 3]
    traffic_types: torch.Tensor = None         # [total_traffic] 类型索引, 0=dummy, 1=drone, 2=evtol
    traffic_safety_radius: torch.Tensor = None # [total_traffic]
    traffic_future_traj: torch.Tensor = None   # [total_traffic, predict_steps+1, 3]
    
    # 预测相关参数
    predict_steps: int = 5
    pred_timestep: float = 2.0


@dataclass
class MdpNamespace:
    """MDP 相关状态命名空间
    统一维护跨组件访问所需的张量：观测、奖励、终止与截断。
    """
    observations: dict | None = None            # 最近一次观测（可为字典 of tensors）
    reward: torch.Tensor | None = None          # [num_envs]
    terminated: torch.Tensor | None = None      # [num_envs] bool
    truncated: torch.Tensor | None = None       # [num_envs] bool


class EnvState:
    """统一的环境状态管理类，支持多个命名空间"""
    
    def __init__(self, device: str = "cuda", num_envs: int = 1):
        self.device = device
        self.num_envs = num_envs
        
        # 基础命名空间
        self.ego_drone = EgoDroneNamespace()
        self.navigation = NavigationNamespace() 
        self.collision = CollisionNamespace()
        self.perception = PerceptionNamespace()
        self.mission = MissionNamespace()
        self.map = MapNamespace()
        
        # 可选命名空间
        self.traffic = None  # 只在需要时初始化
        # MDP 命名空间
        self.mdp = MdpNamespace()
        
        # 其他可扩展的命名空间
        self._custom_namespaces = {}
        
    def init_traffic_namespace(self, predict_steps: int = 5, pred_timestep: float = 2.0):
        """初始化交通命名空间"""
        self.traffic = TrafficNamespace()
        self.traffic.predict_steps = predict_steps
        self.traffic.pred_timestep = pred_timestep
        
    def add_custom_namespace(self, name: str, namespace_obj: Any):
        """添加自定义命名空间"""
        self._custom_namespaces[name] = namespace_obj
        
    def get_custom_namespace(self, name: str) -> Any:
        """获取自定义命名空间"""
        return self._custom_namespaces.get(name)
        
    def initialize_basic_tensors(self, state_dim: int = 13):
        """初始化基础张量"""
        # 自车无人机状态
        self.ego_drone.drone_state = torch.zeros(self.num_envs, 1, state_dim, device=self.device)
        self.ego_drone.positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ego_drone.velocities = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ego_drone.rotations = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.ego_drone.rotations[:, :, 0] = 1.0  # 初始化为单位四元数
        self.ego_drone.angular_velocities = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ego_drone.previous_velocities = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # 导航状态
        self.navigation.target_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.start_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.current_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.navigation.reached_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.navigation.velocity_commands = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 航路点信息

        self.navigation.waypoints = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.navigation.waypoint_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.navigation.current_waypoint_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # for local 
        self.navigation.local_goals = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.projection_points = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.cross_track_errors = torch.zeros(self.num_envs, device=self.device)
        self.navigation.current_dist_along_path = torch.zeros(self.num_envs, device=self.device)
        # 碰撞状态
        self.collision.collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision.collision_objects = [None] * self.num_envs
        # 感知
        self.perception.lidar_scan = None
        
        # 任务状态
        self.mission.episode_progress = torch.zeros(self.num_envs, device=self.device)
        self.mission.success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mission.failure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # MDP 状态
        self.mdp.reward = torch.zeros(self.num_envs, device=self.device)
        self.mdp.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mdp.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 地图命名空间：初始化为 None（延后构建）
        self.map.point_cloud_xy = None
        self.map.point_cloud_z = None
        self.map.height_map = None
        self.map.pc_shape_hw = None
        self.map.pc_bounds = None
        self.map.pc_resolution = None
        self.map.occupancy_grid = None
        self.map.grid_size = None
        self.map.grid_bounds = None
        
    def update_ego_drone_state(self, drone_state: torch.Tensor):
        """更新自车无人机状态"""
        self.ego_drone.drone_state = drone_state
        # 从完整状态中提取各个组件
        self.ego_drone.positions = drone_state[:, :, :3]

        if self.ego_drone.velocities is not None:
            self.ego_drone.previous_velocities = self.ego_drone.velocities.clone()
        self.ego_drone.velocities = drone_state[:, :, 7:10] if drone_state.shape[-1] > 10 else None

        
        # 可以根据需要提取更多组件
        
    def update_navigation_distances(self):
        """更新导航距离信息"""
        if (self.ego_drone.positions is not None and 
            self.navigation.target_positions is not None):
            # 计算当前到目标的距离
            relative_pos = self.navigation.target_positions[:, :, :2] - self.ego_drone.positions[:, :, :2]
            self.navigation.current_dist_to_target = torch.norm(relative_pos.squeeze(1), dim=1)
            
    def update_reached_target_mask(self, arrival_threshold: float):
        """更新到达目标mask"""
        if self.navigation.current_dist_to_target is not None:
            self.navigation.reached_target_mask = (
                self.navigation.current_dist_to_target <= arrival_threshold
            )
            
    def reset_env_states(self, env_ids: torch.Tensor):
        """重置指定环境的状态"""
        if env_ids is None or len(env_ids) == 0:
            return
            
        # 重置导航状态
        if self.navigation.current_dist_to_target is not None:
            self.navigation.current_dist_to_target[env_ids] = 0.0

        if self.navigation.reached_target_mask is not None:
            self.navigation.reached_target_mask[env_ids] = False
            
        # 重置碰撞状态
        if self.collision.collision_mask is not None:
            self.collision.collision_mask[env_ids] = False

        if self.navigation.current_dist_along_path is not None:
            self.navigation.current_dist_along_path[env_ids] = 0.0
            
        # 重置任务状态
        if self.mission.success_mask is not None:
            self.mission.success_mask[env_ids] = False
        if self.mission.failure_mask is not None:
            self.mission.failure_mask[env_ids] = False
        if self.mission.episode_progress is not None:
            self.mission.episode_progress[env_ids] = 0.0
            
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要（用于调试）"""
        summary = {
            "num_envs": self.num_envs,
            "device": self.device,
            "ego_drone": {
                "has_drone_state": self.ego_drone.drone_state is not None,
                "has_positions": self.ego_drone.positions is not None,
                "has_velocities": self.ego_drone.velocities is not None,
            },
            "navigation": {
                "has_targets": self.navigation.target_positions is not None,
                "targets_reached": self.navigation.reached_target_mask.sum().item() if self.navigation.reached_target_mask is not None else 0,
            },
            "collision": {
                "collisions_detected": self.collision.collision_mask.sum().item() if self.collision.collision_mask is not None else 0,
            },
            "mission": {
                "successes": self.mission.success_mask.sum().item() if self.mission.success_mask is not None else 0,
                "failures": self.mission.failure_mask.sum().item() if self.mission.failure_mask is not None else 0,
            },
            "has_traffic": self.traffic is not None,
            "custom_namespaces": list(self._custom_namespaces.keys())
        }
        return summary
        
    def to_device(self, device: str):
        """将所有张量移动到指定设备"""
        self.device = device
        
        # 移动ego_drone张量
        for attr_name in dir(self.ego_drone):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.ego_drone, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.ego_drone, attr_name, attr_value.to(device))
                    
        # 移动navigation张量
        for attr_name in dir(self.navigation):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.navigation, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.navigation, attr_name, attr_value.to(device))
                    
        # 移动collision张量
        for attr_name in dir(self.collision):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.collision, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.collision, attr_name, attr_value.to(device))
                    
        # 移动mission张量
        for attr_name in dir(self.mission):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.mission, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.mission, attr_name, attr_value.to(device))
        # 移动map张量
        for attr_name in dir(self.map):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.map, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.map, attr_name, attr_value.to(device))
        
        # 移动mdp张量
        for attr_name in dir(self.mdp):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.mdp, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self.mdp, attr_name, attr_value.to(device))
                    
        # 移动traffic张量（如果存在）
        if self.traffic is not None:
            for attr_name in dir(self.traffic):
                if not attr_name.startswith('_'):
                    attr_value = getattr(self.traffic, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self.traffic, attr_name, attr_value.to(device))

    # -------------------- MDP helpers --------------------
    def set_observations(self, observations: dict | None):
        self.mdp.observations = observations

    def set_reward(self, reward: torch.Tensor):
        if reward is None:
            return
        # 保证形状为 [num_envs]
        if reward.ndim > 1:
            reward = reward.squeeze()
        self.mdp.reward = reward

    def set_dones(self, terminated: torch.Tensor, truncated: torch.Tensor):
        if terminated is not None:
            self.mdp.terminated = terminated.bool()
        if truncated is not None:
            self.mdp.truncated = truncated.bool()

    def reset_mdp(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return
        if self.mdp.reward is not None:
            self.mdp.reward[env_ids] = 0.0
        if self.mdp.terminated is not None:
            self.mdp.terminated[env_ids] = False
        if self.mdp.truncated is not None:
            self.mdp.truncated[env_ids] = False
    def update_navigation_state_vectorized(self, lookahead_distance: float = 10.0, env_ids: torch.Tensor | None = None):
        """
        【矢量化版】为指定环境（或全部环境）更新其导航状态。
        - 逻辑与 iterative 版本完全一致，但使用并行的张量运算。
        
        Args:
            lookahead_distance: 计算局部目标时，沿路径前进的距离。
            env_ids: 需要更新的环境ID。如果为 None，则更新所有环境。
        """
        # 如果 env_ids 为 None，则处理所有环境
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 如果没有需要处理的环境，则直接返回
        if len(env_ids) == 0:
            return

        # --- 0. 提取需要处理的环境的数据子集 ---
        positions_3d = self.ego_drone.positions[env_ids, 0]           # shape: (num_ids, 3)
        positions_2d = positions_3d[:, :2]                            # shape: (num_ids, 2)
        waypoints = self.navigation.waypoints[env_ids]                # shape: (num_ids, max_len, 3)
        waypoints_2d = waypoints[:, :, :2]                            # shape: (num_ids, max_len, 2)
        waypoint_lengths = self.navigation.waypoint_lengths[env_ids]  # shape: (num_ids,)
        target_positions = self.navigation.target_positions[env_ids, 0] # shape: (num_ids, 3)

        max_waypoints = waypoints.shape[1]
        num_active_envs = len(env_ids)

        # --- 1. 处理路径点过少的特殊情况 ---
        short_path_mask = waypoint_lengths < 2
        
        # --- 2. 矢量化计算最佳投影点 ---
        # 仅对路径点足够的环境进行计算
        long_path_mask = ~short_path_mask
        if torch.any(long_path_mask):
            # 准备航路段张量
            wp_starts = waypoints_2d[long_path_mask, :-1, :] # (N_long, max_len-1, 2)
            wp_ends = waypoints_2d[long_path_mask, 1:, :]   # (N_long, max_len-1, 2)

            # 矢量化投影计算
            segment_vecs = wp_ends - wp_starts
            segment_lens_sq = torch.sum(segment_vecs**2, dim=-1) + 1e-6
            to_current_vecs = positions_2d[long_path_mask].unsqueeze(1) - wp_starts
            projection_ratios = torch.einsum('nij,nij->ni', to_current_vecs, segment_vecs) / segment_lens_sq
            clamped_ratios = torch.clamp(projection_ratios, 0.0, 1.0)
            
            # 计算所有航路段上的投影点
            projection_points = wp_starts + clamped_ratios.unsqueeze(-1) * segment_vecs
            
            # 计算所有误差并应用掩码
            cross_track_errors_sq = torch.sum((positions_2d[long_path_mask].unsqueeze(1) - projection_points)**2, dim=-1)
            segment_indices = torch.arange(max_waypoints - 1, device=self.device).unsqueeze(0)
            valid_segment_mask = segment_indices < (waypoint_lengths[long_path_mask] - 1).unsqueeze(-1)
            cross_track_errors_sq[~valid_segment_mask] = float('inf')
            
            # 找到每个环境的最佳航路段索引
            best_segment_indices_long = torch.argmin(cross_track_errors_sq, dim=1) # (N_long,)
            best_ratios_long = torch.gather(clamped_ratios, 1, best_segment_indices_long.unsqueeze(-1)).squeeze(-1) # (N_long,)
            
            # --- 3. 显式维护投影状态 ---
            start_points = torch.gather(wp_starts, 1, best_segment_indices_long.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            end_points = torch.gather(wp_ends, 1, best_segment_indices_long.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            final_projection_point_2d = start_points + best_ratios_long.unsqueeze(-1) * (end_points - start_points)
            
            self.navigation.projection_points[env_ids[long_path_mask], 0, :2] = final_projection_point_2d
            self.navigation.projection_points[env_ids[long_path_mask], 0, 2] = positions_3d[long_path_mask, 2]
            min_dist_sq = torch.gather(cross_track_errors_sq, 1, best_segment_indices_long.unsqueeze(-1)).squeeze(-1)
            self.navigation.cross_track_errors[env_ids[long_path_mask]] = torch.sqrt(min_dist_sq)

            # --- 4. 矢量化计算局部目标 (Local Goal) ---
            # 预计算所有有效航路段的长度
            segment_lengths = torch.norm(segment_vecs, dim=-1) # (N_long, max_len-1)
            segment_lengths[~valid_segment_mask] = 0.0 # 忽略无效段
            
            # 计算到投影点的路径总距离
            cumulative_lengths = torch.cumsum(segment_lengths, dim=1)
            # 减去自身长度，得到到航段起点的累积长度
            dist_to_segment_start = cumulative_lengths - segment_lengths
            
            gathered_dist_to_start = torch.gather(dist_to_segment_start, 1, best_segment_indices_long.unsqueeze(-1)).squeeze(-1)
            gathered_segment_len = torch.gather(segment_lengths, 1, best_segment_indices_long.unsqueeze(-1)).squeeze(-1)

            dist_to_projection = gathered_dist_to_start + best_ratios_long * gathered_segment_len

            # --- 新增代码开始: 计算到终点的距离 ---
            # 1. 计算每条路径的总长度
            #    路径总长等于其最后一个有效航路段的累积长度
            last_segment_indices = waypoint_lengths[long_path_mask] - 2
            #    钳制以防止路径只有1个航路段时索引为负
            last_segment_indices = torch.clamp(last_segment_indices, min=0) 
            
            #    使用 gather 批量获取每条路径的总长度
            total_path_lengths = torch.gather(cumulative_lengths, 1, last_segment_indices.unsqueeze(-1)).squeeze(-1)
            
            # 2. 计算投影点到终点的距离
            dist_projection_to_goal = total_path_lengths - dist_to_projection
            #    确保距离不为负
            dist_projection_to_goal = torch.clamp(dist_projection_to_goal, min=0.0)
            
            # 3. 存入状态
            self.navigation.current_dist_along_path[env_ids[long_path_mask]] = dist_projection_to_goal
            # --- 新增代码结束 ---

            # 计算目标点在路径上的总距离
            target_dist_along_path = dist_to_projection + lookahead_distance
            
            # 找到局部目标所在的航路段 (这是最关键的技巧)
            # 比较目标总距离和每个航段终点的累积总距离
            is_past_segment = target_dist_along_path.unsqueeze(1) > cumulative_lengths
            # 对已通过的航路段求和，即可得到目标点所在的航路段索引
            local_goal_segment_indices = torch.sum(is_past_segment, dim=1) # (N_long,)
            
            # 确保索引不越界
            max_valid_segment_idx = waypoint_lengths[long_path_mask] - 2
            local_goal_segment_indices = torch.min(local_goal_segment_indices, max_valid_segment_idx)

            # 计算在目标航路段内的前进距离
            dist_to_goal_segment_start = torch.gather(dist_to_segment_start, 1, local_goal_segment_indices.unsqueeze(-1)).squeeze(-1)
            dist_into_goal_segment = target_dist_along_path - dist_to_goal_segment_start
            
            # 计算并插值得到最终的局部目标
            goal_seg_starts = torch.gather(wp_starts, 1, local_goal_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            goal_seg_ends = torch.gather(wp_ends, 1, local_goal_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            goal_seg_vecs = goal_seg_ends - goal_seg_starts
            goal_seg_lens = torch.norm(goal_seg_vecs, dim=-1, keepdim=True) + 1e-6
            
            ratio_on_goal_segment = (dist_into_goal_segment.unsqueeze(-1) / goal_seg_lens)
            local_goals_2d = goal_seg_starts + torch.clamp(ratio_on_goal_segment, 0.0, 1.0) * goal_seg_vecs

            # 维护局部目标状态
            flight_altitude = waypoints[long_path_mask, 0, 2] # 使用固定高度
            self.navigation.local_goals[env_ids[long_path_mask], 0, :2] = local_goals_2d
            self.navigation.local_goals[env_ids[long_path_mask], 0, 2] = flight_altitude
            
            # --- 5. 矢量化更新当前目标航路点索引 ---
            new_wp_indices_long = local_goal_segment_indices + 1
            self.navigation.current_waypoint_indices[env_ids[long_path_mask]] = new_wp_indices_long

        # --- 6. 合并特殊情况和一般情况的结果 ---
        if torch.any(short_path_mask):
            self.navigation.local_goals[env_ids[short_path_mask], 0] = target_positions[short_path_mask]
            self.navigation.projection_points[env_ids[short_path_mask], 0] = positions_3d[short_path_mask]
            self.navigation.cross_track_errors[env_ids[short_path_mask]] = 0.0
            # 如果路径点>0,则为长度-1，否则为0
            short_path_indices = torch.clamp(waypoint_lengths[short_path_mask] - 1, min=0)
            self.navigation.current_waypoint_indices[env_ids[short_path_mask]] = short_path_indices
            self.navigation.current_dist_along_path[env_ids[short_path_mask]] = torch.norm(target_positions[short_path_mask] - positions_3d[short_path_mask], dim=-1)

    # this func is deprecated
    def update_navigation_state_iterative(self, lookahead_distance: float = 10.0, env_ids: torch.Tensor | None = None):
        """
        通过遍历的方式，为每个环境更新其导航状态。
        - 显式维护投影点和航迹误差。
        - 基于局部目标位置更新当前目标航路点索引。
        
        Args:
            lookahead_distance: 计算局部目标时，沿路径前进的距离。
        """
        
        # --- 准备用于存储结果的张量 ---
        # 我们可以直接在循环中修改 self.navigation 中的张量
        
        # --- 开始遍历所有并行环境 ---
        for i in range(self.num_envs):
            if env_ids is not None and i not in env_ids:
                continue
            # --- 1. 提取第 i 个环境的数据 ---
            current_pos_3d = self.ego_drone.positions[i, 0] # shape: (3,)
            current_pos_2d = current_pos_3d[:2]           # shape: (2,)
            num_waypoints = self.navigation.waypoint_lengths[i]
            
            # 如果航路点少于2个（无法构成航路段），则进行特殊处理
            if num_waypoints < 2:
                final_target_3d = self.navigation.target_positions[i, 0]
                self.navigation.local_goals[i, 0] = final_target_3d
                self.navigation.projection_points[i, 0] = current_pos_3d
                self.navigation.cross_track_errors[i] = 0.0
                self.navigation.current_waypoint_indices[i] = num_waypoints - 1 if num_waypoints > 0 else 0
                self.navigation.current_dist_along_path[i] = 0.0
                continue
                
            waypoints_2d = self.navigation.waypoints[i, :num_waypoints, :2] # shape: (num_waypoints, 2)
            
            # --- 2. 计算在路径上的最佳投影点 ---
            min_dist_sq = torch.full((1,), float('inf'), device=self.device)
            best_segment_idx = 0
            best_projection_ratio = 0.0
            
            for j in range(num_waypoints - 1):
                wp1 = waypoints_2d[j]
                wp2 = waypoints_2d[j + 1]
                segment_vec = wp2 - wp1
                segment_len_sq = torch.dot(segment_vec, segment_vec)
                
                if segment_len_sq < 1e-6: continue
                    
                to_current_vec = current_pos_2d - wp1
                projection_ratio = torch.dot(to_current_vec, segment_vec) / segment_len_sq
                # projection_ratio could be negative, then clamp
                clamped_ratio = torch.clamp(projection_ratio, 0.0, 1.0)
                projection_point = wp1 + clamped_ratio * segment_vec
                dist_sq = torch.sum((current_pos_2d - projection_point)**2)
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_segment_idx = j
                    best_projection_ratio = clamped_ratio

            # --- 3. 显式维护投影状态 ---
            final_projection_point_2d = waypoints_2d[best_segment_idx] + \
                                        best_projection_ratio * (waypoints_2d[best_segment_idx+1] - waypoints_2d[best_segment_idx])
            
            # Z轴使用当前飞机的高度
            self.navigation.projection_points[i, 0, :2] = final_projection_point_2d
            self.navigation.projection_points[i, 0, 2] = current_pos_3d[2]
            self.navigation.cross_track_errors[i] = torch.sqrt(min_dist_sq)

            # --- 4. 计算局部目标 (Local Goal)，并记录其所在航路段 ---
            remaining_dist = lookahead_distance
            local_goal_2d = final_projection_point_2d.clone()
            local_goal_segment_idx = best_segment_idx

            for k in range(best_segment_idx, num_waypoints - 1):
                start_point_2d = waypoints_2d[k]
                if k == best_segment_idx:
                    start_point_2d = final_projection_point_2d
                
                end_point_2d = waypoints_2d[k + 1]
                segment_vec = end_point_2d - start_point_2d
                segment_len = torch.norm(segment_vec)

                if segment_len < 1e-6: continue

                if remaining_dist <= segment_len:
                    # 在当前航路段内即可找到局部目标
                    local_goal_2d = start_point_2d + (remaining_dist / segment_len) * segment_vec
                    local_goal_segment_idx = k
                    break
                else:
                    # 无法在当前段内满足前进距离，移动到下一段的起点
                    remaining_dist -= segment_len
                    # 如果已经是倒数第二个航路段，说明local goal就在最终点
                    # 航路段的数量比waypoints少1
                    if k == num_waypoints - 2:
                        local_goal_2d = end_point_2d
                        local_goal_segment_idx = k
                        break
            
            # 显式维护局部目标
            flight_altitude = self.navigation.waypoints[i, 0, 2] # 使用路径的固定高度
            self.navigation.local_goals[i, 0, :2] = local_goal_2d
            self.navigation.local_goals[i, 0, 2] = flight_altitude
            
            # --- 5. 根据局部目标位置，更新当前目标航路点索引 ---
            # 当前目标航路点，就是局部目标所在航路段的终点
            new_wp_idx = local_goal_segment_idx + 1
            self.navigation.current_waypoint_indices[i] = new_wp_idx