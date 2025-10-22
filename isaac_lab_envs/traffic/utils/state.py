import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    speed: float

@dataclass
class Waypoint_ex:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    qx: float
    qy: float
    qz: float
    qw: float

class TrafficState:
    """
    管理单个manager中所有飞机状态的类，作为外部与Isaac Sim中manager的桥梁
    每个manager实例化一个state，包含飞机的完整状态信息，参考AircraftState设计
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # 飞机数量和基本信息
        self.num_aircraft = 0
        self.names = []  # List[str]
        self.aircraft_types = []  # List[str] 'drone' 或 'evtol'
        self.safety_radius = torch.empty(0, device=device)  # [N] 安全半径
        self.time_stamps = torch.empty(0, device=device)  # [N] 时间戳
        self.max_speed = torch.empty(0, device=device)  # [N] 最大速度

        self.min_speed = torch.empty(0, device=device)  # [N] 最小速度
        self.v_pref = torch.empty(0, device=device)  # [N] 期望速度
        
        # 运动状态 - 对应AircraftState中的运动信息
        self.positions = torch.empty(0, 3, device=device)  # [N, 3]
        self.velocities = torch.empty(0, 3, device=device)  # [N, 3] 
        self.rotations = torch.empty(0, 4, device=device)  # [N, 4] quaternions
        self.angular_velocities = torch.empty(0, 3, device=device)  # [N, 3]
        self.linear_accelerations = torch.empty(0, 3, device=device)  # [N, 3]
        self.angular_accelerations = torch.empty(0, 3, device=device)  # [N, 3]
        
        # 目标和路径信息 - 对应AircraftState中的目标信息
        self.target_positions = torch.empty(0, 3, device=device)  # [N, 3]
        self.start_positions = torch.empty(0, 3, device=device)  # [N, 3]
        self.current_waypoint_indices = torch.empty(0, dtype=torch.long, device=device)  # [N]
        self.velocity_commands = torch.empty(0, 3, device=device)  # [N, 3]
        

        
        # 碰撞信息
        self.has_collided = torch.empty(0, dtype=torch.bool, device=device)  # [N]
        self.collision_objects = []  # List[Optional[str]]
        
        # 航路点信息 - 使用固定长度张量，简化管理
        self.max_waypoints = 20  # 预定义最大航路点数量
        self.waypoints = torch.empty(0, self.max_waypoints, 4, device=device)  # [N, 20, 4] (x,y,z,speed)
        self.waypoint_lengths = torch.empty(0, dtype=torch.long, device=device)  # [N] 每个飞机实际航路点数量
        
        # 地图/占据栅格（可选）
        self.occupancy_grid = None            # [H, W] bool
        self.extended_occupancy_grid = None   # [H, W] bool，按安全半径扩展
        self.grid_bounds = None               # (xmin, xmax, ymin, ymax)
        self.grid_size = None                 # float (meters per cell)

        # 导航派生量（局部目标/投影/误差/沿路径距离）
        self.local_goals = torch.empty(0, 3, device=device)           # [N, 3]
        self.projection_points = torch.empty(0, 3, device=device)     # [N, 3]
        self.cross_track_errors = torch.empty(0, device=device)       # [N]
        self.current_dist_along_path = torch.empty(0, device=device)  # [N]
    
    def initialize_aircraft(self, names: List[str], aircraft_types: List[str], 
                           safety_radius: List[float], max_speed: List[float], 
                           min_speed: List[float], v_pref: List[float], device: str = None):
        """初始化飞机列表"""
        if device is None:
            device = self.device
            
        self.num_aircraft = len(names)
        self.names = names.copy()
        self.aircraft_types = aircraft_types.copy()
        self.safety_radius = torch.tensor(safety_radius, device=device)
        self.time_stamps = torch.zeros(self.num_aircraft, device=device)
        self.max_speed = torch.tensor(max_speed, device=device)
        self.min_speed = torch.tensor(min_speed, device=device)
        self.v_pref = torch.tensor(v_pref, device=device)
        # 初始化运动状态
        self.positions = torch.zeros(self.num_aircraft, 3, device=device)
        self.velocities = torch.zeros(self.num_aircraft, 3, device=device)
        self.rotations = torch.zeros(self.num_aircraft, 4, device=device)
        self.rotations[:, 0] = 1.0  # 初始化为单位四元数 [w, x, y, z]
        self.angular_velocities = torch.zeros(self.num_aircraft, 3, device=device)
        self.linear_accelerations = torch.zeros(self.num_aircraft, 3, device=device)
        self.angular_accelerations = torch.zeros(self.num_aircraft, 3, device=device)
        
        # 初始化目标和路径
        self.target_positions = torch.zeros(self.num_aircraft, 3, device=device)
        self.start_positions = torch.zeros(self.num_aircraft, 3, device=device)
        self.current_waypoint_indices = torch.zeros(self.num_aircraft, dtype=torch.long, device=device)
        self.velocity_commands = torch.zeros(self.num_aircraft, 3, device=device)
        
        # 初始化状态
        self.has_collided = torch.zeros(self.num_aircraft, dtype=torch.bool, device=device)
        self.collision_objects = [None] * self.num_aircraft
        
        # 初始化航路点数据
        self.waypoints = torch.zeros(self.num_aircraft, self.max_waypoints, 4, device=device)
        self.waypoint_lengths = torch.zeros(self.num_aircraft, dtype=torch.long, device=device)
        # 初始化导航派生量
        self.local_goals = torch.zeros(self.num_aircraft, 3, device=device)
        self.projection_points = torch.zeros(self.num_aircraft, 3, device=device)
        self.cross_track_errors = torch.zeros(self.num_aircraft, device=device)
        self.current_dist_along_path = torch.zeros(self.num_aircraft, device=device)
    
    def set_waypoints_for_aircraft(self, aircraft_idx: int, waypoints: List[Waypoint]):
        """为特定飞机设置航路点，自动处理长度填充"""
        if not (0 <= aircraft_idx < self.num_aircraft):
            return
        
        num_waypoints = len(waypoints)
        actual_length = min(num_waypoints, self.max_waypoints)
        
        # 设置实际航路点
        for i in range(actual_length):
            self.waypoints[aircraft_idx, i, 0] = waypoints[i].x
            self.waypoints[aircraft_idx, i, 1] = waypoints[i].y
            self.waypoints[aircraft_idx, i, 2] = waypoints[i].z
            self.waypoints[aircraft_idx, i, 3] = waypoints[i].speed
        
        # 用最后一个waypoint填充剩余位置
        if actual_length < self.max_waypoints and actual_length > 0:
            last_waypoint = self.waypoints[aircraft_idx, actual_length - 1]
            for i in range(actual_length, self.max_waypoints):
                self.waypoints[aircraft_idx, i] = last_waypoint
        
        self.waypoint_lengths[aircraft_idx] = actual_length
    
    def get_aircraft_waypoints(self, aircraft_idx: int) -> Optional[torch.Tensor]:
        """获取特定飞机的有效航路点"""
        if not (0 <= aircraft_idx < self.num_aircraft):
            return None
        
        length = self.waypoint_lengths[aircraft_idx].item()
        return self.waypoints[aircraft_idx, :length].clone()
    

    
    def update_collision_info(self, has_collided: torch.Tensor, collision_objects: List[Optional[str]]):
        """更新碰撞信息"""
        self.has_collided = has_collided.clone()
        self.collision_objects = collision_objects.copy()
    
    def get_aircraft_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取特定飞机的完整状态"""
        if name in self.names:
            idx = self.names.index(name)
            return {
                "name": self.names[idx],
                "aircraft_type": self.aircraft_types[idx],
                "position": self.positions[idx],
                "velocity": self.velocities[idx],
                "rotation": self.rotations[idx],
                "target": self.target_positions[idx],
                "start": self.start_positions[idx],
                "waypoint_idx": self.current_waypoint_indices[idx].item(),
                "has_collided": self.has_collided[idx].item(),
                "waypoints": self.get_aircraft_waypoints(idx)
            }
        return None
    
    def get_aircraft_by_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """根据索引获取特定飞机的完整状态"""
        if 0 <= idx < self.num_aircraft:
            return {
                "name": self.names[idx],
                "aircraft_type": self.aircraft_types[idx],
                "position": self.positions[idx],
                "velocity": self.velocities[idx],
                "rotation": self.rotations[idx],
                "target": self.target_positions[idx],
                "start": self.start_positions[idx],
                "waypoint_idx": self.current_waypoint_indices[idx].item(),
                "has_collided": self.has_collided[idx].item(),
                "waypoints": self.get_aircraft_waypoints(idx)
            }
        return None
    
    def get_aircraft_states_dict(self) -> Dict[str, torch.Tensor]:
        """获取所有飞机状态的字典形式 (用于兼容原接口)"""
        states = {}
        for i, name in enumerate(self.names):
            states[name] = torch.cat([
                self.positions[i],      # position [3]
                self.rotations[i],      # rotation [4] 
                self.velocities[i]      # velocity [3]
            ])  # total [10]
        return states
    
    def check_collision(self, external_positions: torch.Tensor, safety_radius: float = 2.0) -> torch.Tensor:
        """
        检查外部飞机与本manager管理的飞机的碰撞风险
        
        Args:
            external_positions: 外部飞机位置 [N, 3]
            safety_radius: 安全距离阈值
            
        Returns:
            碰撞风险布尔张量 [N]
        """
        if self.positions.shape[0] == 0:
            return torch.zeros(external_positions.shape[0], dtype=torch.bool, device=self.device)
        
        # 计算距离矩阵
        distances = torch.cdist(external_positions, self.positions)
        
        # 检查是否有任何距离小于安全阈值
        collision_risk = (distances < safety_radius).any(dim=1)
        
        return collision_risk

    def are_positions_safe(self, positions: torch.Tensor) -> torch.Tensor:
        """
        使用扩展占据网格检查若干位置是否安全（未处于占据单元内）。
        Args:
            positions: [N, 3] 或 [3]
        Returns:
            [N] bool，True 表示安全或超出地图（视为安全）。
        """
        grid = self.extended_occupancy_grid
        bounds = self.grid_bounds
        grid_size = self.grid_size
        if grid is None or bounds is None or grid_size is None:
            if positions.ndim == 1:
                return torch.ones(1, dtype=torch.bool, device=self.device)
            return torch.ones(positions.shape[0], dtype=torch.bool, device=self.device)

        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        device = grid.device
        positions = positions.to(device)
        x = positions[:, 0]
        y = positions[:, 1]
        xmin, xmax, ymin, ymax = map(float, bounds)
        gs = float(grid_size)
        H, W = grid.shape[-2], grid.shape[-1]

        ix = torch.floor((x - xmin) / gs).long()
        iy = torch.floor((y - ymin) / gs).long()

        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        safe = torch.ones(positions.shape[0], dtype=torch.bool, device=device)
        if torch.any(in_bounds):
            ix_in = ix[in_bounds]
            iy_in = iy[in_bounds]
            occupied = grid[iy_in, ix_in]
            safe[in_bounds] = ~occupied

        return safe

    def update_navigation_state_vectorized(self, lookahead_distance: float = 10.0, indices: Optional[torch.Tensor] = None):
        """
        矢量化更新局部导航状态（局部目标、投影点、横向误差、沿路径距离）。
        逻辑参考 direct.mdp.state.update_navigation_state_vectorized，但适配 traffic 的张量形状：
        - 当前位置: positions [N, 3]
        - 航路点: waypoints [N, M, 4]，仅使用前3列 (x, y, z)
        - 航路点长度: waypoint_lengths [N]
        输出到:
        - local_goals [N, 3]
        - projection_points [N, 3]
        - cross_track_errors [N]
        - current_dist_along_path [N]
        """
        if indices is None:
            if self.num_aircraft == 0:
                return
            indices = torch.arange(self.num_aircraft, device=self.device)
        if indices.numel() == 0:
            return

        positions_3d = self.positions[indices]                     # (K, 3)
        positions_2d = positions_3d[:, :2]                         # (K, 2)
        waypoints_xyz = self.waypoints[indices, :, :3]             # (K, M, 3)
        waypoints_2d = waypoints_xyz[:, :, :2]                     # (K, M, 2)
        waypoint_lengths = self.waypoint_lengths[indices]          # (K,)
        max_waypoints = waypoints_xyz.shape[1]

        # mask: path < 2 points
        short_path_mask = waypoint_lengths < 2
        long_path_mask = ~short_path_mask

        if torch.any(long_path_mask):
            wp_starts = waypoints_2d[long_path_mask, :-1, :]
            wp_ends = waypoints_2d[long_path_mask, 1:, :]
            segment_vecs = wp_ends - wp_starts
            segment_lens_sq = torch.sum(segment_vecs**2, dim=-1) + 1e-6
            to_current_vecs = positions_2d[long_path_mask].unsqueeze(1) - wp_starts
            projection_ratios = torch.einsum('nij,nij->ni', to_current_vecs, segment_vecs) / segment_lens_sq
            clamped_ratios = torch.clamp(projection_ratios, 0.0, 1.0)

            projection_points = wp_starts + clamped_ratios.unsqueeze(-1) * segment_vecs
            segment_indices = torch.arange(max_waypoints - 1, device=self.device).unsqueeze(0)
            valid_segment_mask = segment_indices < (waypoint_lengths[long_path_mask] - 1).unsqueeze(-1)

            cross_track_errors_sq = torch.sum((positions_2d[long_path_mask].unsqueeze(1) - projection_points)**2, dim=-1)
            cross_track_errors_sq[~valid_segment_mask] = float('inf')

            best_segment_indices = torch.argmin(cross_track_errors_sq, dim=1)
            best_ratios = torch.gather(clamped_ratios, 1, best_segment_indices.unsqueeze(-1)).squeeze(-1)

            start_points = torch.gather(wp_starts, 1, best_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            end_points = torch.gather(wp_ends, 1, best_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            final_proj_2d = start_points + best_ratios.unsqueeze(-1) * (end_points - start_points)

            # write projection and cross-track error
            self.projection_points[indices[long_path_mask], :2] = final_proj_2d
            self.projection_points[indices[long_path_mask], 2] = positions_3d[long_path_mask, 2]
            min_dist_sq = torch.gather(cross_track_errors_sq, 1, best_segment_indices.unsqueeze(-1)).squeeze(-1)
            self.cross_track_errors[indices[long_path_mask]] = torch.sqrt(min_dist_sq)

            # cumulative lengths along path
            segment_lengths = torch.norm(segment_vecs, dim=-1)
            segment_lengths[~valid_segment_mask] = 0.0
            cumulative_lengths = torch.cumsum(segment_lengths, dim=1)
            dist_to_segment_start = cumulative_lengths - segment_lengths

            gathered_dist_to_start = torch.gather(dist_to_segment_start, 1, best_segment_indices.unsqueeze(-1)).squeeze(-1)
            gathered_segment_len = torch.gather(segment_lengths, 1, best_segment_indices.unsqueeze(-1)).squeeze(-1)
            dist_to_projection = gathered_dist_to_start + best_ratios * gathered_segment_len

            # distance from projection to goal (total length - dist_to_projection)
            last_segment_indices = torch.clamp(waypoint_lengths[long_path_mask] - 2, min=0)
            total_path_lengths = torch.gather(cumulative_lengths, 1, last_segment_indices.unsqueeze(-1)).squeeze(-1)
            dist_projection_to_goal = torch.clamp(total_path_lengths - dist_to_projection, min=0.0)
            self.current_dist_along_path[indices[long_path_mask]] = dist_projection_to_goal

            # local goal
            target_dist_along_path = dist_to_projection + float(lookahead_distance)
            is_past_segment = target_dist_along_path.unsqueeze(1) > cumulative_lengths
            local_goal_segment_indices = torch.sum(is_past_segment, dim=1)
            max_valid_segment_idx = torch.clamp(waypoint_lengths[long_path_mask] - 2, min=0)
            local_goal_segment_indices = torch.min(local_goal_segment_indices, max_valid_segment_idx)

            dist_to_goal_segment_start = torch.gather(dist_to_segment_start, 1, local_goal_segment_indices.unsqueeze(-1)).squeeze(-1)
            dist_into_goal_segment = target_dist_along_path - dist_to_goal_segment_start

            goal_seg_starts = torch.gather(wp_starts, 1, local_goal_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            goal_seg_ends = torch.gather(wp_ends, 1, local_goal_segment_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            goal_seg_vecs = goal_seg_ends - goal_seg_starts
            goal_seg_lens = torch.norm(goal_seg_vecs, dim=-1, keepdim=True) + 1e-6
            ratio_on_goal_segment = (dist_into_goal_segment.unsqueeze(-1) / goal_seg_lens).clamp(0.0, 1.0)
            local_goals_2d = goal_seg_starts + ratio_on_goal_segment * goal_seg_vecs

            flight_altitude = waypoints_xyz[long_path_mask, 0, 2]
            self.local_goals[indices[long_path_mask], :2] = local_goals_2d
            self.local_goals[indices[long_path_mask], 2] = flight_altitude

            # update current waypoint indices (optional; closest next waypoint)
            new_wp_indices = local_goal_segment_indices + 1
            self.current_waypoint_indices[indices[long_path_mask]] = new_wp_indices

        if torch.any(short_path_mask):
            # fall back: local goal = target
            self.local_goals[indices[short_path_mask]] = self.target_positions[indices[short_path_mask]]
            self.projection_points[indices[short_path_mask]] = positions_3d[short_path_mask]
            self.cross_track_errors[indices[short_path_mask]] = 0.0
            # current index = last valid point (length-1) or 0
            short_idx = torch.clamp(waypoint_lengths[short_path_mask] - 1, min=0)
            self.current_waypoint_indices[indices[short_path_mask]] = short_idx
            # distance to target in 3D
            d = torch.norm(self.target_positions[indices[short_path_mask]] - positions_3d[short_path_mask], dim=-1)
            self.current_dist_along_path[indices[short_path_mask]] = d
