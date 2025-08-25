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
    
    def initialize_aircraft(self, names: List[str], aircraft_types: List[str], 
                           safety_radius: List[float], max_speed: List[float], device: str = None):
        """初始化飞机列表"""
        if device is None:
            device = self.device
            
        self.num_aircraft = len(names)
        self.names = names.copy()
        self.aircraft_types = aircraft_types.copy()
        self.safety_radius = torch.tensor(safety_radius, device=device)
        self.time_stamps = torch.zeros(self.num_aircraft, device=device)
        self.max_speed = torch.tensor(max_speed, device=device)
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
