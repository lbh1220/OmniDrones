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
    

@dataclass
class NavigationNamespace:
    """导航相关状态命名空间"""
    # 目标和路径
    target_positions: torch.Tensor = None    # 目标位置 [num_envs, 1, 3]
    start_positions: torch.Tensor = None     # 起始位置 [num_envs, 1, 3]
    
    # 距离和状态
    current_dist_to_target: torch.Tensor = None  # 当前到目标距离 [num_envs]
    prev_dist_to_target: torch.Tensor = None     # 上一步到目标距离 [num_envs]
    
    # 任务状态
    reached_target_mask: torch.Tensor = None     # 到达目标mask [num_envs]
    

@dataclass
class CollisionNamespace:
    """碰撞检测相关状态命名空间"""
    collision_mask: torch.Tensor = None      # 碰撞mask [num_envs]
    safety_radius: float = 1.0               # 安全半径
    collision_objects: List[Optional[str]] = field(default_factory=list)  # 碰撞对象列表


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
    traffic_types: torch.Tensor = None         # [total_traffic] 类型索引
    traffic_safety_radius: torch.Tensor = None # [total_traffic]
    traffic_future_traj: torch.Tensor = None   # [total_traffic, predict_steps+1, 3]
    
    # 预测相关参数
    predict_steps: int = 5
    pred_timestep: float = 2.0


class EnvState:
    """统一的环境状态管理类，支持多个命名空间"""
    
    def __init__(self, device: str = "cuda", num_envs: int = 1):
        self.device = device
        self.num_envs = num_envs
        
        # 基础命名空间
        self.ego_drone = EgoDroneNamespace()
        self.navigation = NavigationNamespace() 
        self.collision = CollisionNamespace()
        self.mission = MissionNamespace()
        
        # 可选命名空间
        self.traffic = None  # 只在需要时初始化
        
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
        
        # 导航状态
        self.navigation.target_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.start_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.navigation.current_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.navigation.prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.navigation.reached_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 碰撞状态
        self.collision.collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision.collision_objects = [None] * self.num_envs
        
        # 任务状态
        self.mission.episode_progress = torch.zeros(self.num_envs, device=self.device)
        self.mission.success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mission.failure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def update_ego_drone_state(self, drone_state: torch.Tensor):
        """更新自车无人机状态"""
        self.ego_drone.drone_state = drone_state
        # 从完整状态中提取各个组件
        self.ego_drone.positions = drone_state[:, :, :3]
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
        if self.navigation.prev_dist_to_target is not None:
            self.navigation.prev_dist_to_target[env_ids] = 0.0
        if self.navigation.reached_target_mask is not None:
            self.navigation.reached_target_mask[env_ids] = False
            
        # 重置碰撞状态
        if self.collision.collision_mask is not None:
            self.collision.collision_mask[env_ids] = False
            
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
                    
        # 移动traffic张量（如果存在）
        if self.traffic is not None:
            for attr_name in dir(self.traffic):
                if not attr_name.startswith('_'):
                    attr_value = getattr(self.traffic, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self.traffic, attr_name, attr_value.to(device))
