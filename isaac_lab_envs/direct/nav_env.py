# MIT License
#
# Copyright (c) 2023 Isaac Lab Nav Environment Implementation

"""Nav Navigation Environment using Direct RL Workflow

这个环境直接复用了原始OmniDrones项目中的robot系统，包括：
1. MultirotorBase、Firefly等完整的drone实现
2. 原始的旋翼动力学模型和RotorGroup
3. 原始的参数配置系统（yaml文件）
4. 原始的控制器系统
5. Isaac Lab的RayCaster激光雷达系统和DirectRLEnv框架

核心设计理念：
- 100% 复用原始drone系统的精心设计
- 使用Isaac Lab的DirectRLEnv作为基础框架
- 保持原始环境的所有功能和行为
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
import omni.isaac.lab.utils.math as math_utils

from omni_drones.traffic.cfg.config import AreaBoundsCfg

# 导入原始OmniDrones的robot系统
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


# 导入原始的TensorDict相关
from tensordict.tensordict import TensorDict


from isaac_lab_envs.direct.mdp.state import EnvState


##
# Pre-defined configs
##
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class NavEnvWindow(BaseEnvWindow):
    """Window manager for the Nav environment."""

    def __init__(self, env: NavEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    """Configuration for the Nav navigation environment."""

    seed = None
    
    # environment settings
    episode_length_s = 300.0
    decimation = 10  # env step every 1 sim steps
    num_actions = 2  # 只输出vx,vy
    num_observations = 7  # robot_node(5) + temporal_edges(2) = 7
    num_states = 0
    debug_vis = False
    debug_vis_num_envs = 10  # only visualize the first 5 environments
    is_training = True

    ui_window_class_type = NavEnvWindow
    viewer: ViewerCfg = field(default_factory=lambda: ViewerCfg(
        resolution=(960, 720),
        eye=(8, 0., 6.),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    ))
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 20,  # 60Hz simulation
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=8.0, replicate_physics=False)


    
    # drone配置 - 使用原始OmniDrones的配置系统
    drone_model: str = "hummingbird"  # 可以选择: firefly, crazyflie, hummingbird, iris等
    controller: Optional[str] = "PlanarSpeedController"  # 可以设置控制器，如 "lee_controller"
    # task config
    flight_height: float = 20.0
    safety_radius: float = 1.0
    arrival_threshold: float = 2.0
    max_speed: float = 1.0
    area_bounds: AreaBoundsCfg = field(default_factory=lambda: AreaBoundsCfg(
        xmin=-80.0,
        xmax=80.0,
        ymin=-80.0,
        ymax=80.0
    ))
    
    # action space config
    action_space_type: str = "beta" # discrete or beta or gaussian
    action_space_num_per_dim: int = 7  # 每个维度的离散动作数量
    action_mode: str = "speed_direction"  # "velocity_components" or "speed_direction"

        # 原始参数配置
    lidar_range: float = 4.0
    lidar_vfov: tuple[float, float] = (-10.0, 20.0)  # degrees
    lidar_resolution: tuple[int, int] = (36, 4)  # horizontal x vertical
    # 奖励权重（完全按照原始配置）
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_action_penalty = -0.1  # 动作惩罚系数（速度变化惩罚）

    
    # 随机化配置
    randomization: Dict = field(default_factory=dict)
    time_encoding: bool = True

    # global planner
    use_global_path: bool = True
    lookahead_distance: float = 10.0
    rew_cross_track_coeff = 0.0
    rew_cross_track_alpha = 1.0


class NavEnv(DirectRLEnv):
    """Nav navigation environment for drones using Direct RL workflow."""
    
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kwargs):
        # 保存配置参数（在父类初始化之前）
        # self.reward_effort_weight = cfg.reward_effort_weight  # 暂时不需要
        self.obs_processor = None
        self.reward_calculator = None
        self.time_encoding = cfg.time_encoding
        self.randomization = cfg.randomization
        self.has_payload = "payload" in self.randomization.keys()
        self.lidar_resolution = cfg.lidar_resolution
        # should seed before init, to control the initialization of drones and traffic
        if cfg.seed is not None:
            self.seed(cfg.seed)
        # 初始化观测和奖励处理器
        self._init_mdp_components(cfg)
        
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        
        # 在父类初始化完成后进行无人机特定的初始化
        self._post_init_setup()
        
        # 初始化状态管理对象
        self.state = EnvState(device=self.device, num_envs=self.num_envs)
        self.state.initialize_basic_tensors()
        self.state.collision.safety_radius = cfg.safety_radius

        
        # 设置离散动作空间映射
        if cfg.action_space_type == "discrete":
            self._setup_discrete_action()
        
        # 初始姿态分布
        self.init_rpy_dist = torch.distributions.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        
        # 保留command_vel_xy作为控制命令（不是状态的一部分）
        self.command_vel_xy = None
        
        self.extras = {
            "goal_reached": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "collision": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        if self.cfg.use_global_path:
            self.extras["cross_track_error_avg"] = torch.zeros(self.num_envs, device=self.device)
            self.extras["eposide_cross_error"] = torch.zeros(self.num_envs, device=self.device)
            
        # 加速度统计
        self.extras["acceleration_avg"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["episode_acceleration"] = torch.zeros(self.num_envs, device=self.device)
            
        # debug可视化
        if self.sim.has_gui():
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
        else:
            self.debug_draw = None
        self.alpha = 0.8
        self.circle_radius = min(self.cfg.area_bounds.xmax - self.cfg.area_bounds.xmin, 
                                self.cfg.area_bounds.ymax - self.cfg.area_bounds.ymin)/2.0
        
        # debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _init_mdp_components(self, cfg: NavEnvCfg):
        """初始化模块化组件"""
        from isaac_lab_envs.direct.mdp.observations import NavObservationProcessor
        from isaac_lab_envs.direct.mdp.rewards import NavRewardCalculator
        self.obs_processor = NavObservationProcessor(cfg)
        self.reward_calculator = NavRewardCalculator(cfg)

    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors."""
        print(f"设置场景，使用无人机：{self.cfg.drone_model}")
        
        # 1. 首先创建无人机系统（但还不生成）
        self.drone, self.controller = MultirotorBase.make(self.cfg.drone_model, self.cfg.controller, self.device)
        
        # 2. 在模板环境中生成一个无人机
        translations = [(0.0, 0.0, 2.0)]
        drone_prims = self.drone.spawn(translations)
        print(f"在模板环境中生成无人机: {drone_prims}")

        # 3. 设置地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 4. 设置传感器（在克隆之前）
        self._setup_lidar()
        
        # 5. 设置光照
        self._setup_lights()

        # 6. 克隆环境并处理碰撞
        print("克隆环境...")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        print("环境克隆完成")

    def _post_init_setup(self):
        """在场景设置完成后进行无人机系统初始化"""
        print("开始无人机后初始化...")
        
        # 设置无人机的shape以匹配环境数量
        self.drone.shape = (self.num_envs, 1)
        
        # 初始化无人机系统
        print("初始化无人机系统...")
        self.drone.initialize()
        
        # 设置随机化
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        
        # 初始化LiDAR
        print("初始化LiDAR...")
        self._lidar._initialize_impl()
        
        # 获取初始状态
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        # 设置环境位置偏移
        self.drone._envs_positions = self._terrain.env_origins.unsqueeze(1)
        
        print(f"无人机初始化完成，shape: {self.drone.shape}")
        print(f"环境数量: {self.num_envs}")
        print(f"无人机位置形状: {self.drone._envs_positions.shape if hasattr(self.drone, '_envs_positions') else 'None'}")

        # 设置处理器的设备
        self.obs_processor.device = self.device
        self.reward_calculator.device = self.device
        

    def _setup_lidar(self):
        """Setup the lidar sensor exactly like original implementation."""
        # 转换垂直FOV角度
        lidar_vfov_rad = (
            max(-89.0, self.cfg.lidar_vfov[0]) * math.pi / 180.0,
            min(89.0, self.cfg.lidar_vfov[1]) * math.pi / 180.0
        )
        
        # 创建垂直射线角度（完全按照原始配置）
        vertical_ray_angles = torch.linspace(lidar_vfov_rad[0], lidar_vfov_rad[1], 4)

        # 注意：这里需要匹配原始的prim路径格式
        # 原始使用 "/World/envs/env_.*/Hummingbird_0/base_link"
        # 我们需要根据实际的无人机类型调整
        ray_caster_cfg = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/{self.cfg.drone_model.capitalize()}_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                vertical_ray_angles=vertical_ray_angles
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            max_distance=self.cfg.lidar_range,
        )
        self._lidar = RayCaster(ray_caster_cfg)
    
    def _setup_lights(self):
        """Setup lights exactly like original implementation."""
        # Distant light
        light_cfg = sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        light_cfg.func("/World/light", light_cfg)
        
        # Sky light  
        sky_light_cfg = sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0)
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions to the drone using original apply_action method."""
        # 处理离散动作空间
        if self.cfg.action_space_type == "discrete":
            # 将离散动作转换为连续动作
            continuous_actions = self.discrete_to_continuous_action(actions)
            if self.cfg.action_mode == "speed_direction":
                raise ValueError("Speed direction action mode not supported for discrete action space")
        else:
            continuous_actions = actions
        
        # 根据action_mode处理连续动作
        if self.cfg.action_mode == "velocity_components":
            # 原模式：vx, vy速度分量
            self.command_vel_xy = self._process_velocity_components(continuous_actions)
        elif self.cfg.action_mode == "speed_direction":
            # 新模式：速度大小 + 方向
            self.command_vel_xy = self._process_speed_direction(continuous_actions)
        else:
            raise ValueError(f"Unknown action mode: {self.cfg.action_mode}")
        
        # 确保速度在合理范围内 - 按模长缩放而非简单裁切
        speed_magnitude = torch.norm(self.command_vel_xy, dim=-1, keepdim=True)
        # 如果模长超过max_speed，则按比例缩放到max_speed
        scale_factor = torch.where(
            speed_magnitude > self.cfg.max_speed,
            self.cfg.max_speed / speed_magnitude,
            torch.ones_like(speed_magnitude)
        )
        self.command_vel_xy = self.command_vel_xy * scale_factor
        self.command_vel_xy = self.command_vel_xy.unsqueeze(1)
        self.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
    
    def _process_velocity_components(self, actions: torch.Tensor) -> torch.Tensor:
        """处理速度分量模式的动作"""
        if self.cfg.action_space_type == "beta":
            # Beta分布输出[0,1]，需要映射到[-1,1]然后乘以max_speed
            actions_scaled = (actions * 2.0) - 1.0  # [0,1] -> [-1,1]
            command_vel_xy = actions_scaled * self.cfg.max_speed
        else:
            # 高斯分布输出[-1,1]，直接乘以max_speed
            command_vel_xy = actions * self.cfg.max_speed
        
        return command_vel_xy
    
    def _process_speed_direction(self, actions: torch.Tensor) -> torch.Tensor:
        """处理速度大小+方向模式的动作"""
        if self.cfg.action_space_type == "beta":
            # Beta分布输出[0,1]
            # 第一个维度：速度大小，直接乘以max_speed
            speed = actions[:, 0] * self.cfg.max_speed
            # 第二个维度：方向，映射到[0, 2π]
            direction = actions[:, 1] * 2.0 * math.pi
        else:
            # 高斯分布输出[-1,1]
            # 第一个维度：速度大小，从[-1,1]映射到[0,1]再乘以max_speed
            speed = ((actions[:, 0] + 1.0) / 2.0) * self.cfg.max_speed
            # 第二个维度：方向，从[-1,1]映射到[0, 2π]
            direction = (actions[:, 1] + 1.0) / 2.0 * 2.0 * math.pi
        
        # 将极坐标转换为笛卡尔坐标
        vx = speed * torch.cos(direction)
        vy = speed * torch.sin(direction)
        
        # 组合成velocity命令
        command_vel_xy = torch.stack([vx, vy], dim=1)
        
        return command_vel_xy

    def _apply_action(self):
        """Actions are applied in _pre_physics_step."""
        drone_state = self.drone.get_state(env_frame=False)[..., :13]#[num_envs, N, 13]
        if self.command_vel_xy is None:
            self.command_vel_xy = torch.zeros(self.num_envs, 1, 2, device=self.device)
            self.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        target_height = self.cfg.flight_height * torch.ones(self.num_envs, 1, 1, device=self.device)
        rotor_commands = self.controller.compute(
            root_state=drone_state,  # shape [num_envs, N, 3]
            target_vel_xy=self.command_vel_xy,  # shape [num_envs, N, 2]
            target_height=target_height,  # shape [num_envs, N, 1]
        ) 
        self.drone.apply_action(rotor_commands)
    
    def _post_physics_step(self, env_ids: torch.Tensor = None):
        """
        Update sensors after physics step.
        direct rl env中没有这个函数
        """
        # 如果放在apply action之后，那更新太频繁了
        # 暂时放在get dones之前和reset_idx之后
        self._lidar.update(self.step_dt)
        
        # 更新状态对象
        drone_state = self.drone.get_state(env_frame=False)  # [num_envs, 1, 25]
        self.state.update_ego_drone_state(drone_state)
        self.state.update_navigation_distances()
        self.state.update_reached_target_mask(self.cfg.arrival_threshold)
        if self.cfg.use_global_path:
            # self.state.update_navigation_state_iterative(self.cfg.lookahead_distance, env_ids)
            self.state.update_navigation_state_vectorized(self.cfg.lookahead_distance, env_ids)
            # pass
            
            # 更新cross_track_error的累积平均值
            self._update_cross_track_error_avg()
            
        # 更新acceleration的累积平均值（使用previous和current velocity）
        self._update_acceleration_avg()

    def _update_cross_track_error_avg(self):
        """更新cross_track_error的累积平均值"""
        if (self.state.navigation.cross_track_errors is not None and 
            self.cfg.use_global_path):
            
            # 当前episode的步数 (从1开始计数)
            current_step = self.episode_length_buf + 1  # [num_envs]
            
            # 当前cross_track_error
            current_errors = self.state.navigation.cross_track_errors  # [num_envs]
            
            # 递增平均值公式: new_avg = (old_avg * (n-1) + new_value) / n
            old_avg = self.extras["cross_track_error_avg"]  # [num_envs]
            new_avg = (old_avg * (current_step - 1) + current_errors) / current_step
            
            self.extras["cross_track_error_avg"] = new_avg

    def _update_acceleration_avg(self):
        """更新acceleration的累积平均值"""
        # 当前episode的步数 (从1开始计数)
        current_step = self.episode_length_buf + 1  # [num_envs]
        
        # 计算当前加速度（速度变化的模）
        if (self.state.ego_drone.velocities is not None and 
            self.state.ego_drone.previous_velocities is not None):
            
            current_velocity = self.state.ego_drone.velocities   # [num_envs, 1, 3]
            previous_velocity = self.state.ego_drone.previous_velocities   # [num_envs, 1, 3]
            
            # 计算速度变化（加速度）
            velocity_change = current_velocity - previous_velocity  # [num_envs, 1, 3]
            
            # 计算速度变化的模（L2范数）
            current_acceleration = torch.norm(velocity_change.squeeze(1), dim=1)  # [num_envs]
            
            # 递增平均值公式: new_avg = (old_avg * (n-1) + new_value) / n
            old_avg = self.extras["acceleration_avg"]  # [num_envs]
            new_avg = (old_avg * (current_step - 1) + current_acceleration) / current_step
            
            self.extras["acceleration_avg"] = new_avg

    def _get_observations(self) -> dict:
        """计算基于字典格式的导航观测。"""
        # 使用观测处理器计算观测
        observations = self.obs_processor.process_observation(self.state)
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算基于2D导航的奖励。"""
        # 使用奖励计算器计算奖励
        reward = self.reward_calculator.compute_reward(self.state)
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算基于2D导航的终止条件。"""
        self._post_physics_step()
        
        # 从状态对象获取数据
        reached_target = self.state.navigation.reached_target_mask
        self.extras["goal_reached"] = reached_target
        
        # 3. 高度异常条件（保持在合理高度范围内）
        robot_height = self.state.ego_drone.positions.squeeze(1)[:, 2]  # [num_envs]
        height_abnormal = (
            (robot_height < (self.cfg.flight_height - 3*self.cfg.safety_radius)) |
            (robot_height > (self.cfg.flight_height + 3*self.cfg.safety_radius))
        )
        
        # 4. NaN检测
        hasnan = torch.isnan(self.state.ego_drone.drone_state).any(dim=(1, 2))
        
        # 终止条件：到达目标、高度异常或NaN
        terminated = reached_target | height_abnormal | hasnan
        
        # 超时条件：由DirectRLEnv框架自动处理
        truncated = self.episode_length_buf >= self.max_episode_length 
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments exactly like original implementation."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 使用原始无人机系统的重置方法
        self.drone._reset_idx(env_ids, self.cfg.is_training)

        
        # 重置统计

        
        # 重置机器人到初始位置（完全按照原始实现）
        if self.cfg.use_global_path:
            start, goal, waypoints = self._generate_crossing_task_with_waypoints(len(env_ids), flight_height=self.cfg.flight_height)
            self.state.navigation.waypoints[env_ids] = waypoints
            self.state.navigation.waypoint_lengths[env_ids] = waypoints.shape[1]
            self.state.navigation.current_waypoint_indices[env_ids] = 0
            # TODO, 应该为每个env配置他的实际waypoints长度
        else:
            start, goal = self._generate_crossing_task(len(env_ids), flight_height=self.cfg.flight_height)
        
        # 随机初始姿态（使用原始分布）
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        
        # 更新状态对象中的目标位置
        self.state.navigation.target_positions[env_ids] = goal
        self.state.navigation.start_positions[env_ids] = start
        
        
        
        # 设置位置和姿态（使用原始方法）
        self.drone.set_world_poses(start, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        # 重置状态对象
        self.state.reset_env_states(env_ids)
        
        # 重置cross_track_error累积平均值
        if self.cfg.use_global_path:
            self.extras["eposide_cross_error"][env_ids] = self.extras["cross_track_error_avg"][env_ids].clone()
            self.extras["cross_track_error_avg"][env_ids] = 0.0
            
        # 重置acceleration累积平均值
        self.extras["episode_acceleration"][env_ids] = self.extras["acceleration_avg"][env_ids].clone()
        self.extras["acceleration_avg"][env_ids] = 0.0
        
        # 更新状态信息
        self._post_physics_step(env_ids=env_ids)
        
        # 重置奖励计算器的势能缓存
        self.reward_calculator.reset_potential(self.state, env_ids)
        
        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "drone_pos_visualizer"):
                # Create green marker for drone positions
                drone_marker_cfg = CUBOID_MARKER_CFG.copy()
                drone_marker_cfg.markers["cuboid"].size = (2.0, 2.0, 2.0)
                drone_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # Green color
                drone_marker_cfg.prim_path = "/Visuals/Command/drone_position"
                self.drone_pos_visualizer = VisualizationMarkers(drone_marker_cfg)
            if self.cfg.use_global_path:
                if not hasattr(self, "local_goal_visualizer"):
                    # Create blue marker for local goals
                    local_goal_marker_cfg = CUBOID_MARKER_CFG.copy()
                    local_goal_marker_cfg.markers["cuboid"].size = (0.18, 0.18, 0.18)
                    local_goal_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)  # Blue color
                    local_goal_marker_cfg.prim_path = "/Visuals/Command/local_goal"
                    self.local_goal_visualizer = VisualizationMarkers(local_goal_marker_cfg)

                if not hasattr(self, "projection_point_visualizer"):
                    # Create orange marker for projection points
                    projection_point_marker_cfg = CUBOID_MARKER_CFG.copy()
                    projection_point_marker_cfg.markers["cuboid"].size = (0.12, 0.12, 0.12)
                    projection_point_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.5, 0.0)  # Orange color
                    projection_point_marker_cfg.prim_path = "/Visuals/Command/projection_point"
                    self.projection_point_visualizer = VisualizationMarkers(projection_point_marker_cfg)

            
            # 确保它是可见的
            self.target_pos_visualizer.set_visibility(True)
            self.drone_pos_visualizer.set_visibility(True)
            if self.cfg.use_global_path:
                # Only show local goals and projection points when using global path
                self.local_goal_visualizer.set_visibility(self.cfg.use_global_path)
                self.projection_point_visualizer.set_visibility(self.cfg.use_global_path)
        else:
            if hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer.set_visibility(False)
            if hasattr(self, "drone_pos_visualizer"):
                self.drone_pos_visualizer.set_visibility(False)
            if hasattr(self, "local_goal_visualizer"):
                self.local_goal_visualizer.set_visibility(False)
            if hasattr(self, "projection_point_visualizer"):
                self.projection_point_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "target_pos_visualizer"):
            # only visualize the first 10 targets
            if self.state.navigation.target_positions.shape[0] < self.cfg.debug_vis_num_envs:
                vis_target_pos = self.state.navigation.target_positions.squeeze(1)
            else:
                vis_target_pos = self.state.navigation.target_positions.squeeze(1)[:self.cfg.debug_vis_num_envs]
            self.target_pos_visualizer.visualize(vis_target_pos)
        
        if hasattr(self, "drone_pos_visualizer"):
            # Visualize drone positions - squeeze to remove the middle dimension (n_env, 3)
            # only visualize the first 10 drones
            # but need to in case drones are less than 10
            if self.drone.pos.shape[0] < self.cfg.debug_vis_num_envs:
                vis_drone_pos = self.drone.pos.squeeze(1)
            else:
                vis_drone_pos = self.drone.pos.squeeze(1)[:self.cfg.debug_vis_num_envs]
            self.drone_pos_visualizer.visualize(vis_drone_pos)
            
        # Visualize local goals and projection points (only when using global path)
        if self.cfg.use_global_path:
            if hasattr(self, "local_goal_visualizer") and self.state.navigation.local_goals is not None:
                if self.state.navigation.local_goals.shape[0] < self.cfg.debug_vis_num_envs:
                    vis_local_goals = self.state.navigation.local_goals.squeeze(1)
                else:
                    vis_local_goals = self.state.navigation.local_goals.squeeze(1)[:self.cfg.debug_vis_num_envs]
                self.local_goal_visualizer.visualize(vis_local_goals)
            
            if hasattr(self, "projection_point_visualizer") and self.state.navigation.projection_points is not None:
                if self.state.navigation.projection_points.shape[0] < self.cfg.debug_vis_num_envs:
                    vis_projection_points = self.state.navigation.projection_points.squeeze(1)
                else:
                    vis_projection_points = self.state.navigation.projection_points.squeeze(1)[:self.cfg.debug_vis_num_envs]
                self.projection_point_visualizer.visualize(vis_projection_points)

    def _generate_crossing_task(self, num_env: int = 1, flight_height: float = 20.0):
        if num_env <= 0:
            raise ValueError("num_aircraft must be greater than 0")
        area_center = torch.tensor([(self.cfg.area_bounds.xmin + self.cfg.area_bounds.xmax) / 2, 
                                    (self.cfg.area_bounds.ymin + self.cfg.area_bounds.ymax) / 2, 
                                    flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)
        start_tensor = math_utils.sample_cylinder(self.circle_radius, (0, 0), num_env, self.device)
        goal_tensor = start_tensor.clone()
        goal_tensor = -goal_tensor
        start_tensor = start_tensor + area_center
        goal_tensor = goal_tensor + area_center

        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1)
    
    def _generate_crossing_task_with_waypoints(self, num_env: int = 1, flight_height: float = 20.0):
        if num_env <= 0:
            raise ValueError("num_aircraft must be greater than 0")
        area_center = torch.tensor([(self.cfg.area_bounds.xmin + self.cfg.area_bounds.xmax) / 2, 
                                    (self.cfg.area_bounds.ymin + self.cfg.area_bounds.ymax) / 2, 
                                    flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)
        start_tensor = math_utils.sample_cylinder(self.circle_radius, (0, 0), num_env, self.device)
        goal_tensor = start_tensor.clone()
        goal_tensor = -goal_tensor
        start_tensor = start_tensor + area_center
        goal_tensor = goal_tensor + area_center

        inter_points = math_utils.sample_cylinder(self.circle_radius/2.0, (0, 0), num_env, self.device)
        inter_points = inter_points + area_center
        waypoints = torch.cat([start_tensor.unsqueeze(1), inter_points.unsqueeze(1), goal_tensor.unsqueeze(1)], dim=1)
        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), waypoints

    
    def _setup_discrete_action(self):
        """设置离散动作空间"""
        total_actions = self.cfg.action_space_num_per_dim * self.cfg.action_space_num_per_dim
        self._create_discrete_action_mapping()
        print(f"Created discrete action mapping: {self.cfg.action_space_num_per_dim}x{self.cfg.action_space_num_per_dim} = {total_actions} actions")
    
    def _create_discrete_action_mapping(self):
        """创建离散动作映射"""
        speed_values = torch.linspace(-1.0, 1.0, self.cfg.action_space_num_per_dim, device=self.device)
        
        action_mapping = []
        for i in range(self.cfg.action_space_num_per_dim):
            for j in range(self.cfg.action_space_num_per_dim):
                vx = speed_values[i]
                vy = speed_values[j]
                action_mapping.append([vx, vy])
        
        self.action_mapping = torch.tensor(action_mapping, device=self.device, dtype=torch.float32)
    
    def discrete_to_continuous_action(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        """将离散动作转换为连续动作
        
        Args:
            discrete_actions: [num_envs] or [num_envs, 1] 离散动作索引
            
        Returns:
            continuous_actions: [num_envs, 2] 连续动作 (vx, vy)
        """
        # 将float32转换为整数索引（处理vec env的numpy/tensor转换）
        if discrete_actions.ndim == 2:
            discrete_actions = discrete_actions.squeeze(1)
        discrete_actions = discrete_actions.long()
        
        # 处理超出范围的动作索引
        discrete_actions = torch.clamp(discrete_actions, 0, len(self.action_mapping) - 1)
        
        # 批量索引映射
        continuous_actions = self.action_mapping[discrete_actions]
        
        return continuous_actions
    
    @property 
    def _should_render(self):
        """Check if should render for debug visualization."""
        return lambda substep: self.sim.has_gui() and substep == 0 

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        super()._configure_gym_env_spaces()
        import gymnasium as gym
        import numpy as np
        
        # 使用观测处理器生成观测空间字典
        policy_space_dict = self.obs_processor.generate_policy_obs_dict()
        
        # 2. 将内层字典包装成一个 gym.spaces.Dict
        policy_space = gym.spaces.Dict(policy_space_dict)

        # 3. 创建最外层的观测空间字典
        self.single_observation_space["policy"] = policy_space


        # bound action space
        if self.cfg.action_space_type == "discrete":
            total_actions = self.cfg.action_space_num_per_dim * self.cfg.action_space_num_per_dim
            self.single_action_space = gym.spaces.Discrete(total_actions)
        elif self.cfg.action_space_type == "beta":
            self.single_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_actions,))
        else:
            self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
