# MIT License
#
# Copyright (c) 2023 Isaac Lab Nav Environment Implementation

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
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
import omni.isaac.lab.utils.math as math_utils

from isaac_lab_envs.traffic.cfg.config import AreaBoundsCfg

# 导入原始OmniDrones的robot系统
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


# 导入原始的TensorDict相关
from tensordict.tensordict import TensorDict


from isaac_lab_envs.direct.mdp.state import EnvState
from isaac_lab_envs.direct.components.metrics import MetricsManager, CrossTrackModule, AccelerationModule, FlagsModule
from isaac_lab_envs.direct.mdp.action import ActionManagerCfg, VelocityXYActionManager


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

    seed = 42
    
    # environment settings
    episode_length_s = 300.0
    decimation = 10  # env step every 1 sim steps
    num_actions = 2  # 只输出vx,vy
    num_observations = 7  # robot_node(5) + temporal_edges(2) = 7
    num_states = 0
    debug_vis = False
    debug_vis_num_envs = 10  # only visualize the first 5 environments
    is_training = True

    total_timesteps = 10000000

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
    min_speed: float = 0.0
    v_pref: float = 1.0 # preferred speed
    area_bounds: AreaBoundsCfg = field(default_factory=lambda: AreaBoundsCfg(
        xmin=-50.0,
        xmax=50.0,
        ymin=-50.0,
        ymax=50.0
    ))
    
    action_manager: ActionManagerCfg = ActionManagerCfg(
        action_space_type="discrete",
        action_space_num_per_dim=7,
        action_mode="velocity_components"
    )

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
    max_waypoints: int = 3
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
        self.action_manager = VelocityXYActionManager(cfg.action_manager, self)
        self._init_mdp_components(cfg)
        
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        
        # 在父类初始化完成后进行无人机特定的初始化
        self._post_init_setup()
        

        self.state.collision.safety_radius = cfg.safety_radius

        
        
        # 初始姿态分布
        self.init_rpy_dist = torch.distributions.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        
        self.extras = {
            "goal_reached": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "collision": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        self.extras["is_success"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            
        # debug可视化
        if self.sim.has_gui():
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
        else:
            self.debug_draw = None
        self.alpha = 0.8
        self.circle_radius = min(self.cfg.area_bounds.xmax - self.cfg.area_bounds.xmin, 
                                self.cfg.area_bounds.ymax - self.cfg.area_bounds.ymin)/2.0
        
        
        self._init_metrics()

        # debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _init_mdp_components(self, cfg: NavEnvCfg):
        """初始化模块化组件"""
        from isaac_lab_envs.direct.mdp.observations import NavObservationProcessor
        from isaac_lab_envs.direct.mdp.rewards import NavRewardCalculator
        self.obs_processor = NavObservationProcessor(cfg)
        self.reward_calculator = NavRewardCalculator(cfg)
        

    def _init_metrics(self):
        # 初始化 Metrics 管理器并注册模块
        self.metrics = MetricsManager(num_envs=self.num_envs, device=self.device)
        self.metrics.bind_env(self)
        self.metrics.register(FlagsModule())
        self.metrics.register(CrossTrackModule())
        self.metrics.register(AccelerationModule())

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

        # 初始化状态管理对象
        self.state = EnvState(device=self.device, num_envs=self.num_envs)
        self.state.initialize_basic_tensors()
        if self.cfg.use_global_path:
            self.state.initialize_navigation_waypoints(max_wps=self.cfg.max_waypoints)

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
        """Delegate action processing to action manager."""
        self.action_manager.process_actions(actions)
    

    def _apply_action(self):
        self.action_manager.apply_action()
    
    def _post_physics_step(self, env_ids: torch.Tensor = None):
        """
        Update sensors after physics step.
        direct rl env中没有这个函数
        """
        # 如果放在apply action之后，那更新太频繁了
        # 暂时放在get dones之前和reset_idx之后

        
        # 更新状态对象
        drone_state = self.drone.get_state(env_frame=False)  # [num_envs, 1, 25]
        self.state.update_ego_drone_state(drone_state)
        self.state.update_navigation_distances()
        self.state.update_reached_target_mask(self.cfg.arrival_threshold)
        if self.cfg.use_global_path:
            # self.state.update_navigation_state_iterative(self.cfg.lookahead_distance, env_ids)
            self.state.update_navigation_state_vectorized(self.cfg.lookahead_distance, env_ids)
        # metrics modules are triggered only on_done

    def _get_observations(self) -> dict:
        """计算基于字典格式的导航观测。"""
        # 使用观测处理器计算观测
        observations = self.obs_processor.process_observation(self.state)
        # write to mdp state for cross-component access
        self.state.set_observations(observations)
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算基于2D导航的奖励。"""
        # 使用奖励计算器计算奖励
        reward = self.reward_calculator.compute_reward(self.state)
        # write to mdp state
        self.state.set_reward(reward)
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算基于Traffic环境的终止条件，包括碰撞检测。"""
        self._post_physics_step()

        reached_target_mask = self.state.navigation.reached_target_mask
        collision_mask = self.state.collision.collision_mask
        # 更新统计信息
        self.extras["goal_reached"] = reached_target_mask.clone()
        self.extras["collision"] = collision_mask.clone()   
        self.extras["is_success"] = reached_target_mask.clone()
        # 3. 高度异常条件（保持在合理高度范围内）
        robot_height = self.state.ego_drone.positions.squeeze(1)[:, 2]  # [num_envs]
        height_abnormal = (
            (robot_height < (self.cfg.flight_height - 3*self.cfg.safety_radius)) |
            (robot_height > (self.cfg.flight_height + 3*self.cfg.safety_radius))
        )
        
        # 4. NaN检测
        hasnan = torch.isnan(self.state.ego_drone.drone_state).any(dim=(1, 2))
        
        # 终止条件：到达目标、碰撞、高度异常或NaN
        terminated = reached_target_mask | collision_mask | height_abnormal | hasnan
        # 超时条件：由DirectRLEnv框架自动处理
        truncated = self.episode_length_buf >= self.max_episode_length 
        
        # 在返回之前，通知 metrics 管理器（用于写入 episode/rolling 指标）
        self.metrics.on_done(terminated, truncated)
        # write to mdp state
        self.state.set_dones(terminated, truncated)
        
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
            start, goal, waypoints, waypoints_length = self._generate_crossing_task_with_waypoints(len(env_ids), flight_height=self.cfg.flight_height)
            self.state.navigation.waypoints[env_ids] = waypoints
            self.state.navigation.waypoint_lengths[env_ids] = waypoints_length
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
        # # 重置 MDP 状态, 不应该重置，这会导致reward清零，但是本步本身是有上个eposide的reward的
        # self.state.reset_mdp(env_ids)
        

        
        # 更新状态信息
        self._post_physics_step(env_ids=env_ids)
        
        # 重置奖励计算器的势能缓存
        self.reward_calculator.reset_potential(self.state, env_ids)
        
        super()._reset_idx(env_ids)
        self.metrics.on_reset(env_ids)
    
    
    def _detect_collisions(self) -> torch.Tensor:
        """
        Nav env中没有障碍物, 为了和子类对齐，返回全zero的collision_mask
        """
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return collision_mask

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
                drone_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/drone_position",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=1.0,  # 设置球体的半径
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 1.0, 0.0),
                                    opacity=0.5
                                ),
                            )
                        },
                    )
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
        waypoints_length = torch.full((num_env,), 3, device=self.device)
        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), waypoints, waypoints_length

    
    # Discrete action mapping is now managed by the ActionManager
    
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

        self.single_action_space = self.action_manager.get_action_space()

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
