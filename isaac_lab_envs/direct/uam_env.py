# MIT License
#
# Copyright (c) 2023 Isaac Lab Nav Environment Implementation

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Dict, Any, Optional, Type
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
from isaac_lab_envs.direct.mdp.action import ActionManagerCfg, VelocityXYActionManager, AccelerationActionManager
from isaac_lab_envs.direct.mdp.rewards import RewardManagerCfg, RewardManager
from isaac_lab_envs.direct.mdp.observations import ObservationManagerCfg, ObservationManager
from isaac_lab_envs.direct.components.map import MapManagerCfg, MapManager
from isaac_lab_envs.direct.components.metrics import MetricsManager, CrossTrackModule, AccelerationModule, FlagsModule, NearCollisionModule
from isaac_lab_envs.direct.components.task_generator import CrossTaskGenerator
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner, GlobalPathPlannerCfg
from isaac_lab_envs.traffic.traffic_simulator import TrafficSimulator
from isaac_lab_envs.direct.components.visualization import VisualizationManager
from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


def take_scale(cfg):
    if cfg.env_scale > 1.0:
        cfg.flight_height = cfg.flight_height * cfg.env_scale
        cfg.safety_radius = cfg.safety_radius * cfg.env_scale
        cfg.v_pref = cfg.v_pref * cfg.env_scale
        cfg.arrival_threshold = cfg.arrival_threshold * cfg.env_scale
        cfg.max_speed = cfg.max_speed * cfg.env_scale
        cfg.min_speed = cfg.min_speed * cfg.env_scale
        cfg.dv_limit = cfg.dv_limit * cfg.env_scale
        cfg.area_bounds.xmax = cfg.area_bounds.xmax * cfg.env_scale
        cfg.area_bounds.xmin = cfg.area_bounds.xmin * cfg.env_scale
        cfg.area_bounds.ymax = cfg.area_bounds.ymax * cfg.env_scale
        cfg.area_bounds.ymin = cfg.area_bounds.ymin * cfg.env_scale
        cfg.area_bounds.grid_size = cfg.area_bounds.grid_size * cfg.env_scale
        cfg.lidar_range = cfg.lidar_range * cfg.env_scale
        if cfg.global_path_planner_cfg is not None:
            cfg.global_path_planner_cfg.lookahead_distance = cfg.global_path_planner_cfg.lookahead_distance * cfg.env_scale
        # 暂时没有用mapmanager, 等用到地面建筑物的时候，需要修改
        if cfg.traffic_sim is not None:
            cfg.traffic_sim.evtol.safety_radius = cfg.traffic_sim.evtol.safety_radius * cfg.env_scale
            cfg.traffic_sim.drone.safety_radius = cfg.traffic_sim.drone.safety_radius * cfg.env_scale
            cfg.traffic_sim.evtol.max_speed = cfg.traffic_sim.evtol.max_speed * cfg.env_scale
            cfg.traffic_sim.evtol.v_pref = cfg.traffic_sim.evtol.v_pref * cfg.env_scale
            cfg.traffic_sim.drone.max_speed = cfg.traffic_sim.drone.max_speed * cfg.env_scale
            cfg.traffic_sim.drone.v_pref = cfg.traffic_sim.drone.v_pref * cfg.env_scale
            cfg.traffic_sim.drone.arrival_threshold = cfg.traffic_sim.drone.arrival_threshold * cfg.env_scale
            cfg.traffic_sim.evtol.arrival_threshold = cfg.traffic_sim.evtol.arrival_threshold * cfg.env_scale
            cfg.traffic_sim.drone.max_offset_radius = cfg.traffic_sim.drone.max_offset_radius * cfg.env_scale
            cfg.traffic_sim.evtol.turn_radius = cfg.traffic_sim.evtol.turn_radius * cfg.env_scale
            cfg.traffic_sim.drone.lookahead_distance = cfg.traffic_sim.drone.lookahead_distance * cfg.env_scale

            cfg.traffic_sim.area_bounds.xmax = cfg.traffic_sim.area_bounds.xmax * cfg.env_scale
            cfg.traffic_sim.area_bounds.xmin = cfg.traffic_sim.area_bounds.xmin * cfg.env_scale
            cfg.traffic_sim.area_bounds.ymax = cfg.traffic_sim.area_bounds.ymax * cfg.env_scale
            cfg.traffic_sim.area_bounds.ymin = cfg.traffic_sim.area_bounds.ymin * cfg.env_scale
            cfg.traffic_sim.area_bounds.grid_size = cfg.traffic_sim.area_bounds.grid_size * cfg.env_scale
            cfg.traffic_sim.flight_height = cfg.traffic_sim.flight_height * cfg.env_scale
            # ORCA的参数没有改，因为ORCA不能搞放缩
            cfg.traffic_sim.orca.neighbor_dist = cfg.traffic_sim.orca.neighbor_dist * cfg.env_scale
            # safety_space最好别动
        # cfg.
    return cfg




class UamEnv(DirectRLEnv):
    """Nav navigation environment for drones using Direct RL workflow."""
    
    cfg: UamEnvCfg

    def __init__(self, cfg: UamEnvCfg, render_mode: str | None = None, resolution: int = 960, **kwargs):
        cfg = take_scale(cfg)
        self.time_encoding = cfg.time_encoding
        self.randomization = cfg.randomization
        self.lidar_resolution = cfg.lidar_resolution
        # should seed before init, to control the initialization of drones and traffic
        if cfg.seed is not None:
            self.seed(cfg.seed)
        # 初始化观测和奖励处理器
        # debug visualization
        range_x = cfg.area_bounds.xmax - cfg.area_bounds.xmin
        flight_height = cfg.flight_height

        cfg.viewer = ViewerCfg(
            resolution=(resolution, resolution),
            eye=(0, 0.0, range_x*1+flight_height),
            lookat=(0., 0., 1.)
        )
        # cfg.viewer = ViewerCfg(
        #     resolution=(1080, 1080),
        #     eye=(range_x*1.2, 0.0, range_x*1.2),
        #     lookat=(0., 0., 1.)
        # )
        self._init_mdp_components(cfg)
        
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)

        # 在父类初始化完成后进行无人机特定的初始化
        self._post_init_setup()

        # 初始姿态分布
        self.init_rpy_dist = torch.distributions.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        
        self.extras = {
            "goal_reached": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "collision": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        self.extras["is_success"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        self.visualization_manager = None
        if self.cfg.debug_vis:
            self.visualization_manager = VisualizationManager(self)
        self._init_metrics()


        self.set_debug_vis(self.cfg.debug_vis)

    def _init_mdp_components(self, cfg):
        """初始化模块化组件"""
        if cfg.action_manager.action_mode == "velocity_components":
            self.action_manager = VelocityXYActionManager(cfg.action_manager, self)
        elif cfg.action_manager.action_mode == "direction_acceleration":
            self.action_manager = AccelerationActionManager(cfg.action_manager, self)
        else:
            raise ValueError(f"Unknown action mode: {cfg.action_manager.action_mode}")
        ObsCls = cfg.observation_processor_cls
        RewCls = cfg.reward_calculator_cls
        self.obs_processor = ObsCls(cfg, cfg.observation_cfg)
        self.reward_calculator = RewCls(cfg, cfg.reward_cfg)
        self.obs_processor.bind_env(self)
        self.reward_calculator.bind_env(self)
        self.task_generator = CrossTaskGenerator(cfg, self)

    def _init_metrics(self):
        # 初始化 Metrics 管理器并注册模块
        use_skrl_metrics = getattr(self.cfg, 'use_skrl_metrics', True)
        self.metrics = MetricsManager(num_envs=self.num_envs, device=self.device, use_skrl_metrics=use_skrl_metrics)
        self.metrics.bind_env(self)
        self.metrics.register(FlagsModule())
        self.metrics.register(CrossTrackModule())
        self.metrics.register(AccelerationModule())
        if self.traffic_sim is not None:
            self.metrics.register(NearCollisionModule())

    def _setup_terrain(self):
        if hasattr(self.cfg, 'urban_terrain'):
            from isaac_lab_envs.direct.components.map import convert_urban_terrain_cfg_to_terrain_importer_cfg
            self.cfg.terrain = convert_urban_terrain_cfg_to_terrain_importer_cfg(self.cfg.urban_terrain, self.cfg.seed)
        # 3. 设置地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors."""

        if self.cfg.terrain.terrain_generator is not None:
            self.cfg.terrain.terrain_generator.seed = self.cfg.seed
        
        # 1. 首先创建无人机系统（但还不生成）
        self.drone, self.controller = MultirotorBase.make(self.cfg.drone_model, self.cfg.controller, self.device)
        
        # 2. 在模板环境中生成一个无人机
        translations = [(0.0, 0.0, 2.0)]
        drone_prims = self.drone.spawn(translations)

        self.traffic_sim = None
        if self.cfg.traffic_sim is not None:
            self.traffic_sim = TrafficSimulator(self.cfg.traffic_sim, self.device)
            self.traffic_sim.create_traffic_prim()


        
        self._setup_terrain()
        # 4. 设置传感器（在克隆之前）
        self._setup_lidar()
        
        # 5. 设置光照
        self._setup_lights()

        # 6. 克隆环境并处理碰撞

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])


    def _post_init_setup(self):
        """在场景设置完成后进行无人机系统初始化"""

        # 设置无人机的shape以匹配环境数量
        self.drone.shape = (self.num_envs, 1)
        self.drone.initialize()

        if self.traffic_sim is not None:
            self.traffic_sim.initialize()
        
        # 设置随机化
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self._lidar._initialize_impl()
        
        # 获取初始状态
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        # 设置环境位置偏移
        self.drone._envs_positions = self._terrain.env_origins.unsqueeze(1)
        

        # 设置处理器的设备
        self.obs_processor.device = self.device
        self.reward_calculator.device = self.device

        # 初始化状态管理对象
        self.state = EnvState(device=self.device, num_envs=self.num_envs)
        self.state.bind_env(self)
        self.state.initialize_basic_tensors()
        self.state.collision.safety_radius = self.cfg.safety_radius



        self.global_path_planner = None

        if self.cfg.use_global_path:
            self.state.initialize_navigation_waypoints(max_wps=self.cfg.global_path_planner_cfg.max_waypoints)
 

        self.map_manager = None
        self.global_path_planner = None
        if self.cfg.map_cfg is not None and self.cfg.terrain.terrain_generator is not None:
            self.map_manager = MapManager(self.cfg.map_cfg, self)
            self.map_manager.create_global_point_cloud()
            self.map_manager.create_occupancy_grid()
            if self.cfg.use_global_path:
                self.global_path_planner = GlobalPathPlanner(self.cfg.global_path_planner_cfg)
                self.global_path_planner.update_grid_map(self.state.map.extended_occupancy_grid, 
                                                    self.state.map.grid_size, 
                                                    bounds=self.state.map.grid_bounds)
            if self.traffic_sim is not None:
                self.map_manager._create_occupancy_grid_for_traffic()
                
        if self.traffic_sim is not None:
            self.state.init_traffic_namespace(self.cfg.predict_steps, self.cfg.pred_timestep)
            self.traffic_sim.reset()
    def _setup_lidar(self):
        lidar_vfov_rad = (
            max(-89.0, self.cfg.lidar_vfov[0]) * math.pi / 180.0,
            min(89.0, self.cfg.lidar_vfov[1]) * math.pi / 180.0
        )
        vertical_ray_angles = torch.linspace(lidar_vfov_rad[0], lidar_vfov_rad[1], self.cfg.lidar_resolution[1])
        mesh_paths = ["/World/ground"]
        ray_caster_cfg = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/{self.cfg.drone_model.capitalize()}_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=self.cfg.lidar_attach_yaw_only,
            pattern_cfg=patterns.BpearlPatternCfg(
                vertical_ray_angles=vertical_ray_angles
            ),
            debug_vis=False,
            mesh_prim_paths=mesh_paths,
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
        dt_for_evtol = self.cfg.sim.dt * self.cfg.decimation
        if self.traffic_sim is not None:
            self.traffic_sim._pre_physics_step(dt=dt_for_evtol)

        self.action_manager.process_actions(actions)
    

    def _apply_action(self):
        if self.traffic_sim is not None:
            self.traffic_sim._apply_actions()
        self.action_manager.apply_action()
    
    def _post_physics_step(self, env_ids: torch.Tensor = None):
        """
        Update sensors after physics step.
        direct rl env中没有这个函数
        """
        # 如果放在apply action之后，那更新太频繁了
        # 暂时放在get dones之前和reset_idx之后
        # 应该先更新traffic, 这样方便ego drone的collision detection
        if self.traffic_sim is not None:
            self.traffic_sim._post_physics_step()
            self.traffic_sim.update_traffic_for_env(self.state)
            self.state.traffic.traffic_future_traj = self.traffic_sim.predict_future_positions_by_manager(self.cfg.predict_steps, self.cfg.pred_timestep)
        
        

        # 更新状态对象
        drone_state = self.drone.get_state(env_frame=False)  # [num_envs, 1, 25]
        self.state.update_ego_drone_state(drone_state)
        self.state.update_navigation_distances()
        self.state.update_reached_target_mask(self.cfg.arrival_threshold)
        if self.cfg.use_global_path:
            self.state.update_navigation_state_vectorized(self.cfg.global_path_planner_cfg.lookahead_distance, env_ids)
        self._lidar.update(self.step_dt)
        self.state.update_lidar_scan(self._lidar, self.cfg.lidar_range, self.cfg.lidar_resolution)
        collision_mask = self._detect_collisions()
        self.state.collision.collision_mask = collision_mask.clone()
        # metrics modules are triggered only on_done

    def _get_observations(self) -> dict:
        """计算基于字典格式的导航观测。"""
        # 使用观测处理器计算观测
        observations = self.obs_processor.process_observation(self.state)
        # write to mdp state for cross-component access
        self.state.set_observations(observations)
        # 逐个key检查是否存在nan值, 这种nan会引起RL的崩溃，是所以必须停止训练
        for key, value in observations["policy"].items():
            if torch.isnan(value).any():
                print(f"UamEnv: {key} is nan: {value}")
                raise ValueError(f"UamEnv: {key} is nan: {value}")
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算基于2D导航的奖励。"""
        # 使用奖励计算器计算奖励
        reward = self.reward_calculator.compute_reward(self.state)
        # write to mdp state
        self.state.set_reward(reward)
        if torch.isnan(reward).any():
            print(f"RewardManager: reward is nan: {reward}")
            raise ValueError(f"RewardManager: reward is nan: {reward}")
        
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
        if hasnan.any():
            # 这种nan可以修复，这里仅打印提醒，不终止训练
            print(f"UamEnv: hasnan: {hasnan}")
        
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
        
        # 重置机器人到初始位置（完全按照原始实现）
        if self.cfg.use_global_path:
            start, goal, waypoints, waypoints_length = self.task_generator.generate_task(len(env_ids), flight_height=self.cfg.flight_height)
            self.state.navigation.waypoints[env_ids] = waypoints
            self.state.navigation.waypoint_lengths[env_ids] = waypoints_length
            self.state.navigation.current_waypoint_indices[env_ids] = 0
        else:
            # start, goal = self.task_generator.generate_task(len(env_ids), flight_height=self.cfg.flight_height)
            # 输出的waypoints只有两个点，起点和终点
            start, goal, waypoints, waypoints_length = self.task_generator.generate_task(len(env_ids), flight_height=self.cfg.flight_height)
            self.state.navigation.waypoints[env_ids] = waypoints
            self.state.navigation.waypoint_lengths[env_ids] = waypoints_length
            self.state.navigation.current_waypoint_indices[env_ids] = 0
        
        # 随机初始姿态（使用原始分布）
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        # 如果使用加速度动作空间，则将初始yaw对准第二个waypoint
        if self.cfg.action_manager.action_mode == "direction_acceleration":
            # start: [M, 1, 3], waypoints: [M, W, 3]
            rpy = torch.zeros_like(rpy)
            start_xy = start[:, 0, :2]
            wp1_xy = waypoints[:, 1, :2]
            dir_xy = wp1_xy - start_xy
            yaw = torch.atan2(dir_xy[:, 1], dir_xy[:, 0])  # radians
            rpy[..., 2] = yaw.unsqueeze(-1)
        rot = euler_to_quaternion(rpy)
        # rot = torch.zeros_like(rot)
        # rot[..., 0] = 1.0
        
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
        self.reward_calculator.reset(self.state, env_ids)
        
        super()._reset_idx(env_ids)
        self.metrics.on_reset(env_ids)
    
    
    def _detect_collisions(self) -> torch.Tensor:
        """
        Nav env中没有障碍物, 为了和子类对齐，返回全zero的collision_mask
        """
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # lidar collision
        if self._lidar is not None:
            scan = self.state.perception.lidar_scan
            scan = scan.reshape(self.num_envs, -1)
            distances = self.cfg.lidar_range - scan
            lidar_collision_mask = (distances <= self.cfg.safety_radius).any(dim=1)
            collision_mask = collision_mask | lidar_collision_mask
        # map collision
        if self.map_manager is not None:
            positions = self.state.ego_drone.positions
            if positions is not None:
                positions_3d = positions[:, 0, :]
                safe_mask = self.map_manager.are_positions_safe(positions_3d)
                occ_collision = ~safe_mask
                collision_mask = collision_mask | occ_collision
        if self.traffic_sim is not None:
            ego_pos = self.state.ego_drone.positions
            ego_safety_radius = torch.ones(self.num_envs, device=self.device) * self.cfg.safety_radius

            traffic_collision_mask = self.traffic_sim.check_collision(
                ego_pos.squeeze(1),
                ego_safety_radius
            )
            collision_mask = collision_mask | traffic_collision_mask
        return collision_mask
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if self.visualization_manager is not None:
            self.visualization_manager.setup_debug_vis(debug_vis)


    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if self.visualization_manager is not None:
            self.visualization_manager.debug_vis_callback(event)

    
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
