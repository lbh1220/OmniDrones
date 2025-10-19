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
from typing import List
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
from omni.isaac.core.materials import PhysicsMaterial
import omni.isaac.lab.utils.math as math_utils

from omni_drones.traffic.cfg.config import AreaBoundsCfg

# 导入原始OmniDrones的robot系统
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


# 导入原始的TensorDict相关
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

from isaac_lab_envs.direct.nav_env import NavEnvCfg, NavEnv
from isaac_lab_envs.direct.mdp.metrics import MetricsManager, FlagsModule, CrossTrackModule, AccelerationModule, NearCollisionModule
from omni_drones.traffic import TrafficSimulator, TrafficCfg, TrafficEvtolCfg, TrafficDroneCfg, AreaBoundsCfg
from omni_drones.traffic.cfg.config import OrcaCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

# Import ORCA policy for collision avoidance
try:
    from isaac_lab_envs.direct.policies.simple_policies import ORCAPolicy
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False
    print("Warning: ORCA policy not available. Install rvo2 for ORCA support.")

@configclass
class TrafficCurriculumCfg:
    """Configuration for the Traffic curriculum learning environment."""
    drones_num: int = 1
    evtol_num: int = 0
    evtol_radius: float = 10.0
    
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
class TrafficEnvCfg(NavEnvCfg):
    """Configuration for the Nav navigation environment."""
    
    traffic_sim: TrafficCfg = field(default_factory=lambda: TrafficCfg(
        num_drones=20,
        num_evtols=3,
        flight_height=20.0,
        area_bounds=AreaBoundsCfg(
            xmin=-80.0,
            xmax=80.0,
            ymin=-80.0,
            ymax=80.0
        )
    ))
    
    predict_steps: int = 5
    pred_timestep: float = 2.0
    observation_radius: float = 100.0
    observation_norm_scale: float = 10.0
    use_angle_distance_obs: bool = False


    # ORCA collision avoidance config
    orca: OrcaCfg = field(default_factory=lambda: OrcaCfg(enable=False))

    # reward config
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_action_penalty = -0.0
    rew_evtol_future_penalty = -0.0
    rew_drone_future_penalty = -0.0
    rew_time_penalty = 0.0
    rew_drones_threshold_factor = 2.0
    rew_drones_decay_factor = 0.667
    rew_evtols_threshold_factor = 2.0
    rew_evtols_decay_factor = 0.9
    ## drones in previous, 2.0, 0.667; evtols in previous, 1.5, 0.9
    rew_cross_track_coeff = 0.0

    rew_ttc_threshold = 10.0
    rew_ttc_alpha = 0.0
    rew_ttc_beta = 5.0
    rew_ttc_idle_penalty = 0.0
    rew_patience_coeff = 0.0
    

class TrafficEnvWithCurriculumCfg(TrafficEnvCfg):
    # curriculum learning
    curriculum_learning: bool = False
    curriculum_list: List[TrafficCurriculumCfg] = None

class TrafficEnv(NavEnv):
    """Nav navigation environment for drones using Direct RL workflow."""
    
    cfg: TrafficEnvCfg

    def __init__(self, cfg: TrafficEnvCfg, render_mode: str | None = None, **kwargs):
        # 保存配置参数（在父类初始化之前）
        self.traffic_sim = None
        
        # 初始化ORCA policy（如果启用）
        self.orca_policy = None
        if cfg.orca.enable and ORCA_AVAILABLE:
            # 直接使用配置中的ORCA设置
            self.orca_policy = ORCAPolicy(cfg.orca, cfg, name="TrafficORCA")
            print(f"ORCA collision avoidance enabled with safety_space={cfg.orca.safety_space}")
        elif cfg.orca.enable and not ORCA_AVAILABLE:
            print("Warning: ORCA requested but not available. Proceeding without ORCA.")
    
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        

        # 初始化traffic命名空间
        self.state.init_traffic_namespace(cfg.predict_steps, cfg.pred_timestep)
        self.traffic_sim.reset()
        # traffic env 不是reset idx，不是每次step都reset

    def _init_mdp_components(self, cfg: TrafficEnvCfg):
        """初始化模块化组件"""
        if cfg.use_global_path:
            from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessorWithPath
            from isaac_lab_envs.direct.mdp.rewards import TrafficRewardCalculatorWithPath
            self.obs_processor = TrafficObservationProcessorWithPath(cfg)
            self.reward_calculator = TrafficRewardCalculatorWithPath(cfg)

        else:
            from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessor
            from isaac_lab_envs.direct.mdp.rewards import TrafficRewardCalculator
            self.obs_processor = TrafficObservationProcessor(cfg)
            self.reward_calculator = TrafficRewardCalculator(cfg)
        # 初始化 Metrics 管理器并注册 Traffic 相关模块
    
    def _init_metrics(self):
        self.metrics = MetricsManager(num_envs=self.num_envs, device=self.device)
        self.metrics.bind_env(self)
        self.metrics.register(FlagsModule())
        self.metrics.register(CrossTrackModule())
        self.metrics.register(AccelerationModule())
        self.metrics.register(NearCollisionModule())
        
    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors."""

        self.traffic_sim = TrafficSimulator(self.cfg.traffic_sim, self.device)
        self.traffic_sim.create_traffic_prim()

        super()._setup_scene()

    def _post_init_setup(self):
        """在场景设置完成后进行无人机系统初始化"""

        self.traffic_sim.initialize()
        print(f"TrafficEnv initialized with {self.traffic_sim.config.num_drones} drones and {self.traffic_sim.config.num_evtols} evtols")
        
        super()._post_init_setup()

    def _pre_physics_step(self, actions: torch.Tensor):
        dt_for_evtol = self.cfg.sim.dt * self.cfg.decimation
        self.traffic_sim._pre_physics_step(dt=dt_for_evtol)
        
        # 调用父类的_pre_physics_step，这会生成velocity_commands
        super()._pre_physics_step(actions)
        
        # 如果启用ORCA，则对velocity_commands进行修正
        if self.orca_policy is not None:
            corrected_velocity = self.orca_policy.compute_corrected_velocity(self.state)
            self.state.navigation.velocity_commands = corrected_velocity
            self.command_vel_xy[:, :, :2] = corrected_velocity[:, :, :2]

    def _apply_action(self):
        """Actions are applied in _pre_physics_step."""
        self.traffic_sim._apply_actions()
        super()._apply_action()
    
    def _post_physics_step(self, env_ids: torch.Tensor = None):
        """Update sensors after physics step."""
        self.traffic_sim._post_physics_step()
        self._update_traffic_obs_processor()
        # 计算碰撞和到达目标的mask
        collision_mask = self._detect_collisions()
        self.state.collision.collision_mask = collision_mask.clone()
        # reached_target_mask will be updated in super()._post_physics_step

        super()._post_physics_step(env_ids)

    def _get_observations(self) -> dict:
        """计算基于字典格式的导航观测。"""
        # 使用观测处理器计算观测（已包含预计算的轨迹）
        observations = self.obs_processor.process_observation(self.state)
        self.state.set_observations(observations)
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算基于Traffic环境的复杂奖励。"""
        # 计算碰撞和到达目标的mask，并更新到state中
        # 使用奖励计算器计算奖励
        reward = self.reward_calculator.compute_reward(self.state)
        self.state.set_reward(reward)

               
        return reward
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""

        if debug_vis:
            if not hasattr(self, "traffic_visualizer"):
                # 2. 直接实例化 VisualizationMarkersCfg，而不是从预设copy()
                traffic_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/traffic_pos",
                    markers={
                        # 3. 在 markers 字典中，手动创建一个球体配置
                        #    这里的 "sphere" 是我们自己起的名字，可以任意
                        "sphere": sim_utils.SphereCfg(
                            radius=1.0,  # 设置球体的半径
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0),  # 设置颜色为红色
                                opacity=0.5
                            ),
                        )
                    },
                )
                # 创建可视化实例
                self.traffic_visualizer = VisualizationMarkers(traffic_marker_cfg)
            
            # 设置可见性
            self.traffic_visualizer.set_visibility(True)
            if not hasattr(self, "drone_pos_visualizer"):
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
            self.drone_pos_visualizer.set_visibility(True)



        else:
            if hasattr(self, "traffic_visualizer"):
                self.traffic_visualizer.set_visibility(False)
            if hasattr(self, "drone_visualizer"):
                self.drone_pos_visualizer.set_visibility(False)
        # 调用父类的方法
        super()._set_debug_vis_impl(debug_vis)
    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "traffic_visualizer") and self.traffic_visualizer.is_visible():
            # 1. 获取位置张量
            positions = self.state.traffic.traffic_positions

            # 2. 获取 agent 的数量
            num_agents = positions.shape[0]

            # 3. 创建 scales 张量。
            #    我们使用 .expand() 方法，这比 .repeat() 更高效，因为它不会复制数据。
            #    同时，确保新的张量与 positions 在同一个设备上（CPU或GPU）。
            scale_shape = torch.tensor([1.0, 1.0, 0.5], device=positions.device, dtype=positions.dtype)
            scales = scale_shape.expand(num_agents, -1) # -1 表示保持维度大小不变 (num_agents, 3)
            scales = scales * self.state.traffic.traffic_safety_radius.unsqueeze(1)

            # 4. 调用 visualize 方法，同时传入 translations 和 scales
            self.traffic_visualizer.visualize(translations=positions, scales=scales)
        super()._debug_vis_callback(event)

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
    
    def _detect_collisions(self) -> torch.Tensor:
        """检测与traffic aircraft的碰撞"""
        
        ego_pos = self.state.ego_drone.positions
        ego_safety_radius = torch.ones(self.num_envs, device=self.device) * self.cfg.safety_radius

        collision_mask = self.traffic_sim.check_collision(
            ego_pos.squeeze(1),
            ego_safety_radius
        )
        
        return collision_mask

    def _update_traffic_obs_processor(self):
        # 预计算traffic轨迹预测，供观测和奖励计算使用
        traffic_positions = self.traffic_sim.get_aircraft_positions()  # [total_traffic, 3]
        traffic_velocities = self.traffic_sim.get_aircraft_velocities()  # [total_traffic, 3]
        traffic_types = self.traffic_sim.get_aircraft_types()  # [total_traffic] tensor
        traffic_safety_radius = self.traffic_sim.get_aircraft_safety_radius()  # [total_traffic] tensor
        
        # 更新state中的traffic数据
        self.state.traffic.traffic_positions = traffic_positions
        self.state.traffic.traffic_velocities = traffic_velocities
        self.state.traffic.traffic_types = traffic_types
        self.state.traffic.traffic_safety_radius = traffic_safety_radius
        
        # 同时更新观测处理器的缓存（保持兼容性）
        self.obs_processor.predict_traffic_trajectory(
            traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius
        )
        self.state.traffic.traffic_future_traj = self.obs_processor.traffic_future_traj.clone()

        # 不再在state中维护eVTOL专用视图，使用时按类型筛选



class TrafficEnvWithCurriculum(TrafficEnv):
    """Traffic environment with curriculum learning"""

    def __init__(self, cfg: TrafficEnvCfg, render_mode: str | None = None, **kwargs):
        self.course_num = len(cfg.curriculum_list)
        self.current_course = 0

        self.active_drones_num = cfg.curriculum_list[0].drones_num
        self.active_evtols_num = cfg.curriculum_list[0].evtol_num

        super().__init__(cfg, render_mode, **kwargs)



    def set_curriculum(self, course_idx: int):
        """Switch to next course"""
        self.current_course = course_idx
        if self.current_course >= self.course_num:
            self.current_course = self.course_num - 1
        self.active_drones_num = self.cfg.curriculum_list[self.current_course].drones_num
        self.active_evtols_num = self.cfg.curriculum_list[self.current_course].evtol_num
        # previous_evtol_radius = self.traffic_sim.evtol_manager.state.safety_radius
        # new_evtol_radius = torch.ones(previous_evtol_radius.shape, device=self.device) * self.cfg.curriculum_list[self.current_course].evtol_radius
        # self.traffic_sim.evtol_manager.state.safety_radius = new_evtol_radius
        # self.traffic_sim.evtol_manager.random_attributes(self.cfg.traffic_sim.evtol.random_speed, self.cfg.traffic_sim.evtol.random_safety_radius)
        self.traffic_sim.evtol_manager.config.evtol.safety_radius = self.cfg.curriculum_list[self.current_course].evtol_radius
        
    def _detect_collisions(self) -> torch.Tensor:
        """检测与traffic aircraft的碰撞"""
        
        ego_pos = self.state.ego_drone.positions
        ego_safety_radius = torch.ones(self.num_envs, device=self.device) * self.cfg.safety_radius

        collision_mask = self.traffic_sim.check_collision(
            ego_pos.squeeze(1),
            ego_safety_radius,
            activate_drones_num=self.active_drones_num,
            activate_evtols_num=self.active_evtols_num
        )
        
        return collision_mask

    def _update_traffic_obs_processor(self):
        # 预计算traffic轨迹预测，供观测和奖励计算使用
        traffic_positions = self.traffic_sim.get_aircraft_positions(activate_drones_num=self.active_drones_num, 
                                                                    activate_evtols_num=self.active_evtols_num)  # [total_traffic, 3]
        traffic_velocities = self.traffic_sim.get_aircraft_velocities(activate_drones_num=self.active_drones_num, 
                                                                    activate_evtols_num=self.active_evtols_num)  # [total_traffic, 3]
        traffic_types = self.traffic_sim.get_aircraft_types(activate_drones_num=self.active_drones_num, 
                                                            activate_evtols_num=self.active_evtols_num)  # [total_traffic] tensor
        traffic_safety_radius = self.traffic_sim.get_aircraft_safety_radius(activate_drones_num=self.active_drones_num, 
                                                            activate_evtols_num=self.active_evtols_num)  # [total_traffic] tensor
        
        # 更新state中的traffic数据
        self.state.traffic.traffic_positions = traffic_positions
        self.state.traffic.traffic_velocities = traffic_velocities
        self.state.traffic.traffic_types = traffic_types
        self.state.traffic.traffic_safety_radius = traffic_safety_radius
        
        # 同时更新观测处理器的缓存（保持兼容性）
        self.obs_processor.predict_traffic_trajectory(
            traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius
        )
        self.state.traffic.traffic_future_traj = self.obs_processor.traffic_future_traj.clone()

