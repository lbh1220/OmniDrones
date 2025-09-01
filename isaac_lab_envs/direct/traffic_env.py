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
from omni_drones.traffic import TrafficSimulator, TrafficCfg, OrcaCfg, TrafficEvtolCfg, TrafficDroneCfg, AreaBoundsCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class TrafficCurriculumCfg:
    """Configuration for the Traffic curriculum learning environment."""
    drones_num: int = 1
    evtol_num: int = 0
    def __init__(self, drones_num: int = 1, evtol_num: int = 0):
        self.drones_num = drones_num
        self.evtol_num = evtol_num

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
            xmin=-50.0,
            xmax=50.0,
            ymin=-50.0,
            ymax=50.0
        )
    ))
    
    predict_steps: int = 5
    pred_timestep: float = 2.0
    observation_radius: float = 100.0
    observation_norm_scale: float = 10.0

    # reward config
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_evtol_future_penalty = -0.8
    rew_drone_future_penalty = -1.0
    rew_time_penalty = 0.0
    rew_drones_threshold_factor = 1.5
    rew_drones_decay_factor = 0.667
    rew_evtols_threshold_factor = 1.5
    rew_evtols_decay_factor = 0.9

    # curriculum learning
    curriculum_learning: bool = False
    curriculum_list: List[TrafficCurriculumCfg] = field(default_factory=lambda: [TrafficCurriculumCfg()])

class TrafficEnv(NavEnv):
    """Nav navigation environment for drones using Direct RL workflow."""
    
    cfg: TrafficEnvCfg

    def __init__(self, cfg: TrafficEnvCfg, render_mode: str | None = None, **kwargs):
        # 保存配置参数（在父类初始化之前）
        self.traffic_sim = None
        
        # 初始化观测和奖励处理器
        self.obs_processor = None
        self.reward_calculator = None
        from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessor
        from isaac_lab_envs.direct.mdp.rewards import TrafficRewardCalculator
        # 初始化处理器（在父类初始化后，这样可以访问device）
        self.obs_processor = TrafficObservationProcessor(cfg)
        self.reward_calculator = TrafficRewardCalculator(cfg)
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        
        self.obs_processor.device = self.device
        self.reward_calculator.device = self.device
        self.traffic_sim.reset()
        # traffic env 不是reset idx，不是每次step都reset


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
        super()._pre_physics_step(actions)

    def _apply_action(self):
        """Actions are applied in _pre_physics_step."""
        self.traffic_sim._apply_actions()
        super()._apply_action()
    
    def _post_physics_step(self):
        """Update sensors after physics step."""
        self.traffic_sim._post_physics_step()
        super()._post_physics_step()

        self._update_traffic_obs_processor()
        
    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        """重置指定环境的状态"""
        # 重置奖励计算器的势能缓存

        super()._reset_idx(env_ids)
        if self.reward_calculator is not None:
            self.reward_calculator.reset_potential(self.drone_state, self.target_pos, env_ids)

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        super()._configure_gym_env_spaces()
        import gymnasium as gym
        import numpy as np

        policy_space_dict = self.obs_processor.generate_policy_obs_dict()
        # 2. 将内层字典包装成一个 gym.spaces.Dict
        policy_space = gym.spaces.Dict(policy_space_dict)

        # 3. 创建最外层的观测空间字典
        self.single_observation_space["policy"] = policy_space

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)

    def _get_observations(self) -> dict:
        """计算基于字典格式的导航观测。"""
        # 使用观测处理器计算观测（已包含预计算的轨迹）

        observations = self.obs_processor.process_observation(
            self.drone_state,     # [num_envs, 1, 13]
            self.target_pos,      # [num_envs, 1, 3]
        )
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算基于Traffic环境的复杂奖励。"""
        # 获取traffic aircraft状态（使用预计算的数据）
        traffic_positions = self.obs_processor.traffic_positions  # [total_traffic, 3]
        traffic_velocities = self.obs_processor.traffic_velocities  # [total_traffic, 3]
        traffic_types = self.obs_processor.traffic_types  # [total_traffic] tensor
        traffic_safety_radius = self.obs_processor.traffic_safety_radius  # [total_traffic] tensor
        traffic_future_traj = self.obs_processor.traffic_future_traj  # [total_traffic, predict_steps+1, 3] tensor
        # 计算碰撞和到达目标的mask
        collision_mask, reached_target_mask = self._compute_collision_and_target_masks()
        
        # 使用奖励计算器计算奖励
        reward = self.reward_calculator.compute_reward(
            self.drone_state,      # [num_envs, 1, 13]
            self.target_pos,       # [num_envs, 1, 3]
            traffic_positions,     # [total_traffic, 3]
            traffic_velocities,    # [total_traffic, 3]
            collision_mask,
            reached_target_mask,
            traffic_future_traj,
            traffic_safety_radius,
            traffic_types

        )
               
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算基于Traffic环境的终止条件，包括碰撞检测。"""
        self._post_physics_step()
        
        # 计算碰撞和到达目标的mask
        collision_mask, reached_target_mask = self._compute_collision_and_target_masks()
        # 更新统计信息
        self.extras["goal_reached"] = reached_target_mask
        self.extras["collision"] = collision_mask
        # 3. 高度异常条件（保持在合理高度范围内）
        robot_height = self.drone_state.squeeze(1)[:, 2]  # [num_envs]
        height_abnormal = (
            (robot_height < (self.cfg.flight_height - 3*self.cfg.safety_radius)) |
            (robot_height > (self.cfg.flight_height + 3*self.cfg.safety_radius))
        )
        
        # 4. NaN检测
        hasnan = torch.isnan(self.drone_state).any(dim=(1, 2))
        
        # 终止条件：到达目标、碰撞、高度异常或NaN
        terminated = reached_target_mask | collision_mask | height_abnormal | hasnan
        
        # 超时条件：由DirectRLEnv框架自动处理
        truncated = self.episode_length_buf >= self.max_episode_length 
        
        return terminated, truncated
    
    def _compute_collision_and_target_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算碰撞和到达目标的mask"""
        # 计算距离目标的距离（用于到达判断和奖励）
        robot_pos = self.drone_state[:, :, :2]  # [num_envs, 1, 2]
        goal_pos = self.target_pos[:, :, :2]    # [num_envs, 1, 2]
        relative_goal_pos = goal_pos - robot_pos # [num_envs, 1, 2]
        self.current_dist_to_target = torch.norm(relative_goal_pos.squeeze(1), dim=1)  # [num_envs]
        
        # 1. 到达目标检测
        reached_target_mask = self.current_dist_to_target <= self.cfg.arrival_threshold
        
        # 2. 碰撞检测
        collision_mask = self._detect_collisions()
        # if collision_mask.any():
        #     print(f"Collision detected at step")
        
        return collision_mask, reached_target_mask
    
    def _detect_collisions(self) -> torch.Tensor:
        """检测与traffic aircraft的碰撞"""
        
        ego_pos = self.drone_state[:, :, :3]
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
        self.obs_processor.predict_traffic_trajectory(
            traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius
        )


class TrafficEnvWithCurriculum(TrafficEnv):
    """Traffic environment with curriculum learning"""

    def __init__(self, cfg: TrafficEnvCfg, render_mode: str | None = None, **kwargs):
        self.course_num = len(cfg.curriculum_list)
        self.current_course = 0

        self.active_drones_num = cfg.curriculum_list[0].drones_num
        self.active_evtols_num = cfg.curriculum_list[0].evtol_num

        super().__init__(cfg, render_mode, **kwargs)



    def set_course(self, course_idx: int):
        """Switch to next course"""
        self.current_course = course_idx
        if self.current_course >= self.course_num:
            self.current_course = self.course_num - 1
        self.active_drones_num = self.cfg.curriculum_list[self.current_course].drones_num
        self.active_evtols_num = self.cfg.curriculum_list[self.current_course].evtol_num
        
    def _detect_collisions(self) -> torch.Tensor:
        """检测与traffic aircraft的碰撞"""
        
        ego_pos = self.drone_state[:, :, :3]
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
        self.obs_processor.predict_traffic_trajectory(
            traffic_positions, traffic_velocities, traffic_types, traffic_safety_radius
        )

