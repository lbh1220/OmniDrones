# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Forest Navigation Environment using Direct RL Workflow

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

# 导入原始OmniDrones的robot系统
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


# 导入原始的TensorDict相关
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

##
# Pre-defined configs
##
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class ForestEnvWindow(BaseEnvWindow):
    """Window manager for the Forest environment."""

    def __init__(self, env: ForestEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class ForestEnvCfg(DirectRLEnvCfg):
    """Configuration for the Forest navigation environment."""
    
    # environment settings
    episode_length_s = 20.0
    decimation = 1  # env step every 1 sim steps
    num_actions = 2  # 只输出vx,vy
    num_observations = 23 + 6 + (36 * 4)  # drone_state + lidar_scan (firefly有6个旋翼)
    num_states = 0
    debug_vis = True

    ui_window_class_type = ForestEnvWindow
    viewer: ViewerCfg = field(default_factory=lambda: ViewerCfg(
        resolution=(960, 720),
        eye=(8, 0., 6.),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    ))
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 60Hz simulation
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
    
    # terrain with obstacles (完全按照原始配置)
    # 创建一个由 5x5 网格拼接而成的大型地形。
    # 网格中的每一个地块（8x8米）都是一个“障碍物场”，其中随机散布着40个高度和宽度各不相同的柱状障碍物
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=40,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(3.0, 4.0),
                    platform_width=1.5,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=8.0, replicate_physics=False)

    # 原始参数配置
    lidar_range: float = 4.0
    lidar_vfov: tuple[float, float] = (-10.0, 20.0)  # degrees
    lidar_resolution: tuple[int, int] = (36, 4)  # horizontal x vertical
    
    # drone配置 - 使用原始OmniDrones的配置系统
    drone_model: str = "firefly"  # 可以选择: firefly, crazyflie, hummingbird, iris等
    controller: Optional[str] = "PlanarSpeedController"  # 可以设置控制器，如 "lee_controller"
    
    # 奖励权重（完全按照原始配置）
    reward_effort_weight: float = 0.0  # 原始代码中被注释了
    
    # 随机化配置
    randomization: Dict = field(default_factory=dict)
    time_encoding: bool = True


class ForestEnv(DirectRLEnv):
    """Forest navigation environment for drones using Direct RL workflow."""
    
    cfg: ForestEnvCfg

    def __init__(self, cfg: ForestEnvCfg, render_mode: str | None = None, **kwargs):
        # 保存配置参数（在父类初始化之前）
        self.reward_effort_weight = cfg.reward_effort_weight
        self.time_encoding = cfg.time_encoding
        self.randomization = cfg.randomization
        self.has_payload = "payload" in self.randomization.keys()
        self.lidar_resolution = cfg.lidar_resolution
        
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        
        # 在父类初始化完成后进行无人机特定的初始化
        self._post_init_setup()
        
        # 初始位置和速度分布
        self.init_rpy_dist = torch.distributions.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        
        # 目标位置（完全按照原始配置）
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.0
        self.target_pos[:, 0, 1] = 24.0
        self.target_pos[:, 0, 2] = 2.0
        
        # 统计信息（原始格式）
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "safety": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.stats = stats_spec.zero()
        
        # debug可视化
        if self.sim.has_gui():
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
        else:
            self.debug_draw = None
        self.alpha = 0.8
        
        # debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

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
        # 使用原始无人机系统的apply_action方法
        # 应该在这里处理action，无论是做放缩，还是通过controller处理
        # rotor_commands = self.controller.compute(
        #     root_state=drone_state,  # shape [1, N, 3]
        #     target_vel_xy=target_vel_xy,  # shape [1, N, 2]
        #     target_height=target_height,  # shape [1, N, 1]
        #     target_yaw=target_yaws  # shape [1, N]
        #     )
        # 有两种思路，apply action的频率更高，按理说应该这里把控制量算出来，然后apply action实时更新drone state然后重新计算力和 力矩
        self.effort = self.drone.apply_action(actions.unsqueeze(1))  # add robot dimension

    def _apply_action(self):
        """Actions are applied in _pre_physics_step."""
        pass
    
    def _post_physics_step(self):
        """Update sensors after physics step."""
        self._lidar.update(self.step_dt)

    def _get_observations(self) -> dict:
        """Compute observations exactly like original implementation."""
        # 使用原始无人机系统获取状态
        self.drone_state = self.drone.get_state(env_frame=False)
        
        # 计算相对目标位置（保存为实例变量供奖励函数使用）
        self.rpos = self.target_pos - self.drone_state[..., :3]
        distance = self.rpos.norm(dim=-1, keepdim=True)
        rpos_clipped = self.rpos / distance.clamp(1e-6)
        
        # 计算激光雷达扫描（完全按照原始实现）
        self.lidar_scan = self.cfg.lidar_range - (
            (self._lidar.data.ray_hits_w - self._lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.cfg.lidar_range)
            .reshape(self.num_envs, 1, *self.cfg.lidar_resolution)
        )
        
        # 组装观测（完全按照原始格式）
        obs = {
            "state": torch.cat([rpos_clipped, self.drone_state[..., 3:]], dim=-1).squeeze(1),
            "lidar": self.lidar_scan
        }
        
        # # Debug visualization (完全按照原始实现)
        # if self._should_render(0):
        #     self.debug_draw.clear()
        #     x = self._lidar.data.pos_w[0]
        #     # 更新相机视角（跟随无人机）
        #     from omni.isaac.core.utils.viewports import set_camera_view
        #     set_camera_view(
        #         eye=x.cpu() + torch.as_tensor([2.0, 2.0, 2.0]),
        #         target=x.cpu() + torch.as_tensor([0.0, 0.0, 0.0])
        #     )
        #     v = (self._lidar.data.ray_hits_w[0] - x).reshape(*self.cfg.lidar_resolution, 3)
        #     self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
        #     self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards exactly like original implementation."""
        # 计算相对目标位置和速度方向
        distance = self.rpos.norm(dim=-1, keepdim=True)
        vel_direction = self.rpos / distance.clamp_min(1e-6)
        
        # 安全奖励（基于激光雷达）
        reward_safety = torch.log(self.cfg.lidar_range - self.lidar_scan).mean(dim=(2, 3))
        
        # 速度奖励（朝向目标）
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1).clip(max=2.0)
        
        # 姿态奖励（保持直立）
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)
        
        # 总奖励（完全按照原始公式）
        reward = reward_vel + reward_up + 1.0 + reward_safety * 0.2
        
        # 更新统计
        self.stats["safety"].add_(reward_safety.unsqueeze(-1))
        self.stats["return"] += reward.unsqueeze(-1)
        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions exactly like original implementation."""
        # 失控条件（完全按照原始实现）
        misbehave = (
            (self.drone.pos[..., 2] < 0.2) |  # 高度过低
            (self.drone.pos[..., 2] > 4.0) |  # 高度过高  
            (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5) |  # 速度过大
            (self.lidar_scan.view(self.num_envs, -1).max(dim=-1)[0] > (self.cfg.lidar_range - 0.3))  # 碰撞检测
        )
        
        # NaN检测
        hasnan = torch.isnan(self.drone_state).any(-1)
        
        terminated = misbehave | hasnan
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments exactly like original implementation."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 使用原始无人机系统的重置方法
        self.drone._reset_idx(env_ids, self.sim.is_playing())
        
        # 重置统计
        self.stats[env_ids] = 0.0
        
        # 重置机器人到初始位置（完全按照原始实现）
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = (env_ids.float() / self.num_envs - 0.5) * 32.0
        pos[:, 0, 1] = -24.0
        pos[:, 0, 2] = 2.0
        
        # 随机初始姿态（使用原始分布）
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        
        # 设置位置和姿态（使用原始方法）
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.target_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "target_pos_visualizer"):
            self.target_pos_visualizer.visualize(self.target_pos.squeeze(1))
    
    @property 
    def _should_render(self):
        """Check if should render for debug visualization."""
        return lambda substep: self.sim.has_gui() and substep == 0 