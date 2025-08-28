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


class TrafficEnv(NavEnv):
    """Nav navigation environment for drones using Direct RL workflow."""
    
    cfg: TrafficEnvCfg

    def __init__(self, cfg: TrafficEnvCfg, render_mode: str | None = None, **kwargs):
        # 保存配置参数（在父类初始化之前）
        # 
        
        self.traffic_sim = None
        # 父类初始化 - 这会调用 _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)
        
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
