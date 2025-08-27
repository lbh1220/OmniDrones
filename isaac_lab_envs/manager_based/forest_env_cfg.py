# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Forest Navigation Environment Configuration using Manager-based Workflow

这个配置文件定义了森林导航环境的Manager-based实现，包括：
- Scene配置（机器人、地形、传感器）
- Action管理器配置
- Observation管理器配置  
- Reward管理器配置
- Termination管理器配置
- Event管理器配置
"""

import math
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.utils import configclass

# import MDP functions - these would need to be implemented
from . import mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort:skip


##
# Scene definition
##

@configclass
class ForestSceneCfg(InteractiveSceneCfg):
    """Configuration for the Forest navigation scene."""

    # ground plane with obstacles
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
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

    # robot (drone)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lidar sensor
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.BpearlPatternCfg(
            vertical_ray_angles=torch.linspace(-10 * math.pi / 180, 20 * math.pi / 180, 4)
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=4.0,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/SkyLight", 
        spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    # target position command
    target_position = mdp.TargetPositionCommandCfg()


@configclass 
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # drone thrust and moments
    drone_action = mdp.DroneActionCfg(
        asset_name="robot", 
        thrust_scale=1.9,  # thrust to weight ratio
        moment_scale=0.01,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # drone state (relative target, velocities, orientation)
        target_position_rel = ObsTerm(func=mdp.target_position_rel)
        robot_lin_vel = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_ang_vel = ObsTerm(func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg("robot")})
        gravity_vector = ObsTerm(func=mdp.projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # lidar observations
        lidar_scan = ObsTerm(func=mdp.lidar_scan_features, params={"sensor_cfg": SceneEntityCfg("lidar")})
        
        # orientation features (partial rotation matrix)
        orientation_features = ObsTerm(func=mdp.robot_orientation_features, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # survival reward
    survival = RewTerm(func=mdp.survival_reward, weight=1.0)
    
    # velocity penalties
    lin_vel_penalty = RewTerm(
        func=mdp.lin_vel_l2_penalty, 
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    ang_vel_penalty = RewTerm(
        func=mdp.ang_vel_l2_penalty,
        weight=-0.01, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # safety reward from lidar
    safety_reward = RewTerm(
        func=mdp.lidar_safety_reward,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("lidar")}
    )
    
    # reward for moving towards target
    vel_towards_target = RewTerm(
        func=mdp.velocity_towards_target_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # upright reward (encourage staying upright)
    upright_reward = RewTerm(
        func=mdp.upright_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # height constraints
    height_violation = DoneTerm(
        func=mdp.height_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.2, "max_height": 4.0}
    )
    
    # velocity constraint
    velocity_violation = DoneTerm(
        func=mdp.velocity_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 2.5}
    )
    
    # collision detection via lidar
    collision = DoneTerm(
        func=mdp.lidar_collision_termination,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "collision_threshold": 0.3}
    )

    # episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup - randomize robot mass
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_robot_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),  # multiplier range
        },
    )

    # on reset - reset robot state with randomization
    reset_robot_position = EventTerm(
        func=mdp.reset_robot_to_initial_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_noise_std": 0.5,
            "orientation_noise_std": 0.2,
        },
    )
    
    # on reset - generate new target position
    reset_target_position = EventTerm(
        func=mdp.reset_target_position,
        mode="reset",
        params={
            "command_name": "target_position",
        },
    )


##
# Environment configuration
##

@configclass
class ForestManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Forest navigation environment using Manager-based workflow."""

    # Scene settings
    scene = ForestSceneCfg(num_envs=4096, env_spacing=8.0)
    
    # MDP settings
    commands = CommandsCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [5.0, 5.0, 8.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        
        # simulation settings
        self.sim.dt = 1 / 200  # sim step every 5ms: 200Hz
        self.sim.disable_contact_processing = True
        
        # episode settings
        self.episode_length_s = 20.0 