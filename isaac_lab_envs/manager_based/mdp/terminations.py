# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Termination managers for the Forest environment."""

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Termination due to episode timeout."""
    return env.episode_length_buf >= env.max_episode_length


def height_termination(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg, 
    min_height: float, 
    max_height: float
) -> torch.Tensor:
    """Termination due to height violation."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    return torch.logical_or(
        asset.data.root_pos_w[:, 2] < min_height,
        asset.data.root_pos_w[:, 2] > max_height
    )


def velocity_termination(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg, 
    max_velocity: float
) -> torch.Tensor:
    """Termination due to excessive velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    velocity_magnitude = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    return velocity_magnitude > max_velocity


def lidar_collision_termination(
    env: "ManagerBasedRLEnv", 
    sensor_cfg: SceneEntityCfg, 
    collision_threshold: float
) -> torch.Tensor:
    """Termination due to collision detected by lidar."""
    lidar: RayCaster = env.scene[sensor_cfg.name]
    
    # get lidar distances
    lidar_distances = lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1)
    lidar_distances = lidar_distances.norm(dim=-1).clamp_max(lidar.cfg.max_distance)
    
    # convert to occupancy representation
    lidar_scan = lidar.cfg.max_distance - lidar_distances
    
    # check if any ray detects an obstacle closer than threshold
    collision = (lidar_scan > (lidar.cfg.max_distance - collision_threshold)).any(dim=(1, 2))
    
    return collision 