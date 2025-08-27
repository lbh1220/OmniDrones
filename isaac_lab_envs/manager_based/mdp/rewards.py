# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Reward managers for the Forest environment."""

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def survival_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def lin_vel_l2_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 penalty on linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)


def ang_vel_l2_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 penalty on angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def lidar_safety_reward(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Safety reward based on lidar distances."""
    lidar: RayCaster = env.scene[sensor_cfg.name]
    
    # get lidar distances
    lidar_distances = lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1)
    lidar_distances = lidar_distances.norm(dim=-1).clamp_max(lidar.cfg.max_distance)
    
    # convert to occupancy representation
    lidar_scan = lidar.cfg.max_distance - lidar_distances
    
    # safety reward: logarithm of distances
    safety_reward = torch.log(lidar.cfg.max_distance - lidar_scan).mean(dim=(1, 2))
    
    return safety_reward


def velocity_towards_target_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for moving towards the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = env.command_manager.get_command("target_position")
    
    # target direction
    target_direction = target_pos_w - asset.data.root_pos_w
    target_distance = torch.norm(target_direction, dim=-1, keepdim=True)
    target_direction_normalized = target_direction / target_distance.clamp_min(1e-6)
    
    # velocity component towards target
    vel_towards_target = (asset.data.root_lin_vel_w * target_direction_normalized).sum(dim=-1)
    
    return vel_towards_target.clamp(max=2.0)


def upright_reward(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for staying upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # gravity direction in body frame (when upright, z component should be -1)
    up_vector = asset.data.projected_gravity_b
    upright_reward = torch.square((up_vector[:, 2] + 1.0) / 2.0)
    
    return upright_reward 