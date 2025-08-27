# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Observation managers for the Forest environment."""

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def target_position_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get relative target position in robot body frame."""
    robot: Articulation = env.scene["robot"]
    target_pos_w = env.command_manager.get_command("target_position")
    
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        target_pos_w
    )
    
    # normalize by distance
    distance = torch.norm(target_pos_b, dim=-1, keepdim=True)
    target_pos_b_normalized = target_pos_b / distance.clamp_min(1e-6)
    
    return target_pos_b_normalized


def root_lin_vel_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get root linear velocity in body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def root_ang_vel_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get root angular velocity in body frame.""" 
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get projected gravity vector in body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def lidar_scan_features(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get statistical features from lidar scan."""
    lidar: RayCaster = env.scene[sensor_cfg.name]
    
    # get lidar distances
    lidar_distances = lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1)
    lidar_distances = lidar_distances.norm(dim=-1).clamp_max(lidar.cfg.max_distance)
    
    # convert to occupancy representation
    lidar_scan = lidar.cfg.max_distance - lidar_distances
    lidar_scan_flat = lidar_scan.reshape(env.num_envs, -1)
    
    # compute statistical features
    features = torch.stack([
        lidar_scan_flat.mean(dim=-1),      # average distance
        lidar_scan_flat.min(dim=-1)[0],    # closest obstacle
        lidar_scan_flat.max(dim=-1)[0],    # furthest obstacle
        lidar_scan_flat.std(dim=-1),       # distance variance
    ], dim=-1)
    
    return features


def robot_orientation_features(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get robot orientation features (partial rotation matrix)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get quaternion and convert to rotation matrix
    quat = asset.data.root_state_w[:, 3:7]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # compute first two rows of rotation matrix (6 elements)
    rot_matrix = torch.zeros(env.num_envs, 2, 3, device=env.device)
    rot_matrix[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rot_matrix[:, 0, 1] = 2 * (x*y - w*z)
    rot_matrix[:, 0, 2] = 2 * (x*z + w*y)
    rot_matrix[:, 1, 0] = 2 * (x*y + w*z)
    rot_matrix[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rot_matrix[:, 1, 2] = 2 * (y*z - w*x)
    
    return rot_matrix.reshape(env.num_envs, 6) 