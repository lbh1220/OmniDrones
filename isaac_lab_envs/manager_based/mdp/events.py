# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Event managers for the Forest environment."""

import math
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def randomize_robot_mass(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
) -> None:
    """Randomize robot mass."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get current masses
    masses = asset.root_physx_view.get_masses().clone()
    
    # generate random multipliers
    multipliers = torch.rand(len(env_ids), device=env.device)
    multipliers = (
        multipliers * (mass_distribution_params[1] - mass_distribution_params[0]) 
        + mass_distribution_params[0]
    )
    
    # apply randomization
    masses[env_ids] *= multipliers.unsqueeze(-1)
    asset.root_physx_view.set_masses(masses, env_ids)


def reset_robot_to_initial_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_noise_std: float,
    orientation_noise_std: float,
) -> None:
    """Reset robot to initial pose with randomization."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get default root state
    root_state = asset.data.default_root_state[env_ids].clone()
    
    # randomize initial position
    position_noise = torch.randn(len(env_ids), 3, device=env.device) * position_noise_std
    
    # set initial positions (spread along X, start at negative Y)
    initial_pos = torch.zeros_like(root_state[:, :3])
    initial_pos[:, 0] = (env_ids.float() / env.num_envs - 0.5) * 32.0  # spread along X
    initial_pos[:, 1] = -24.0  # start at negative Y
    initial_pos[:, 2] = 2.0    # start at 2m height
    
    root_state[:, :3] = initial_pos + position_noise
    
    # randomize initial orientation (small angles)
    rpy_noise = torch.randn(len(env_ids), 3, device=env.device) * orientation_noise_std
    rpy_noise[:, 2] *= 10  # allow larger yaw variation
    
    # convert to quaternion
    cos_half = torch.cos(rpy_noise / 2)
    sin_half = torch.sin(rpy_noise / 2)
    
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0] = cos_half[:, 0] * cos_half[:, 1] * cos_half[:, 2] + sin_half[:, 0] * sin_half[:, 1] * sin_half[:, 2]  # w
    quat[:, 1] = sin_half[:, 0] * cos_half[:, 1] * cos_half[:, 2] - cos_half[:, 0] * sin_half[:, 1] * sin_half[:, 2]  # x
    quat[:, 2] = cos_half[:, 0] * sin_half[:, 1] * cos_half[:, 2] + sin_half[:, 0] * cos_half[:, 1] * sin_half[:, 2]  # y
    quat[:, 3] = cos_half[:, 0] * cos_half[:, 1] * sin_half[:, 2] - sin_half[:, 0] * sin_half[:, 1] * cos_half[:, 2]  # z
    
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0  # zero initial velocities
    
    # set the root state
    asset.write_root_state_to_sim(root_state, env_ids)


def reset_target_position(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    command_name: str,
) -> None:
    """Reset target position command."""
    env.command_manager.get_term(command_name).reset(env_ids) 