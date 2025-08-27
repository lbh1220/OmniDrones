# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Command managers for the Forest environment."""

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


@configclass
class TargetPositionCommandCfg(CommandTermCfg):
    """Configuration for target position command."""
    
    class_type: type = "TargetPositionCommand"
    
    # target generation parameters
    target_range_x: tuple[float, float] = (-16.0, 16.0)  # X range for target
    target_range_y: tuple[float, float] = (20.0, 28.0)   # Y range for target (ahead of drone)
    target_range_z: tuple[float, float] = (1.5, 2.5)     # Z range for target


class TargetPositionCommand(CommandTerm):
    """Command term for generating target positions."""
    
    cfg: TargetPositionCommandCfg

    def __init__(self, cfg: TargetPositionCommandCfg, env: "ManagerBasedRLEnv"):
        """Initialize the command term.
        
        Args:
            cfg: The configuration instance.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # target position buffer
        self.target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        
        # generate initial targets
        self._generate_targets()

    def _generate_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """Generate new target positions.
        
        Args:
            env_ids: The environment indices to generate targets for. If None, generates for all.
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        # generate random target positions within specified ranges
        self.target_pos_w[env_ids, 0] = torch.rand(
            len(env_ids), device=self._env.device
        ) * (self.cfg.target_range_x[1] - self.cfg.target_range_x[0]) + self.cfg.target_range_x[0]
        
        self.target_pos_w[env_ids, 1] = torch.rand(
            len(env_ids), device=self._env.device
        ) * (self.cfg.target_range_y[1] - self.cfg.target_range_y[0]) + self.cfg.target_range_y[0]
        
        self.target_pos_w[env_ids, 2] = torch.rand(
            len(env_ids), device=self._env.device
        ) * (self.cfg.target_range_z[1] - self.cfg.target_range_z[0]) + self.cfg.target_range_z[0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the command.
        
        Args:
            env_ids: The environment indices to reset. If None, resets all.
        """
        self._generate_targets(env_ids)

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the command.
        
        Args:
            dt: The time step.
            
        Returns:
            The target position command. Shape: (num_envs, 3)
        """
        return self.target_pos_w 