# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation

"""Action managers for the Forest environment."""

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


@configclass
class DroneActionCfg(ActionTermCfg):
    """Configuration for drone action term."""
    
    thrust_scale: float = 1.9  # thrust to weight ratio
    moment_scale: float = 0.01  # moment scaling factor
    

class DroneAction(ActionTerm):
    """Action term for applying thrust and moments to a drone."""
    
    cfg: DroneActionCfg

    def __init__(self, cfg: DroneActionCfg, env: "ManagerBasedRLEnv"):
        """Initialize the action term.
        
        Args:
            cfg: The configuration instance.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # get robot asset
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # get robot properties
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        
        # get body id for applying forces
        self._body_id = self._robot.find_bodies("body")[0]
        
        # thrust and moment buffers
        self._thrust = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self._moment = torch.zeros(env.num_envs, 1, 3, device=env.device)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return 4  # thrust + 3 moments (roll, pitch, yaw)

    def apply_actions(self, actions: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Apply actions to the robot.
        
        Args:
            actions: The actions to apply. Shape: (num_envs, 4)
                     actions[:, 0] = collective thrust
                     actions[:, 1:4] = roll, pitch, yaw moments
            env_ids: The environment indices to apply actions to. If None, applies to all.
        """
        # handle environment indices
        if env_ids is None:
            env_ids = slice(None)
        
        # clamp actions to [-1, 1]
        actions = actions.clamp(-1.0, 1.0)
        
        # convert thrust action to actual thrust force
        # action[0] from [-1, 1] -> thrust from [0, 2*thrust_scale*weight]
        self._thrust[env_ids, 0, 2] = (
            self.cfg.thrust_scale * self._robot_weight * (actions[env_ids, 0] + 1.0) / 2.0
        )
        
        # convert moment actions to actual moments
        self._moment[env_ids, 0, :] = self.cfg.moment_scale * actions[env_ids, 1:]
        
        # apply forces and torques to the robot
        self._robot.set_external_force_and_torque(
            self._thrust[env_ids], 
            self._moment[env_ids], 
            body_ids=self._body_id,
            env_ids=env_ids
        ) 