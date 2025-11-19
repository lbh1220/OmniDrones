#!/usr/bin/env python3

"""Base policy interface for model-based navigation methods"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional, Union




class PolicyConfig:
    """Simple config class for model-based policies"""
    def __init__(self, **kwargs):
        # Default values
        self.arrival_threshold = 2.0
        self.repulsion_gain = 2.0
        self.attraction_gain = 1.0
        self.obstacle_threshold = 15.0
        self.lookahead_distance = 3.0
        
        # ORCA-specific parameters
        self.time_step = 0.16
        self.neighbor_dist = 1000.0
        self.max_neighbors = 10
        self.time_horizon = 11.0
        self.time_horizon_obst = 11.0
        self.safety_space = 5.0
        
        # PDC-specific parameters (Practical Distributed Control)
        # Gains and shape parameters
        # best practice parameters for small scale experiments
        # [1.0, 2.0, 1e-5, 1e-6, 1.0, 2.0, 4.0]
        # best practice parameters for large scale experiments
        # [1.0, 1.0, 1e-5, 1e-6, 1.0, 3.0, 4.0]
        self.k1 = 1.0              # attraction gain
        self.k2 = 2.0           # repulsion gain
        self.epsilon = 1e-5         # epsilon in denominator
        self.epsilon_s = 1e-6       # epsilon for s(x) smoothing
        self.l_i = 1.0            # filter gain for xi = p + v / l_i
        # Interaction distance scales (multipliers on sum of radii)
        self.d1 = 2.0
        self.d2 = 4.0
        
        # Update with provided values
        for key, value in kwargs.items():
            setattr(self, key, value)

class BasePolicy(ABC):
    """Base class for all model-based policies"""
    
    def __init__(self, name: str = "BasePolicy"):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = None

    def bind_env(self, env):
        self.env = env

    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Predict action given observation
        
        Args:
            observation: Dictionary containing observation data
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Action tensor [num_envs, action_dim]
            states: Optional internal states (None for model-based methods)
        """
        pass
    
    def reset(self):
        """Reset policy internal states if any"""
        pass
    
    def to_torch_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Convert data to torch tensor and move to correct device
        
        Args:
            data: Input data as torch tensor or numpy array
            
        Returns:
            torch.Tensor: Converted tensor on self.device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.from_numpy(data).to(self.device)
    
    def to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert data to numpy array (moves to CPU if needed)
        
        Args:
            data: Input data as torch tensor or numpy array
            
        Returns:
            np.ndarray: Converted numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return data


class ModelBasedPolicy(BasePolicy):
    """Base class for model-based navigation policies"""
    
    def __init__(self, policy_cfg, env_cfg, name: str = "ModelBased_Policy"):
        super().__init__(name)
        self.cfg = policy_cfg
        self.env_cfg = env_cfg
        

    
    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Convert observation to action
        Default implementation returns zero action, subclasses should override
        """
        num_envs = observation['robot_node'].shape[0]
        action = torch.zeros((num_envs, 2), device=self.device)
        return action, None



class PurePursuitPolicy(ModelBasedPolicy):
    """Pure pursuit policy following local goal from global path"""
    
    def __init__(self, policy_cfg, env_cfg, name: str = "PurePursuit"):
        super().__init__(policy_cfg, env_cfg, name)
        self.max_speed = getattr(env_cfg, 'max_speed', 5.0)
        self.min_speed = getattr(env_cfg, 'min_speed', 0.0)
        self.v_pref = getattr(env_cfg, 'v_pref', 1.0)


    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Predict action by computing direction toward local goal
        
        Args:
            observation: Dictionary containing observation data
                - robot_node: [num_envs, 1, 9] containing:
                    [0:2]: relative final goal position (normalized)
                    [2]: robot radius
                    [3]: robot max speed
                    [4]: robot yaw
                    [5:7]: relative local goal position (normalized)
                    [7:9]: relative projection position (normalized)
            deterministic: Not used for model-based policy
            
        Returns:
            action: Normalized velocity direction [num_envs, 2]
            states: None (no internal states)
        """
        robot_node = observation['robot_node']  # [num_envs, 1, 9]
        
        # Extract local goal position (already relative to robot)
        local_goal_relative_pos = robot_node[:, 0, 5:7] * (2 * self.circle_radius) / self.observation_norm_scale  # [num_envs, 2]
        
        # Ensure correct tensor type and device
        local_goal_relative_pos = self.to_torch_tensor(local_goal_relative_pos)
        
        # Calculate normalized direction toward local goal
        pursuit_distance = torch.norm(local_goal_relative_pos, dim=1, keepdim=True)
        desired_velocity = local_goal_relative_pos / (pursuit_distance + 1e-8) * self.v_pref # [num_envs, 2]

        desired_velocity = desired_velocity / self.max_speed

        return desired_velocity, None
