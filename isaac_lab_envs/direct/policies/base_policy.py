#!/usr/bin/env python3

"""Base policy interface for model-based navigation methods"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional, Union


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
