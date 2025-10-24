from abc import ABC, abstractmethod
import math
import torch
from typing import Dict

from omni.isaac.lab.utils import configclass

import gymnasium as gym
import numpy as np

@configclass
class ActionManagerCfg:
    action_space_type: str = "discrete"
    action_space_num_per_dim: int = 7
    action_mode: str = "velocity_components"  # or "speed_direction"
    # 未来可能还会扩展到用world or body frame


class ActionManager(ABC):
    """Base action manager interface."""
    def __init__(self, cfg: ActionManagerCfg, env):
        self.cfg = cfg
        self.env = env
        self.action_mapping: torch.Tensor | None = None
        

    def get_action_space(self) -> gym.spaces.Space:
        """Get single_action_space for the action manager."""
        self.num_actions = self.env.cfg.num_actions
        if self.cfg.action_space_type == "discrete":
            total_actions = self.cfg.action_space_num_per_dim * self.cfg.action_space_num_per_dim
            return gym.spaces.Discrete(total_actions)
        elif self.cfg.action_space_type == "beta":
            return gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_actions,))
        else:
            # isaac lab的default action是unbounded的
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Decode raw policy actions and write commands into env/state."""
        raise NotImplementedError

    @abstractmethod
    def apply_action(self) -> None:
        """Compute rotor commands and actuate the robot from the commands."""
        raise NotImplementedError




class VelocityXYActionManager(ActionManager):
    """VelocityXY-based manager supporting multiple spaces and modes."""

    def __init__(self, cfg: ActionManagerCfg, env):
        super().__init__(cfg, env)
        self.command_vel_xy = None
        self.action_mapping = None


    def process_actions(self, actions: torch.Tensor) -> None:



        if self.cfg.action_space_type == "discrete":
            if self.action_mapping is None:
                self._setup_discrete_action()
        env = self.env
        env_cfg = env.cfg


        # 1) discrete -> continuous if needed
        if self.cfg.action_space_type == "discrete":
            continuous_actions = self.discrete_to_continuous_action(actions)
            if self.cfg.action_mode == "speed_direction":
                raise ValueError("Speed direction action mode not supported for discrete action space")
        else:
            continuous_actions = actions

        # 2) build (vx, vy)
        if self.cfg.action_mode == "velocity_components":
            command_vel_xy = self._process_velocity_components(continuous_actions)
        elif self.cfg.action_mode == "speed_direction":
            command_vel_xy = self._process_speed_direction(continuous_actions)
        else:
            raise ValueError(f"Unknown action mode: {self.cfg.action_mode}")

        # 3) scale to respect min/max speed (allow near-zero)
        speed_magnitude = torch.norm(command_vel_xy, dim=-1, keepdim=True)
        eps = 1e-3
        scale_factor = torch.ones_like(speed_magnitude)
        too_fast = speed_magnitude > env_cfg.max_speed
        scale_factor = torch.where(too_fast, env_cfg.max_speed / speed_magnitude, scale_factor)
        too_slow = (speed_magnitude > eps) & (speed_magnitude < env_cfg.min_speed)
        scale_factor = torch.where(too_slow, env_cfg.min_speed / speed_magnitude, scale_factor)
        command_vel_xy = command_vel_xy * scale_factor

        # 4) write back to env/state
        self.command_vel_xy = command_vel_xy.unsqueeze(1)  # [num_envs, 1, 2]
        env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()

    def _process_velocity_components(self, actions: torch.Tensor) -> torch.Tensor:
        env_cfg = self.env.cfg
        if self.cfg.action_space_type == "beta":
            actions_scaled = (actions * 2.0) - 1.0
            command_vel_xy = actions_scaled * env_cfg.max_speed
        else:
            command_vel_xy = actions * env_cfg.max_speed
        return command_vel_xy

    def _process_speed_direction(self, actions: torch.Tensor) -> torch.Tensor:
        env_cfg = self.env.cfg
        if self.cfg.action_space_type == "beta":
            speed = actions[:, 0] * (env_cfg.max_speed - env_cfg.min_speed) + env_cfg.min_speed
            direction = actions[:, 1] * 2.0 * math.pi
        else:
            normalized_speed = (actions[:, 0] + 1.0) / 2.0
            speed = normalized_speed * (env_cfg.max_speed - env_cfg.min_speed) + env_cfg.min_speed
            direction = (actions[:, 1] + 1.0) / 2.0 * 2.0 * math.pi
        vx = speed * torch.cos(direction)
        vy = speed * torch.sin(direction)
        return torch.stack([vx, vy], dim=1)

    def apply_action(self) -> None:
        env = self.env
        drone_state = env.drone.get_state(env_frame=False)[..., :13]
        if self.command_vel_xy is None:
            self.command_vel_xy = torch.zeros(env.num_envs, 1, 2, device=env.device)
            env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        target_height = env.cfg.flight_height * torch.ones(env.num_envs, 1, 1, device=env.device)
        rotor_commands = env.controller.compute(
            root_state=drone_state,
            target_vel_xy=self.command_vel_xy,
            target_height=target_height,
        )
        env.drone.apply_action(rotor_commands)

    def _setup_discrete_action(self):
        """设置离散动作空间"""
        total_actions = self.cfg.action_space_num_per_dim * self.cfg.action_space_num_per_dim
        self._create_discrete_action_mapping()
        print(f"Created discrete action mapping: {self.cfg.action_space_num_per_dim}x{self.cfg.action_space_num_per_dim} = {total_actions} actions")
    
    def _create_discrete_action_mapping(self):
        """创建离散动作映射"""
        speed_values = torch.linspace(-1.0, 1.0, self.cfg.action_space_num_per_dim, device=self.env.device)
        
        action_mapping = []
        for i in range(self.cfg.action_space_num_per_dim):
            for j in range(self.cfg.action_space_num_per_dim):
                vx = speed_values[i]
                vy = speed_values[j]
                action_mapping.append([vx, vy])
        
        self.action_mapping = torch.tensor(action_mapping, device=self.env.device, dtype=torch.float32)
    
    def discrete_to_continuous_action(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        """将离散动作转换为连续动作
        
        Args:
            discrete_actions: [num_envs] or [num_envs, 1] 离散动作索引
            
        Returns:
            continuous_actions: [num_envs, 2] 连续动作 (vx, vy)
        """
        # 将float32转换为整数索引（处理vec env的numpy/tensor转换）
        if discrete_actions.ndim == 2:
            discrete_actions = discrete_actions.squeeze(1)
        discrete_actions = discrete_actions.long()
        
        # 处理超出范围的动作索引
        discrete_actions = torch.clamp(discrete_actions, 0, len(self.action_mapping) - 1)
        
        # 批量索引映射
        continuous_actions = self.action_mapping[discrete_actions]
        
        return continuous_actions
    