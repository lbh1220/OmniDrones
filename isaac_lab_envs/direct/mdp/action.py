from abc import ABC, abstractmethod
import math
import torch
from typing import Dict

from omni.isaac.lab.utils import configclass

import gymnasium as gym
import numpy as np
from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
@configclass
class ActionManagerCfg:
    action_space_type: str = "discrete"
    action_space_num_per_dim: int = 7
    action_mode: str = "velocity_components"  # or "speed_direction" or "direction_acceleration"
    rl_action_frame: str = "body"  # or "world" # rl 输出的动作坐标系， action这边应该将其转换为world frame

def body_to_world(body_xy: torch.Tensor, cy: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
    x = body_xy[..., 0]
    y = body_xy[..., 1]
    num_extra_dims = x.dim() - cy.dim()
    if num_extra_dims > 0:
        # 在 cy 和 sy 的末尾添加所需的 singleton 维度
        cy = cy.view(cy.shape + (1,) * num_extra_dims)
        sy = sy.view(sy.shape + (1,) * num_extra_dims)

    world_x = x * cy - y * sy
    world_y = x * sy + y * cy
    return torch.stack([world_x, world_y], dim=-1)

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




class AccelerationActionManager(ActionManager):
    """Acceleration-level manager using (dv, d_theta) actions in [-1, 1].
    
    - dv: change in speed (m/s), scaled from [-1, 1] to [-dv_limit, dv_limit]
    - d_theta: change in velocity heading (rad), scaled from [-1, 1] to [-dtheta_limit, dtheta_limit]
    
    Always operates in world frame since actions are relative deltas, not absolute components.
    """
    def __init__(self, cfg: ActionManagerCfg, env):
        super().__init__(cfg, env)
        self.command_vel_xy = None
        self.action_mapping = None  # for discrete grid mapping (index -> [dv_norm, dtheta_norm])

    def _get_action_limits(self):
        """Resolve dv and dtheta limits from env.cfg with sensible defaults."""
        env_cfg = self.env.cfg
        max_speed = getattr(env_cfg, "max_speed", 1.0)
        # dv_limit: symmetric around 0
        dv_limit = getattr(env_cfg, "dv_limit", 0.2 * max_speed)
        # dtheta_limit: prefer degrees if provided, else radians, fallback 30 deg
        if hasattr(env_cfg, "dtheta_limit_deg"):
            dtheta_limit = math.radians(getattr(env_cfg, "dtheta_limit_deg"))
        elif hasattr(env_cfg, "dtheta_limit_rad"):
            dtheta_limit = getattr(env_cfg, "dtheta_limit_rad")
        else:
            dtheta_limit = math.radians(5.0)
        return float(dv_limit), float(dtheta_limit), float(max_speed)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map (dv_norm, dtheta_norm) to world-frame velocity command."""
        env = self.env
        device = env.device
        # Prepare normalized actions in [-1, 1]: [N, 2]
        if self.cfg.action_space_type == "discrete":
            if self.action_mapping is None:
                self._setup_discrete_action()
            actions_norm = self._discrete_to_continuous_action(actions)  # [-1,1] grid
        elif self.cfg.action_space_type == "beta":
            # actions expected in [0,1] -> map to [-1,1]
            if actions.ndim == 1 and actions.shape[0] == 2:
                actions = actions.unsqueeze(0)
            if actions.ndim != 2 or actions.shape[1] != 2:
                raise ValueError(f"Expected beta actions shape [N,2] or [2], got {actions.shape}")
            actions_norm = (torch.clamp(actions, 0.0, 1.0) * 2.0) - 1.0
        else:
            # unbounded or other: clamp to [-1,1] assuming policy outputs deltas already
            if actions.ndim == 1 and actions.shape[0] == 2:
                actions = actions.unsqueeze(0)
            if actions.ndim != 2 or actions.shape[1] != 2:
                raise ValueError(f"Expected actions shape [N,2] or [2], got {actions.shape}")
            actions_norm = torch.clamp(actions, -1.0, 1.0)

        dv_limit, dtheta_limit, max_speed = self._get_action_limits()
        # Scale from [-1, 1] to actual ranges
        dv = actions_norm[:, 0] * dv_limit
        dtheta = actions_norm[:, 1] * dtheta_limit

        # Current world-frame velocity and yaw
        vel_xy = self._get_current_velocity_world()  # [N, 2]
        speed = torch.norm(vel_xy, dim=-1)           # [N]
        base_theta = self._get_base_heading(vel_xy, speed)  # [N]

        # Apply deltas
        new_speed = torch.clamp(speed + dv, min=0.0, max=max_speed)
        new_theta = base_theta + dtheta

        vx = new_speed * torch.cos(new_theta)
        vy = new_speed * torch.sin(new_theta)
        command_vel_xy = torch.stack([vx, vy], dim=-1)  # [N, 2]

        self.command_vel_xy = command_vel_xy.unsqueeze(1)  # [N,1,2]
        env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()

    def _get_current_velocity_world(self) -> torch.Tensor:
        """Return current XY velocity in world frame: [N, 2]."""
        vel_xy = self.env.state.ego_drone.velocities[:, :, :2]  # [N,1,2]
        return vel_xy.squeeze(1)

    def _get_base_heading(self, vel_xy: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Compute current heading; if speed is tiny, fall back to robot yaw."""
        eps = 1e-5
        theta_vel = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])  # [N]
        robot_quat = self.env.state.ego_drone.rotations      # [N,1,4]
        robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1].squeeze(1)  # [N]
        small = speed <= eps
        return torch.where(small, robot_yaw, theta_vel)

    def _setup_discrete_action(self):
        """Build uniform grid mapping for (dv_norm, dtheta_norm) in [-1,1]^2."""
        total_actions = self.cfg.action_space_num_per_dim * self.cfg.action_space_num_per_dim
        self._create_discrete_action_mapping()
        print(f"Created discrete accel action mapping: {self.cfg.action_space_num_per_dim}x{self.cfg.action_space_num_per_dim} = {total_actions} actions")

    def _create_discrete_action_mapping(self):
        values = torch.linspace(-1.0, 1.0, self.cfg.action_space_num_per_dim, device=self.env.device)
        mapping = []
        for i in range(self.cfg.action_space_num_per_dim):
            for j in range(self.cfg.action_space_num_per_dim):
                dv_norm = values[i]
                dtheta_norm = values[j]
                mapping.append([dv_norm, dtheta_norm])
        self.action_mapping = torch.tensor(mapping, device=self.env.device, dtype=torch.float32)

    def _discrete_to_continuous_action(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        """Map discrete index [N] or [N,1] to normalized [-1,1] (dv_norm, dtheta_norm)."""
        if discrete_actions.ndim == 2:
            discrete_actions = discrete_actions.squeeze(1)
        discrete_actions = discrete_actions.long()
        discrete_actions = torch.clamp(discrete_actions, 0, len(self.action_mapping) - 1)
        return self.action_mapping[discrete_actions]

    def apply_action(self) -> None:
        env = self.env
        if self.command_vel_xy is None:
            self.command_vel_xy = torch.zeros(env.num_envs, 1, 2, device=env.device)
            env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        drone_state = env.drone.get_state(env_frame=False)[..., :13]
        if torch.isnan(drone_state).any():
            print(f"ActionManager: drone_state is nan: {drone_state}")
            return
        target_height = env.cfg.flight_height * torch.ones(env.num_envs, 1, 1, device=env.device)
        rotor_commands = env.controller.compute(
            root_state=drone_state,
            target_vel_xy=self.command_vel_xy,
            target_height=target_height,
        )
        env.drone.apply_action(rotor_commands)


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

        self.command_vel_xy = self._convert_action_frame(command_vel_xy)

        env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        # # --- Temporary clamp: prevent moving away from path when far from projection ---
        # try:
        #     state = env.state
        #     proj_pts = getattr(state.navigation, "projection_points", None)
        #     positions = getattr(state.ego_drone, "positions", None)
        #     if (proj_pts is not None) and (positions is not None) and proj_pts.numel() > 0 and positions.numel() > 0:
        #         # Vector from current position to projection point (XY)
        #         to_proj_xy = (proj_pts - positions).squeeze(1)[..., :2]  # [N,2]
        #         dist_xy = torch.norm(to_proj_xy, dim=-1)                 # [N]
        #         safety_radius = float(getattr(env.cfg, "safety_radius", 1.0))
        #         threshold = 1.5 * safety_radius
        #         # Dot between to-projection vector and current command velocity
        #         cmd_xy = self.command_vel_xy.squeeze(1)                  # [N,2]
        #         dot_val = torch.sum(to_proj_xy * cmd_xy, dim=-1)         # [N]
        #         # If far from path and moving away (dot < 0), clamp command to zero
        #         mask = (dist_xy > threshold) & (dot_val < 0.0)
        #         if torch.any(mask):
        #             cmd_xy[mask] = 0.0
        #             self.command_vel_xy = cmd_xy.unsqueeze(1)
        #             env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        # except Exception:
        #     # Best-effort safeguard; do not interfere with control if anything goes wrong
        #     pass
    def _convert_action_frame(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        if self.cfg.rl_action_frame == "world":
            return velocity_commands.unsqueeze(1) # [num_envs, 1, 2]
        elif self.cfg.rl_action_frame == "body":
            robot_quat = self.env.state.ego_drone.rotations
            robot_yaw = quaternion_to_euler(robot_quat)[:, :, -1]  # [N,1]
            cy = torch.cos(robot_yaw)
            sy = torch.sin(robot_yaw)
            world_velocity_commands = body_to_world(velocity_commands.unsqueeze(1), cy, sy)
            return world_velocity_commands # [num_envs, 1, 2]
        else:
            raise ValueError(f"Unknown action frame: {self.cfg.rl_action_frame}")
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
        if self.command_vel_xy is None:
            self.command_vel_xy = torch.zeros(env.num_envs, 1, 2, device=env.device)
            env.state.navigation.velocity_commands[:, :, :2] = self.command_vel_xy.clone()
        drone_state = env.drone.get_state(env_frame=False)[..., :13]
        if torch.isnan(drone_state).any():
            print(f"ActionManager: drone_state is nan: {drone_state}")
            return
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
    