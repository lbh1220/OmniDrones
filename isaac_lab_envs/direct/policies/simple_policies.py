#!/usr/bin/env python3

"""Simple model-based policies for testing"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from .base_policy import ModelBasedPolicy
try:
    import rvo2
except ImportError:
    rvo2 = None
    print("Warning: rvo2 not installed. ORCAPolicy will not be available.")

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
        self.neighbor_dist = 100.0
        self.max_neighbors = 10
        self.time_horizon = 11.0
        self.time_horizon_obst = 11.0
        self.safety_space = 1.5
        
        # Update with provided values
        for key, value in kwargs.items():
            setattr(self, key, value)

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


class ORCAPolicy(ModelBasedPolicy):
    """
    ORCA (Optimal Reciprocal Collision Avoidance) policy with obstacle avoidance
        RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors,
                     float timeHorizon, float timeHorizonObst, float radius,
                     float maxSpeed, const Vector2 & velocity)
        RVOSimulator set the default param for all agents, but could be set for each agent using addAgent()
        size_t addAgent(const Vector2 & position)
        size_t addAgent(const Vector2 & position, float neighborDist,
                        size_t maxNeighbors, float timeHorizon,
                        float timeHorizonObst, float radius, float maxSpeed,
                        const Vector2 & velocity)

    """
    
    def __init__(self, policy_cfg, env_cfg, name: str = "ORCA"):
        super().__init__(policy_cfg, env_cfg, name)
        
        if rvo2 is None:
            raise ImportError("rvo2 is required for ORCAPolicy. Please install it using: pip install rvo2")
        
        # Save env_cfg reference for accessing safety_radius
        self.env_cfg = env_cfg
        
        # ORCA parameters from environment config
        self.time_step = getattr(env_cfg, 'time_step', 0.5)
        self.max_speed = getattr(env_cfg, 'max_speed', 5.0)
        self.min_speed = getattr(env_cfg, 'min_speed', 0.0)
        self.v_pref = getattr(env_cfg, 'v_pref', 1.0)
        self.predict_timestep = getattr(env_cfg, 'pred_timestep', 2.0)
        
        # ORCA parameters from policy config (OrcaCfg)
        self.neighbor_dist = policy_cfg.neighbor_dist
        self.max_neighbors = policy_cfg.max_neighbors
        self.time_horizon = policy_cfg.time_horizon
        self.time_horizon_obst = policy_cfg.time_horizon_obst
        self.safety_space = policy_cfg.safety_space
        print(f"ORCA parameters: {self.time_step}, {self.predict_timestep}, {self.neighbor_dist}, {self.max_neighbors}, {self.time_horizon}, {self.time_horizon_obst}, {self.safety_space}")

    def _compute_orca_2d(self,
                          robot_pos_np: np.ndarray,
                          robot_vel_np: np.ndarray,
                          robot_radius_np: np.ndarray,
                          pref_vel_np: np.ndarray,
                          traffic_pos_np: np.ndarray,
                          traffic_vel_np: np.ndarray,
                          traffic_radius_np: np.ndarray) -> np.ndarray:
        """Compute ORCA 2D optimal velocities for each env using absolute coordinates.

        Args:
            robot_pos_np: [num_envs, 2]
            robot_vel_np: [num_envs, 2]
            robot_radius_np: [num_envs]
            pref_vel_np: [num_envs, 2] preferred velocity (XY)
            traffic_pos_np: [total_traffic, 2]
            traffic_vel_np: [total_traffic, 2]
            traffic_radius_np: [total_traffic]

        Returns:
            np.ndarray: [num_envs, 2] optimal XY velocities
        """
        num_envs = robot_pos_np.shape[0]
        corrected_velocities = np.zeros((num_envs, 2), dtype=np.float32)

        for env_idx in range(num_envs):
            sim = rvo2.PyRVOSimulator(
                self.time_step,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                float(robot_radius_np[env_idx]) + self.safety_space,
                self.max_speed
            )

            ego_pos = (float(robot_pos_np[env_idx, 0]), float(robot_pos_np[env_idx, 1]))
            ego_vel = (float(robot_vel_np[env_idx, 0]), float(robot_vel_np[env_idx, 1]))

            agent_id = sim.addAgent(
                ego_pos,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                float(robot_radius_np[env_idx]) + self.safety_space,
                self.max_speed,
                ego_vel
            )

            sim.setAgentPrefVelocity(agent_id, (float(pref_vel_np[env_idx, 0]), float(pref_vel_np[env_idx, 1])))

            ego_pos_2d = robot_pos_np[env_idx]
            total_traffic = 0 if traffic_pos_np is None else int(traffic_pos_np.shape[0])
            for traffic_idx in range(total_traffic):
                traffic_pos_2d = traffic_pos_np[traffic_idx]
                distance = np.linalg.norm(traffic_pos_2d - ego_pos_2d)
                if distance > self.neighbor_dist:
                    continue

                traffic_vel = (float(traffic_vel_np[traffic_idx, 0]), float(traffic_vel_np[traffic_idx, 1]))

                sim.addAgent(
                    (float(traffic_pos_2d[0]), float(traffic_pos_2d[1])),
                    self.neighbor_dist,
                    0,
                    self.time_horizon,
                    self.time_horizon_obst,
                    float(traffic_radius_np[traffic_idx]) + self.safety_space,
                    np.linalg.norm(traffic_vel) + 1e-8,
                    traffic_vel
                )

            sim.doStep()
            computed_vel = sim.getAgentVelocity(agent_id)
            corrected_velocities[env_idx] = np.array(computed_vel, dtype=np.float32)

        return corrected_velocities
    def _predict(self, state) -> torch.Tensor:

        robot_positions_2d = state.ego_drone.positions.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        robot_velocities_2d = state.ego_drone.velocities.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        # Robot radius is stored in environment config, not in state
        robot_radius = torch.full((robot_positions_2d.shape[0],), 
                                  self.env_cfg.safety_radius, 
                                  device=robot_positions_2d.device, dtype=torch.float32)
        
        if state.navigation.local_goals is not None:
            goal = state.navigation.local_goals.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        else:
            goal = state.navigation.target_positions.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        
        relative_goal = goal - robot_positions_2d
        goal_distance = torch.norm(relative_goal, dim=-1, keepdim=True) + 1e-8
        robot_pref_vel = relative_goal / goal_distance * self.v_pref
        
        # Traffic information (2D only)
        traffic_positions_2d = state.traffic.traffic_positions[:, :2]
        traffic_velocities_2d = state.traffic.traffic_velocities[:, :2]
        traffic_safety_radius = state.traffic.traffic_safety_radius

        robot_pos_np = self.to_numpy(robot_positions_2d)
        robot_vel_np = self.to_numpy(robot_velocities_2d)
        robot_radius_np = self.to_numpy(robot_radius)
        traffic_pos_np = self.to_numpy(traffic_positions_2d)
        traffic_vel_np = self.to_numpy(traffic_velocities_2d)
        traffic_radius_np = self.to_numpy(traffic_safety_radius)
        pref_vel_np = self.to_numpy(robot_pref_vel)

        corrected_vel_np = self._compute_orca_2d(
            robot_pos_np,
            robot_vel_np,
            robot_radius_np,
            pref_vel_np,
            traffic_pos_np,
            traffic_vel_np,
            traffic_radius_np,
        )

        corrected_velocity_2d = torch.from_numpy(corrected_vel_np).to(robot_positions_2d.device)
        return corrected_velocity_2d
    def act(self, observation, timestep, timesteps):
        # For skrl API, we do not need observation, timestep, timesteps
        # we use the state from binded env
        if self.env is None or self.env.state is None:
            return torch.zeros((1, 2), device=self.device)
        state_env = self.env.state
        raw_velocity_2d = self._predict(state_env)
        scaled_velocity_2d = raw_velocity_2d / self.max_speed
        # skrl's act function returns actions, log_prob, outputs
        return scaled_velocity_2d, None, None
    
    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Predict action using ORCA for collision avoidance
        FOR sb3 API
        Args:
            observation: Dictionary containing observation data
                - robot_node: [num_envs, 1, 9] robot state
                - temporal_edges: [num_envs, 1, 2] robot velocity
                - spatial_edges: [num_envs, total_traffic_num, spatial_dim] traffic info
                - detected_human_num: [num_envs, 1] number of detected traffic
            deterministic: Not used for model-based policy
            
        Returns:
            action: ORCA velocity [num_envs, 2]
            states: None
        """
        # Switch to env.state (do not rely on external observation)
        state_env = self.env.state
        raw_velocity_2d = self._predict(state_env)
        scaled_velocity_2d = raw_velocity_2d / self.max_speed
        return scaled_velocity_2d, None

    def compute_corrected_velocity(self, state) -> torch.Tensor:
        """
        Compute ORCA-corrected velocity using RL-generated velocity commands as preferred velocity
        
        Args:
            state: Environment state object containing:
                - ego_drone: robot state information
                - navigation: RL-generated velocity commands and targets
                - traffic: traffic agent positions, velocities, types, safety radius
                
        Returns:
            corrected_velocity: ORCA-corrected velocity commands [num_envs, 2]
        """
        if rvo2 is None:
            # If rvo2 not available, return original XY with z set to 0
            rl_velocity_commands = state.navigation.velocity_commands  # [num_envs, 1, 3] or [num_envs, 3]
            if rl_velocity_commands.dim() == 3:
                xy = rl_velocity_commands[:, 0, :2]
            else:
                xy = rl_velocity_commands[:, :2]
            zeros_z = torch.zeros_like(xy[:, :1])
            out = torch.cat([xy, zeros_z], dim=-1).unsqueeze(1)
            return out
        
        # Extract robot information (2D only)
        robot_positions_2d = state.ego_drone.positions.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        robot_velocities_2d = state.ego_drone.velocities.squeeze(1)[:, :2]  # [num_envs, 2] - only x,y
        rl_velocity_commands = state.navigation.velocity_commands.squeeze(1)  # [num_envs, 3]
        # Robot radius is stored in environment config, not in state
        robot_radius = torch.full((robot_positions_2d.shape[0],), 
                                  self.env_cfg.safety_radius, 
                                  device=robot_positions_2d.device, dtype=torch.float32)
        
        # Traffic information (2D only)
        traffic_positions_2d = state.traffic.traffic_positions[:, :2]  # [total_traffic, 2] - only x,y
        traffic_velocities_2d = state.traffic.traffic_velocities[:, :2]  # [total_traffic, 2] - only x,y  
        traffic_safety_radius = state.traffic.traffic_safety_radius  # [total_traffic]
        
        # Convert to numpy for CPU processing
        robot_pos_np = self.to_numpy(robot_positions_2d)
        robot_vel_np = self.to_numpy(robot_velocities_2d)
        rl_vel_commands_np = self.to_numpy(rl_velocity_commands)
        robot_radius_np = self.to_numpy(robot_radius)
        traffic_pos_np = self.to_numpy(traffic_positions_2d)
        traffic_vel_np = self.to_numpy(traffic_velocities_2d)
        traffic_radius_np = self.to_numpy(traffic_safety_radius)

        pref_vel_np = rl_vel_commands_np[:, :2]

        corrected_vel_np = self._compute_orca_2d(
            robot_pos_np,
            robot_vel_np,
            robot_radius_np,
            pref_vel_np,
            traffic_pos_np,
            traffic_vel_np,
            traffic_radius_np,
        )

        corrected_velocity_xy = torch.from_numpy(corrected_vel_np).to(robot_positions_2d.device)
        zeros_z = torch.zeros((corrected_velocity_xy.shape[0], 1), device=robot_positions_2d.device, dtype=corrected_velocity_xy.dtype)
        corrected_velocity = torch.cat([corrected_velocity_xy, zeros_z], dim=-1).unsqueeze(1)

        return corrected_velocity
