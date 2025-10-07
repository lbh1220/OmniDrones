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
        
        # ORCA parameters
        
        self.time_step = getattr(env_cfg, 'time_step', 0.5)
        self.max_speed = getattr(env_cfg, 'max_speed', 5.0)
        self.min_speed = getattr(env_cfg, 'min_speed', 0.0)
        self.v_pref = getattr(env_cfg, 'v_pref', 1.0)
        self.predict_timestep = getattr(env_cfg, 'pred_timestep', 2.0)
        self.neighbor_dist = getattr(policy_cfg, 'neighbor_dist', 100.0)
        self.max_neighbors = getattr(policy_cfg, 'max_neighbors', 10)
        self.time_horizon = getattr(policy_cfg, 'time_horizon', 10.0)
        self.time_horizon_obst = getattr(policy_cfg, 'time_horizon_obst', 10.0)
        self.safety_space = getattr(policy_cfg, 'safety_space', 1.5)
        print(f"ORCA parameters: {self.time_step}, {self.predict_timestep}, {self.neighbor_dist}, {self.max_neighbors}, {self.time_horizon}, {self.time_horizon_obst}, {self.safety_space}")
        
    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Predict action using ORCA for collision avoidance
        
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
        robot_node = observation['robot_node']  # [num_envs, 1, 9]
        robot_vel = observation['temporal_edges']  # [num_envs, 1, 2]
        spatial_edges = observation['spatial_edges']  # [num_envs, total_traffic_num, spatial_dim]
        detected_num = observation['detected_human_num']  # [num_envs, 1]
        
        num_envs = robot_node.shape[0]
        
        # Extract robot information
        robot_radius = robot_node[:, 0, 2]  # [num_envs]
        robot_v_pref = robot_node[:, 0, 3]  # [num_envs]
        local_goal_relative_pos = robot_node[:, 0, 5:7] * (2 * self.circle_radius) / self.observation_norm_scale  # [num_envs, 2]
        
        # Convert to numpy for CPU processing (safe conversion)
        robot_vel_np = self.to_numpy(robot_vel)  # [num_envs, 1, 2]
        local_goal_np = self.to_numpy(local_goal_relative_pos)  # [num_envs, 2]
        robot_radius_np = self.to_numpy(robot_radius)  # [num_envs]
        robot_v_pref_np = self.to_numpy(robot_v_pref)  # [num_envs]
        detected_num_np = self.to_numpy(detected_num)  # [num_envs, 1]
        spatial_edges_np = self.to_numpy(spatial_edges)  # [num_envs, total_traffic, spatial_dim]
        
        # Process each environment independently
        actions = np.zeros((num_envs, 2), dtype=np.float32)
        
        for env_idx in range(num_envs):
            # Create ORCA simulator for this environment
            sim = rvo2.PyRVOSimulator(
                self.time_step,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                robot_radius_np[env_idx] + self.safety_space,
                self.max_speed
            )
            
            # Add ego agent
            ego_pos = (0.0, 0.0)  # Ego is at origin in relative coordinates
            ego_vel = (robot_vel_np[env_idx, 0, 0], robot_vel_np[env_idx, 0, 1])
            
            agent_id = sim.addAgent(
                ego_pos,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                robot_radius_np[env_idx] + self.safety_space,
                self.max_speed,
                ego_vel
            )
            goal_distance = np.linalg.norm(local_goal_np[env_idx]) + 1e-8
            pref_vel = (local_goal_np[env_idx] / goal_distance) * robot_v_pref_np[env_idx]
            sim.setAgentPrefVelocity(agent_id, tuple(pref_vel))
            
            # Add traffic agents that are in range
            num_detected = int(detected_num_np[env_idx, 0])
            
            for traffic_idx in range(num_detected):
                # Extract traffic info from spatial_edges
                # spatial_edges format: [2*(predict_steps+1) + 1]
                # First 2 values are current relative position
                traffic_data = spatial_edges_np[env_idx, traffic_idx]
                
                # Check if this traffic is valid (not padding)
                if np.abs(traffic_data[0]) >= self.observation_norm_scale - 1e-3:
                    continue
                
                # Extract current position (first 2 values, normalized)
                traffic_rel_pos = traffic_data[:2] * (2 * self.circle_radius) / self.observation_norm_scale
                traffic_radius = traffic_data[-1]  # Last value is safety radius
                
                # Estimate velocity from position change
                # If we have predicted positions, we can use them
                # Otherwise, assume constant velocity
                if len(traffic_data) >= 4:
                    next_pos = traffic_data[2:4] * (2 * self.circle_radius) / self.observation_norm_scale
                    traffic_vel = (next_pos - traffic_rel_pos) / self.predict_timestep
                else:
                    traffic_vel = np.zeros(2)
                
                # Add traffic agent
                sim.addAgent(
                    tuple(traffic_rel_pos),
                    self.neighbor_dist,
                    0,  # Traffic agents don't need neighbors
                    self.time_horizon,
                    self.time_horizon_obst,
                    traffic_radius + self.safety_space,
                    np.linalg.norm(traffic_vel) + 1e-8,
                    tuple(traffic_vel)
                )
            
            # Run ORCA simulation
            sim.doStep()
            
            # Get computed velocity
            computed_vel = sim.getAgentVelocity(agent_id)
            
            # Normalize to unit direction (environment will scale by max_speed)
            actions[env_idx] = np.array(computed_vel) / self.max_speed
        
        # Convert back to torch tensor
        action = torch.from_numpy(actions).to(self.device)
        
        return action, None
