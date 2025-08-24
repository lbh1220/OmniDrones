# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
TrafficSimulator - A dynamic traffic simulation module for Isaac Lab reinforcement learning environments.

This module implements a simplified traffic simulation system adapted from AirSim to Isaac Sim,
providing background traffic drones that can interact with RL agents.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from omni.isaac.core.utils import prims as prim_utils
from omni.isaac.core.utils import stage as stage_utils

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers.lee_position_controller import LeePositionController
from omni_drones.utils.torch import quat_axis, normalize


@dataclass
class TrafficConfig:
    """Configuration for traffic simulation."""
    
    # Basic settings
    num_drones: int = 10
    drone_model: str = "crazyflie"  # drone model to use
    
    # Flight parameters
    flight_height: float = 5.0  # meters
    max_speed: float = 3.0  # m/s
    arrival_threshold: float = 1.0  # meters
    
    # Area bounds
    area_bounds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.area_bounds is None:
            self.area_bounds = {
                "xmin": -20.0,
                "xmax": 20.0,
                "ymin": -20.0,
                "ymax": 20.0
            }


class TrafficAircraft:
    """Individual traffic aircraft based on MultirotorBase."""
    
    def __init__(self, name: str, drone: MultirotorBase, controller: LeePositionController, 
                 config: TrafficConfig, device: str = "cuda", drone_index: int = 0):
        self.name = name
        self.drone = drone
        self.controller = controller
        self.config = config
        self.device = device
        self.drone_index = drone_index  # Index in the batch of drones
        
        # Flight state
        self.current_target = torch.zeros(3, device=device)
        self.velocity_command = torch.zeros(3, device=device)
        self.is_at_target = False
        self.target_updated_time = 0.0
        
        # Navigation parameters
        self.max_speed = config.max_speed
        self.arrival_threshold = config.arrival_threshold
        
    def set_target(self, target: torch.Tensor):
        """Set a new target position for the aircraft."""
        self.current_target = target.clone()
        self.is_at_target = False
        self.target_updated_time = 0.0
        
    def update_control(self, dt: float) -> torch.Tensor:
        """Update aircraft control based on current target."""
        # Get current position using the correct drone index
        current_pos = self.drone.pos[self.drone_index]
        
        # Calculate direction to target
        direction = self.current_target - current_pos
        distance = torch.norm(direction)
        
        # Check if arrived
        if distance < self.arrival_threshold:
            self.is_at_target = True
            self.velocity_command.zero_()
        else:
            # Calculate velocity command
            if distance > 0:
                direction_normalized = direction / distance
                # Simple speed control: full speed if far, slow down when close
                speed = min(self.max_speed, distance * 0.5)
                speed = max(speed, 0.1)  # minimum speed
                self.velocity_command = direction_normalized * speed
        
        # Use controller to convert velocity command to rotor commands
        root_state = self.drone.get_state(env_frame=False)[self.drone_index]
        
        # Create target for Lee controller (current position + velocity for next step)
        target_pos = current_pos + self.velocity_command * dt
        target_vel = self.velocity_command
        
        rotor_commands = self.controller.compute(
            root_state=root_state.unsqueeze(0),
            target_pos=target_pos.unsqueeze(0),
            target_vel=target_vel.unsqueeze(0)
        )
        
        return rotor_commands.squeeze(0)


class TrafficSimulator:
    """
    Main traffic simulation class that manages multiple traffic aircraft.
    
    This class is designed to be used as a plugin in Isaac Lab RL environments,
    providing dynamic background traffic that doesn't interfere with environment cloning.
    """
    
    def __init__(self, config: TrafficConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Traffic namespace - not cloned across environments
        self.traffic_prim_path = "/World/Traffic"
        
        # Aircraft storage
        self.aircraft: Dict[str, TrafficAircraft] = {}
        self.aircraft_list: List[TrafficAircraft] = []
        
        # Simulation state
        self.is_initialized = False
        self.step_count = 0
        
        # Target management
        self.target_generator = TargetGenerator(config)
        
        logging.info(f"TrafficSimulator initialized with {config.num_drones} aircraft")
        self.drone = None
    

    def create_traffic_prim(self):
        # Create traffic namespace
        if not prim_utils.is_prim_path_valid(self.traffic_prim_path):
            prim_utils.define_prim(self.traffic_prim_path)
        self.drone, self.controller = MultirotorBase.make(drone_model=self.config.drone_model,
                                            device=self.device,
                                            controller="LeePositionController")
        # Spawn all aircraft with the generated positions
        # Generate initial positions for all aircraft
        initial_positions = []
        prim_paths = []
        for i in range(self.config.num_drones):
            pos = self._generate_random_position()
            initial_positions.append((pos[0].item(), pos[1].item(), pos[2].item()))
            prim_paths.append(f"{self.traffic_prim_path}/traffic_drone_{i}")
        aircraft_prims = self.drone.spawn(
            translations=initial_positions, 
            prim_paths=prim_paths
        )

    def initialize(self):
        """Initialize the traffic simulation system."""
        if self.is_initialized:
            return
            

        
        # Create aircraft
        self._initialize_aircraft()
        
        # Set initial targets
        self._assign_initial_targets()
        
        self.is_initialized = True
        logging.info("TrafficSimulator initialization complete")
    
    def _initialize_aircraft(self):
        """Create all traffic aircraft."""
        
        # Initialize the drone view (this sets up all cloned drones)
        self.drone.initialize(prim_paths_expr=f"{self.traffic_prim_path}/traffic_drone_*")
        
        # Create controllers and TrafficAircraft wrappers for each drone
        for i in range(self.config.num_drones):
            aircraft_name = f"traffic_drone_{i}"
            
            # Create controller for this aircraft
            controller = LeePositionController(
                g=9.81,
                uav_params=self.drone.params
            ).to(self.device)
            
            # Create TrafficAircraft wrapper
            aircraft = TrafficAircraft(
                name=aircraft_name,
                drone=self.drone,  # All aircraft share the same drone view but use different indices
                controller=controller,
                config=self.config,
                device=self.device,
                drone_index=i  # Add index to identify which drone in the batch
            )
            
            self.aircraft[aircraft_name] = aircraft
            self.aircraft_list.append(aircraft)
    
    def _assign_initial_targets(self):
        """Assign initial targets to all aircraft."""
        for aircraft in self.aircraft_list:
            target = self.target_generator.generate_target()
            aircraft.set_target(target)
    
    def _generate_random_position(self) -> torch.Tensor:
        """Generate a random position within the area bounds."""
        bounds = self.config.area_bounds
        x_range = bounds["xmax"] - bounds["xmin"]
        y_range = bounds["ymax"] - bounds["ymin"]
        
        x = torch.rand(1, device=self.device) * x_range + bounds["xmin"]
        y = torch.rand(1, device=self.device) * y_range + bounds["ymin"]
        z = torch.tensor([self.config.flight_height], device=self.device)
        return torch.cat([x, y, z])
    
    def step(self, dt: float = 0.02):
        """Execute one simulation step."""
        if not self.is_initialized:
            self.initialize()
        
        # Collect all rotor commands in a batch
        all_rotor_commands = []
        
        # Update each aircraft
        for aircraft in self.aircraft_list:
            # Generate new target if current one is reached
            if aircraft.is_at_target:
                new_target = self.target_generator.generate_target()
                aircraft.set_target(new_target)
            
            # Update control
            rotor_commands = aircraft.update_control(dt)
            all_rotor_commands.append(rotor_commands)
        
        # Apply all commands to the drone batch at once
        if all_rotor_commands:
            batch_commands = torch.stack(all_rotor_commands)
            self.aircraft_list[0].drone.apply_action(batch_commands)
        
        self.step_count += 1
    
    def get_aircraft_states(self) -> Dict[str, torch.Tensor]:
        """Get states of all traffic aircraft."""
        states = {}
        for name, aircraft in self.aircraft.items():
            state = aircraft.drone.get_state(env_frame=False)
            states[name] = state[0]  # get first drone state
        return states
    
    def get_aircraft_positions(self) -> torch.Tensor:
        """Get positions of all traffic aircraft as a tensor."""
        if not self.aircraft_list:
            return torch.empty(0, 3, device=self.device)
        
        # Since all aircraft share the same drone view, we can get all positions at once
        return self.aircraft_list[0].drone.pos[:len(self.aircraft_list)]
    
    def get_aircraft_velocities(self) -> torch.Tensor:
        """Get velocities of all traffic aircraft as a tensor."""
        if not self.aircraft_list:
            return torch.empty(0, 3, device=self.device)
        
        # Get linear velocities for all traffic aircraft
        all_velocities = self.aircraft_list[0].drone.vel[:len(self.aircraft_list), :3]
        return all_velocities
    
    def check_collision(self, external_positions: torch.Tensor, safety_radius: float = 2.0) -> torch.Tensor:
        """
        Check for potential collisions between external aircraft and traffic aircraft.
        
        Args:
            external_positions: Tensor of shape (N, 3) containing positions of external aircraft
            safety_radius: Safety distance threshold
            
        Returns:
            Tensor of shape (N,) containing boolean values indicating collision risk
        """
        if len(self.aircraft_list) == 0:
            return torch.zeros(external_positions.shape[0], dtype=torch.bool, device=self.device)
        
        traffic_positions = self.get_aircraft_positions()
        
        # Calculate distances between all external and traffic aircraft
        # external_positions: (N, 3), traffic_positions: (M, 3)
        # distances: (N, M)
        distances = torch.cdist(external_positions, traffic_positions)
        
        # Check if any distance is below safety threshold
        collision_risk = (distances < safety_radius).any(dim=1)
        
        return collision_risk
    
    def get_aircraft_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all traffic aircraft."""
        info = {
            "num_drones": len(self.aircraft_list),
            "positions": self.get_aircraft_positions(),
            "velocities": self.get_aircraft_velocities(),
            "flight_height": self.config.flight_height,
            "area_bounds": self.config.area_bounds,
            "step_count": self.step_count
        }
        return info
    
    def reset(self):
        """Reset the traffic simulation."""
        if not self.aircraft_list:
            return
            
        # Generate new positions for all aircraft
        new_positions = []
        new_rotations = []
        for i in range(len(self.aircraft_list)):
            new_pos = self._generate_random_position()
            new_positions.append(new_pos)
            new_rotations.append(torch.tensor([1., 0., 0., 0.], device=self.device))
        
        # Set all positions at once
        positions_batch = torch.stack(new_positions)
        rotations_batch = torch.stack(new_rotations)
        
        self.aircraft_list[0].drone.set_world_poses(positions_batch, rotations_batch)
        
        # Reset drone states
        env_ids = torch.arange(1, device=self.device)
        self.aircraft_list[0].drone._reset_idx(env_ids)
        
        # Assign new targets
        for aircraft in self.aircraft_list:
            new_target = self.target_generator.generate_target()
            aircraft.set_target(new_target)
        
        self.step_count = 0
        logging.info("TrafficSimulator reset complete")


class TargetGenerator:
    """Generate flight targets for traffic aircraft."""
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.bounds = config.area_bounds
        self.device = "cuda"  # Assume GPU device
    
    def generate_target(self) -> torch.Tensor:
        """Generate a random target position within bounds."""
        x_range = self.bounds["xmax"] - self.bounds["xmin"]
        y_range = self.bounds["ymax"] - self.bounds["ymin"]
        
        x = torch.rand(1, device=self.device) * x_range + self.bounds["xmin"]
        y = torch.rand(1, device=self.device) * y_range + self.bounds["ymin"]
        z = torch.tensor([self.config.flight_height], device=self.device)
        return torch.cat([x, y, z])
    
    def generate_waypoint_targets(self, num_waypoints: int = 3) -> List[torch.Tensor]:
        """Generate a series of waypoint targets for more complex flight patterns."""
        targets = []
        for _ in range(num_waypoints):
            targets.append(self.generate_target())
        return targets
