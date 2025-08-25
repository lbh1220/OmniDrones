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

from omni_drones.traffic.traffic_drone_manager import TrafficDroneManager
from omni_drones.traffic.traffic_evtol_manager import TrafficEVTOLManager
# from omni_drones.traffic.utils.config import TrafficConfig




class TrafficSimulator:
    """
    Main traffic simulation class that manages traffic aircraft.
    
    This class is designed to be used as a plugin in Isaac Lab RL environments,
    providing dynamic background traffic that doesn't interfere with environment cloning.
    
    Architecture follows the pattern from AirSim traffic simulator with dedicated
    manager classes for different aircraft types.
    """
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Traffic namespace - not cloned across environments
        self.traffic_prim_path = "/World/Traffic"
        
        # Aircraft managers - similar to AirSim architecture
        self.drone_manager = TrafficDroneManager(
            config, device, self.traffic_prim_path
        ) if config.num_drones > 0 else None
        
        self.evtol_manager = TrafficEVTOLManager(
            config, device, self.traffic_prim_path
        ) if hasattr(config, 'num_evtols') and config.num_evtols > 0 else None
        
        # Simulation state
        self.is_initialized = False
        self.step_count = 0
        
        num_evtols = config.num_evtols if hasattr(config, 'num_evtols') else 0
        logging.info(f"TrafficSimulator initialized with {config.num_drones} drones and {num_evtols} evtols")
        if config.num_drones > 0:
            logging.info("  - Drone manager created")
        if num_evtols > 0:
            logging.info("  - EVTOL manager created")
    

    def create_traffic_prim(self):
        """Create traffic primitives in the USD stage."""
        # Create drones using the manager
        if self.drone_manager is not None:
            self.drone_manager.create_drones()
        
        # Create eVTOLs
        if self.evtol_manager is not None:
            self.evtol_manager.create_evtols()
        
        logging.info("Traffic primitives creation complete")

    def initialize(self):
        """Initialize the traffic simulation system."""
        if self.is_initialized:
            return
        
        # Initialize drone components if present
        if self.drone_manager is not None:
            self.drone_manager.initialize()
            self.drone_manager.set_initial_targets()
        
        # Initialize eVTOL components
        if self.evtol_manager is not None:
            self.evtol_manager.initialize()
        
        self.is_initialized = True
        logging.info("TrafficSimulator initialization complete")
    

    
    def step(self, dt: float = 0.02):
        """Execute one simulation step."""
        if not self.is_initialized:
            self.initialize()

        # Update eVTOLs
        if self.evtol_manager is not None:
            self.evtol_manager.step(dt)
            if self.drone_manager is not None:
                self.drone_manager.evtol_states = self.evtol_manager.state
        # Update drones if present

        if self.drone_manager is not None:
            self.drone_manager.step(dt)
        

        
        self.step_count += 1
    
    def get_aircraft_states(self) -> Dict[str, torch.Tensor]:
        """Get states of all traffic aircraft."""
        states = {}
        
        # Get drone states
        if self.drone_manager is not None:
            drone_states = self.drone_manager.get_states()  # Shape [1, N, 13]
            for i in range(self.config.num_drones):
                aircraft_name = f"traffic_drone_{i}"
                states[aircraft_name] = drone_states[0, i]  # Extract individual state
        
        # Get eVTOL states
        if self.evtol_manager is not None:
            evtol_states = self.evtol_manager.get_states()  # Shape [1, N, 13]
            num_evtols = evtol_states.shape[1]
            for i in range(num_evtols):
                aircraft_name = f"traffic_evtol_{i}"
                states[aircraft_name] = evtol_states[0, i]  # Extract individual state
        
        return states
    
    def get_aircraft_positions(self) -> torch.Tensor:
        """Get positions of all traffic aircraft as a tensor."""
        positions = []
        
        # Get drone positions
        if self.drone_manager is not None:
            drone_positions = self.drone_manager.get_positions()  # Shape [1, N, 3]
            positions.append(drone_positions.squeeze(0))  # Convert to [N, 3]
        
        # Get eVTOL positions
        if self.evtol_manager is not None:
            evtol_positions = self.evtol_manager.get_positions()  # Shape [1, N, 3]
            positions.append(evtol_positions.squeeze(0))  # Convert to [N, 3]
        
        if positions:
            return torch.cat(positions, dim=0)  # Concatenate all aircraft positions
        else:
            return torch.empty(0, 3, device=self.device)
    
    def get_aircraft_velocities(self) -> torch.Tensor:
        """Get velocities of all traffic aircraft as a tensor."""
        velocities = []
        
        # Get drone velocities
        if self.drone_manager is not None:
            drone_velocities = self.drone_manager.get_velocities()  # Shape [1, N, 3]
            velocities.append(drone_velocities.squeeze(0))  # Convert to [N, 3]
        
        # Get eVTOL velocities
        if self.evtol_manager is not None:
            evtol_velocities = self.evtol_manager.get_velocities()  # Shape [1, N, 3]
            velocities.append(evtol_velocities.squeeze(0))  # Convert to [N, 3]
        
        if velocities:
            return torch.cat(velocities, dim=0)  # Concatenate all aircraft velocities
        else:
            return torch.empty(0, 3, device=self.device)
    
    def check_collision(self, external_positions: torch.Tensor, safety_radius: float = 2.0) -> torch.Tensor:
        """
        Check for potential collisions between external aircraft and traffic aircraft.
        
        Args:
            external_positions: Tensor of shape (N, 3) containing positions of external aircraft
            safety_radius: Safety distance threshold
            
        Returns:
            Tensor of shape (N,) containing boolean values indicating collision risk
        """
        traffic_positions = self.get_aircraft_positions()
        
        if traffic_positions.shape[0] == 0:
            return torch.zeros(external_positions.shape[0], dtype=torch.bool, device=self.device)
        
        # Calculate distances between all external and traffic aircraft
        # external_positions: (N, 3), traffic_positions: (M, 3)
        # distances: (N, M)
        distances = torch.cdist(external_positions, traffic_positions)
        
        # Check if any distance is below safety threshold
        collision_risk = (distances < safety_radius).any(dim=1)
        
        return collision_risk
    
    def get_aircraft_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all traffic aircraft."""
        total_aircraft = 0
        num_evtols = 0
        
        if self.drone_manager is not None:
            total_aircraft += self.config.num_drones
        
        if self.evtol_manager is not None:
            num_evtols = self.config.num_evtols if hasattr(self.config, 'num_evtols') else 0
            total_aircraft += num_evtols
        
        info = {
            "total_aircraft": total_aircraft,
            "num_drones": self.config.num_drones,
            "num_evtols": num_evtols,
            "positions": self.get_aircraft_positions(),
            "velocities": self.get_aircraft_velocities(),
            "flight_height": self.config.flight_height,
            "area_bounds": self.config.area_bounds,
            "step_count": self.step_count
        }
        return info
    
    def reset(self):
        """Reset the traffic simulation."""
        if self.drone_manager is not None:
            self.drone_manager.reset()
        
        # Reset eVTOLs
        if self.evtol_manager is not None:
            self.evtol_manager.reset()
        
        self.step_count = 0
        logging.info("TrafficSimulator reset complete")
