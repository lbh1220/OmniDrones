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

from sympy import Q
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from isaac_lab_envs.traffic.traffic_drone_manager import TrafficDroneManager
from isaac_lab_envs.traffic.traffic_evtol_manager import TrafficEVTOLManager
from isaac_lab_envs.traffic.cfg.config import TrafficCfg

from dataclasses import field



class TrafficSimulator:
    """
    Main traffic simulation class that manages traffic aircraft.
    
    This class is designed to be used as a plugin in Isaac Lab RL environments,
    providing dynamic background traffic that doesn't interfere with environment cloning.
    
    Architecture follows the pattern from AirSim traffic simulator with dedicated
    manager classes for different aircraft types.
    """
    
    def __init__(self, config: TrafficCfg, device: str = "cuda"):
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

        
        # Initialize eVTOL components
        if self.evtol_manager is not None:
            self.evtol_manager.initialize()
        
        self.is_initialized = True
        logging.info("TrafficSimulator initialization complete")
    
    def update_grid_map(self, no_extended_grid: torch.Tensor, grid_size: float, bounds: tuple[float, float, float, float]):
        """Propagate occupancy map and bounds to managers and their planners."""
        if self.drone_manager is not None:
            self.drone_manager.update_grid_map(no_extended_grid, grid_size, bounds)
        if self.evtol_manager is not None:
            self.evtol_manager.update_grid_map(no_extended_grid, grid_size, bounds)

    def _pre_physics_step(self, dt: float=0.16):
        '''
        evtol manager需要按照时间来更新一大步，移动到新的位置，因为不含动力学，没必要精细控制
        drone manager需要根据evtol的新state来更新速度command，但是不计算底层的rotor command
        '''
        if self.evtol_manager is not None:
            self.evtol_manager.step(dt)
            if self.drone_manager is not None:
                self.drone_manager.evtol_states = self.evtol_manager.state
        if self.drone_manager is not None:
            self.drone_manager._pre_physics_step(dt)

    def _post_physics_step(self):
        # 更新state的内容，用于提供observations
        if self.drone_manager is not None:
            self.drone_manager._post_physics_step()

        self.step_count += 1

        if self.step_count % self.config.reset_interval == 0:
            self.reset()

    def _apply_actions(self):
        # 根据新的velocity,高频计算rotor command
        if self.drone_manager is not None:
            self.drone_manager._apply_actions()

    def step(self, dt: float = 0.02):
        """
        Execute one simulation step.
        this one is deprecated
        """
        # if not self.is_initialized:
        #     self.initialize()

        # # Update eVTOLs
        # if self.evtol_manager is not None:
        #     self.evtol_manager.step(dt)
        #     if self.drone_manager is not None:
        #         self.drone_manager.evtol_states = self.evtol_manager.state
        # # Update drones if present

        # if self.drone_manager is not None:
        #     self.drone_manager.step(dt)
        
        # self.step_count += 1

        self._pre_physics_step(dt)
        self._apply_actions()
        self._post_physics_step()
    
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
    
    def get_aircraft_positions(self, 
                                activate_drones_num: int = None, 
                                activate_evtols_num: int = None) -> torch.Tensor:
        """Get positions of all traffic aircraft as a tensor.
        Args:
            activate_drones_num: 激活的无人机数量, for curriculum learning
            activate_evtols_num: 激活的eVTOL数量
        """
        positions = []
        
        # Get drone positions
        if self.drone_manager is not None:
            drone_positions = self.drone_manager.get_positions()  # Shape [1, N, 3]
            if activate_drones_num is not None:
                if activate_drones_num >= 0 and activate_drones_num < drone_positions.shape[1]:
                    drone_positions = drone_positions[:, :activate_drones_num]
            positions.append(drone_positions.squeeze(0))  # Convert to [N, 3]
        
        # Get eVTOL positions
        if self.evtol_manager is not None:
            evtol_positions = self.evtol_manager.get_positions()  # Shape [1, N, 3]
            if activate_evtols_num is not None:
                if activate_evtols_num >= 0 and activate_evtols_num < evtol_positions.shape[1]:
                    evtol_positions = evtol_positions[:, :activate_evtols_num]
            positions.append(evtol_positions.squeeze(0))  # Convert to [N, 3]
        
        if positions:
            return torch.cat(positions, dim=0)  # Concatenate all aircraft positions
        else:
            return torch.empty(0, 3, device=self.device)
    def get_aircraft_safety_radius(self, 
                                   activate_drones_num: int = None, 
                                   activate_evtols_num: int = None) -> torch.Tensor:
        """Get safety radius of all traffic aircraft."""
        safety_radius = []
        if self.drone_manager is not None:
            drone_safety_radius = self.drone_manager.get_safety_radius()
            if activate_drones_num is not None:
                if activate_drones_num >= 0 and activate_drones_num < drone_safety_radius.shape[0]:
                    drone_safety_radius = drone_safety_radius[:activate_drones_num]
            safety_radius.append(drone_safety_radius)
        if self.evtol_manager is not None:
            evtol_safety_radius = self.evtol_manager.get_safety_radius()
            if activate_evtols_num is not None:
                if activate_evtols_num >= 0 and activate_evtols_num < evtol_safety_radius.shape[0]:
                    evtol_safety_radius = evtol_safety_radius[:activate_evtols_num]
            safety_radius.append(evtol_safety_radius)

        if len(safety_radius) == 0:
            return torch.empty(0, device=self.device)
        return torch.cat(safety_radius, dim=0)
    
    def get_aircraft_types(self, 
                           activate_drones_num: int = None, 
                           activate_evtols_num: int = None) -> torch.Tensor:
        """
        Get types of all traffic aircraft.
        drone: 1
        evtol: 2
        (dummy padding will be handled downstream as 0 if needed)
        """
        types = []
        if self.drone_manager is not None:
            drone_types = torch.ones(self.config.num_drones, device=self.device, dtype=torch.long)
            if activate_drones_num is not None:
                if activate_drones_num >= 0 and activate_drones_num < drone_types.shape[0]:
                    drone_types = drone_types[:activate_drones_num]
            types.append(drone_types)
        if self.evtol_manager is not None:
            evtol_types = torch.full((self.config.num_evtols,), 2, device=self.device, dtype=torch.long)
            if activate_evtols_num is not None:
                if activate_evtols_num >= 0 and activate_evtols_num < evtol_types.shape[0]:
                    evtol_types = evtol_types[:activate_evtols_num]
            types.append(evtol_types)
        if len(types) == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return torch.cat(types, dim=0).to(dtype=torch.long)

    def get_aircraft_velocities(self, 
                                activate_drones_num: int = None, 
                                activate_evtols_num: int = None) -> torch.Tensor:
        """Get velocities of all traffic aircraft as a tensor."""
        velocities = []
        
        # Get drone velocities
        if self.drone_manager is not None:
            drone_velocities = self.drone_manager.get_velocities()  # Shape [1, N, 3]
            if activate_drones_num is not None:
                if activate_drones_num >= 0 and activate_drones_num < drone_velocities.shape[1]:
                    drone_velocities = drone_velocities[:, :activate_drones_num]
            velocities.append(drone_velocities.squeeze(0))  # Convert to [N, 3]
        
        # Get eVTOL velocities
        if self.evtol_manager is not None:
            evtol_velocities = self.evtol_manager.get_velocities()  # Shape [1, N, 3]
            if activate_evtols_num is not None:
                if activate_evtols_num >= 0 and activate_evtols_num < evtol_velocities.shape[1]:
                    evtol_velocities = evtol_velocities[:, :activate_evtols_num]
            velocities.append(evtol_velocities.squeeze(0))  # Convert to [N, 3]
        
        if velocities:
            return torch.cat(velocities, dim=0)  # Concatenate all aircraft velocities
        else:
            return torch.empty(0, 3, device=self.device)
    
    def check_collision(
        self, 
        external_positions: torch.Tensor, # 形状: (env_num, m, 3) 或 (m, 3)
        external_safety_radii: torch.Tensor, # 形状: (env_num, m) 或 (m)
        activate_drones_num: int = None,
        activate_evtols_num: int = None,
    ) -> torch.Tensor:
        """
        检查外部无人机与交通无人机之间是否存在潜在碰撞。
        此函数可以处理2D (m, ...) 或 3D (env_num, m, ...) 的输入。
        """
        traffic_positions = self.get_aircraft_positions(activate_drones_num, activate_evtols_num)
        
        if traffic_positions.shape[0] == 0:
            return torch.zeros_like(external_safety_radii, dtype=torch.bool)
        
        # --- 新增的保障层：检查输入维度 ---
        # 记录原始形状，以便最后恢复
        original_shape = external_safety_radii.shape
        
        # 如果输入是2D的 (m, 3)，我们给它增加一个批处理维度，变成 (1, m, 3)
        if external_positions.ndim == 2:
            external_positions = external_positions.unsqueeze(0)
            external_safety_radii = external_safety_radii.unsqueeze(0)
        # ------------------------------------

        # 1. 维度重塑，便于批处理计算
        # 将输入的 env_num * m 架无人机展平为一个维度
        env_num, m, _ = external_positions.shape
        flat_external_positions = external_positions.view(-1, 3)     # 形状变为: (env_num * m, 3)
        flat_external_radii = external_safety_radii.view(-1)         # 形状变为: (env_num * m)

        # 2. 计算距离矩阵
        # 计算每一架外部无人机到每一架交通无人机的距离
        # distances 形状: (env_num * m, N)
        distances = torch.cdist(flat_external_positions, traffic_positions)

        # 3. 计算阈值矩阵 (核心改动)
        # 我们需要一个和 distances 形状相同的阈值矩阵，
        # 其中每个元素 (i, j) 的值是第 i 架外部无人机和第 j 架交通无人机的安全半径之和。
        traffic_radii = self.get_aircraft_safety_radius(activate_drones_num, activate_evtols_num) # 形状: (N)

        # 利用广播机制：(env_num * m, 1) + (N,) -> (env_num * m, N)
        # unsqueeze(-1) 将 flat_external_radii 变为列向量
        thresholds = flat_external_radii.unsqueeze(-1) + traffic_radii

        # 4. 执行碰撞判断
        # 逐元素比较距离是否小于对应的阈值
        collision_matrix = distances < thresholds # 形状: (env_num * m, N)
        
        # 检查每架外部无人机是否与 *任何* 一架交通无人机发生了碰撞
        collision_risk = collision_matrix.any(dim=1) # 形状: (env_num * m)

        # 5. 恢复原始形状并返回
        # collision_risk 的形状是 (env_num * m)，我们将其恢复为输入的原始批处理形状
        return collision_risk.view(original_shape)
    
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

    def predict_future_positions(self, predict_steps: int, pred_timestep: float) -> torch.Tensor:
        """
        根据当前位置和速度，以匀速模型预测未来多个时间戳的位置。

        Args:
            current_positions: 当前飞机的位置张量，形状为 [N, 3]。
            current_velocities: 当前飞机的速度张量，形状为 [N, 3]。
            predict_steps: 需要预测的未来时间戳的数量。
            pred_timestep: 每个预测时间戳之间的时间间隔（秒）。

        Returns:
            一个形状为 [predict_steps + 1, N, 3] 的张量，
            包含了从当前时刻 (t=0) 到未来 predict_steps 个时刻的所有位置。
        """
        current_positions = self.get_aircraft_positions()
        current_velocities = self.get_aircraft_velocities()
        # 1. 创建时间向量
        #    生成一个从 0 到 predict_steps 的序列，代表时间戳的倍数。
        #    形状: [predict_steps + 1]
        time_multipliers = torch.arange(
            0, predict_steps + 1, 
            device=self.device, 
            dtype=torch.float32
        )

        # 2. 计算每个时间戳的实际时间
        #    形状: [predict_steps + 1]
        future_times = time_multipliers * pred_timestep

        # 3. 计算位移 (Displacement)
        #    利用广播机制，用速度乘以时间向量。
        #    - future_times.view(-1, 1, 1) 的形状变为 [predict_steps + 1, 1, 1]
        #    - current_velocities 的形状是 [N, 3]
        #    广播后，相当于用每个时间点乘以每架飞机的速度
        #    displacement 的形状变为 [predict_steps + 1, N, 3]
        displacement = future_times.view(-1, 1, 1) * current_velocities

        # 4. 计算最终位置
        #    同样利用广播机制，将初始位置加到每一个时间点的位移上。
        #    - current_positions 的形状是 [N, 3]
        #    - displacement 的形状是 [predict_steps + 1, N, 3]
        #    广播后，相当于将初始位置加到每一个时间戳的预测位置上
        #    predicted_positions 的形状变为 [predict_steps + 1, N, 3]
        predicted_positions = current_positions + displacement
        predicted_positions = predicted_positions.permute(1, 0, 2) # [N, predict_steps+1, 3]

        return predicted_positions
    
    def update_traffic_for_env(self, state, drones_num: int = None, evtols_num: int = None):
        """Update the traffic for the environment."""
        if hasattr(state, 'traffic'):
            traffic_positions = self.get_aircraft_positions(activate_drones_num=drones_num, activate_evtols_num=evtols_num)
            traffic_velocities = self.get_aircraft_velocities(activate_drones_num=drones_num, activate_evtols_num=evtols_num)
            traffic_types = self.get_aircraft_types(activate_drones_num=drones_num, activate_evtols_num=evtols_num)
            traffic_safety_radius = self.get_aircraft_safety_radius(activate_drones_num=drones_num, activate_evtols_num=evtols_num)
            state.traffic.traffic_positions = traffic_positions
            state.traffic.traffic_velocities = traffic_velocities
            state.traffic.traffic_types = traffic_types
            state.traffic.traffic_safety_radius = traffic_safety_radius
