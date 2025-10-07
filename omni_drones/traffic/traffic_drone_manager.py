
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from omni.isaac.core.utils import prims as prim_utils
from omni.isaac.core.utils import stage as stage_utils

from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers.lee_position_controller import LeePositionController, PlanarSpeedController
from omni_drones.utils.torch import quat_axis, normalize
from omni_drones.traffic.utils.state import TrafficState
from omni_drones.traffic.utils.policy import ORCA
from omni_drones.traffic.utils.generator import DroneTargetGenerator

class TrafficDroneManager:
    """Manager class for batch processing of traffic drones."""
    
    def __init__(self, config, device: str = "cuda", traffic_prim_path: str = "/World/Traffic"):
        self.config = config
        self.device = device
        self.num_drones = config.num_drones
        self.traffic_prim_path = traffic_prim_path
        
        # Create traffic namespace if needed
        if not prim_utils.is_prim_path_valid(self.traffic_prim_path):
            prim_utils.define_prim(self.traffic_prim_path)
        
        # Drone and controller - fully managed by this class
        self.drone: Optional[MultirotorBase] = None
        self.controller: Optional[PlanarSpeedController] = None
        
        # Initialization state
        self.is_created = False
        self.is_initialized = False
        
        # 状态管理器 - 替代之前的分散状态变量
        self.state = TrafficState(device)
        
        # 临时计算变量
        self.is_at_target = torch.zeros(self.num_drones, dtype=torch.bool, device=device)
        self.target_updated_times = torch.zeros(self.num_drones, device=device)
        
        # Navigation parameters
        self.max_speed = config.drone.max_speed
        self.min_speed = config.drone.min_speed
        self.v_pref = config.drone.v_pref
        self.arrival_threshold = config.drone.arrival_threshold
        self.safety_radius = config.drone.safety_radius
        
        # Target generator - managed internally
        self.target_generator = DroneTargetGenerator(config, device)
        
        # Logging
        self.logger = logging.getLogger(__name__)

        # policy for collision avoidance
        self.policy = None
        self.evtol_states = None
    
    def create_drones(self):
        """Create drone primitives in the USD stage."""
        if self.is_created:
            self.logger.warning("Drones already created")
            return
        
        if self.num_drones <= 0:
            self.logger.info("No drones to create")
            self.is_created = True
            return
        
        # Create drone and controller
        self.drone, self.controller = MultirotorBase.make(
            drone_model=self.config.drone.model,
            device=self.device,
            controller="PlanarSpeedController"
        )
        
        # Generate initial positions for all drones
        initial_positions = []
        prim_paths = []
        for i in range(self.num_drones):
            pos = self._generate_random_position()
            initial_positions.append((pos[0].item(), pos[1].item(), pos[2].item()))
            prim_paths.append(f"{self.traffic_prim_path}/traffic_drone_{i}")
        
        # Spawn all drones with the generated positions
        aircraft_prims = self.drone.spawn(
            translations=initial_positions, 
            prim_paths=prim_paths
        )
        
        self.is_created = True
        self.logger.info(f"Created {len(initial_positions)} traffic drone primitives")
    
    def initialize(self):
        """Initialize all traffic drones."""
        if self.is_initialized:
            self.logger.warning("Drones already initialized")
            return
        
        if not self.is_created:
            self.logger.error("Drones not created yet. Call create_drones() first.")
            return
        
        if self.num_drones <= 0:
            self.logger.info("No drones to initialize")
            self.is_initialized = True
            return
        
        # Initialize the drone view (this sets up all cloned drones)
        self.drone.initialize(prim_paths_expr=f"{self.traffic_prim_path}/traffic_drone_*")
        self.target_generator.initialize_targets(self.config.drone.target_num)

        # 初始化状态管理器
        names = [f"traffic_drone_{i}" for i in range(self.num_drones)]
        aircraft_types = ["drone"] * self.num_drones
        safety_radius = [self.safety_radius] * self.num_drones
        max_speed = [self.max_speed] * self.num_drones
        min_speed = [self.min_speed] * self.num_drones
        v_pref = [self.v_pref] * self.num_drones
        self.state.initialize_aircraft(names, aircraft_types, safety_radius, max_speed, min_speed, v_pref, self.device)

        self.random_attributes(self.config.drone.random_speed, self.config.drone.random_safety_radius)

        self.reset_kinematics()

        self.is_initialized = True
        self.logger.info(f"Initialized {self.num_drones} traffic drones")

        if getattr(self.config, 'orca', None) is not None and self.config.orca.enable:
            self.policy = ORCA(self.config)

    def random_attributes(self, random_speed, random_safety_radius):
        """随机生成drone的属性"""
        if random_speed:
            # 为v_pref添加 ±20% 范围内的随机扰动
            speed_perturbation = torch.empty_like(self.state.v_pref).uniform_(-0.2, 0.2)
            self.state.v_pref = self.state.v_pref * (1.0 + speed_perturbation)
            # 确保v_pref在min_speed和max_speed之间
            self.state.v_pref = torch.clamp(self.state.v_pref, min=self.state.min_speed, max=self.state.max_speed)
            
        if random_safety_radius:
            # 为安全半径添加向下的扰动，扰动值为当前半径的0.4
            radius_perturbation = torch.empty_like(self.state.safety_radius).uniform_(-0.2, 0.5)
            self.state.safety_radius = self.state.safety_radius * (1.0 + radius_perturbation)
            self.state.safety_radius = torch.clamp(self.state.safety_radius, min=0.0, max=10.0)

    def _generate_random_position(self) -> torch.Tensor:
        """
        Generate a random position within the area bounds.
        For now, this function is only used to generate initial positions for the drones.
        Targets are set by the target generator in the traffic simulator.
        This fucntion is only used during reset.
        """
        bounds = self.config.area_bounds
        x_range = bounds.xmax - bounds.xmin
        y_range = bounds.ymax - bounds.ymin
        
        x = torch.rand(1, device=self.device) * x_range + bounds.xmin
        y = torch.rand(1, device=self.device) * y_range + bounds.ymin
        z = torch.tensor([self.config.flight_height], device=self.device)
        return torch.cat([x, y, z])
        
    def set_targets(self, targets: torch.Tensor):
        """Set new target positions for all drones.
        
        Args:
            targets: Tensor of shape [N, 3] or [1, N, 3] containing target positions
        """
        if targets.dim() == 3:
            targets = targets.squeeze(0)  # Convert [1, N, 3] to [N, 3]
        
        self.state.target_positions = targets
        self.is_at_target.fill_(False)
        self.target_updated_times.zero_()
        
    def set_target_for_drone(self, drone_idx: int, target: torch.Tensor):
        """Set target position for a specific drone.
        
        Args:
            drone_idx: Index of the drone
            target: Tensor of shape [3] containing target position
        """
        self.state.target_positions[drone_idx] = target
        self.is_at_target[drone_idx] = False
        self.target_updated_times[drone_idx] = 0.0
        

    def _pre_physics_step(self, dt: float=None):
        '''
        为了配合isaac_lab的配置
        在这个pre_physics_step中，更新目标位置，更新目标速度，但是不计算底层控制量
        '''

        if not self.is_initialized:
            self.logger.warning("Drones not initialized yet")
            return
        
        if self.num_drones <= 0:
            return
        
        # Check which drones need new targets and generate them
        # This uses the is_at_target status updated in update_control
        arrived_drones = self.is_at_target
        if arrived_drones.any():
            # 获取需要新目标的无人机索引
            drone_indices = torch.where(arrived_drones)[0]  # 返回需要更新的无人机索引
            num_arrived = len(drone_indices)
            
            if num_arrived > 0:
                # 批量生成新目标
                new_targets = self.target_generator.generate_targets(num_arrived)
                
                # 批量设置目标
                self.state.target_positions[drone_indices, :] = new_targets
                
                # 批量更新状态
                self.is_at_target[drone_indices] = False
                self.target_updated_times[drone_indices] = 0.0

        drone_state = self.drone.get_state(env_frame=False)[..., :13]#[1, N, 13]
        # Get current positions - shape [1, N, 3]
        current_positions = self.drone.pos
        
        # Calculate directions to targets - shape [1, N, 3]
        current_targets = self.state.target_positions.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        directions = current_targets - current_positions
        distances = torch.norm(directions, dim=-1)  # shape [1, N]
        
        # Check which drones have arrived - shape [N]
        arrived_mask = distances.squeeze(0) < self.arrival_threshold
        self.is_at_target = arrived_mask

        # Reset velocity commands
        self.state.velocity_commands.zero_()
        
        # Calculate velocity commands for all drones (keep [1, N, 3] format)
        # Normalize directions (avoid division by zero)
        valid_movement = distances[0] > 1e-6  # shape [N]
        
        if valid_movement.any():
            # Calculate normalized directions for valid movements
            valid_directions = directions[0, valid_movement]  # shape [num_valid, 3]
            valid_distances = distances[0, valid_movement]    # shape [num_valid]
            
            # Normalize directions
            normalized_valid_dirs = valid_directions / valid_distances.unsqueeze(-1)
            
            # Calculate speeds with gradual slowdown
            # Get preferred speeds for valid moving drones
            valid_v_pref = self.state.v_pref[valid_movement]
            speeds = torch.clamp(
                torch.minimum(
                    valid_v_pref,
                    valid_distances * 0.5
                ),
                min=0.1
            )
            
            # Apply speed to directions
            velocity_commands_valid = normalized_valid_dirs * speeds.unsqueeze(-1)
            
            # Update velocity commands for valid movements (maintain [1, N, 3] format)
            self.state.velocity_commands[valid_movement] = velocity_commands_valid
        
        # Get root states for all drones - shape [1, N, 13]
        # root_states = self.drone.get_state(env_frame=False)
        
        # Calculate target velocities and yaws for Lee controller
        target_velocities = self.state.velocity_commands.unsqueeze(0)  # shape [1, N, 3]
        target_vel_xy = target_velocities[:, :, :2]
        


        if self.policy is not None:
            self.policy.predict(self.state, self.evtol_states, dt)
        # this change the state.velocity__commands
    def _apply_actions(self):
        drone_state = self.drone.get_state(env_frame=False)[..., :13]
        target_vel_xy = self.state.velocity_commands.unsqueeze(0)
        target_vel_xy = target_vel_xy[:, :, :2]
        target_yaws = torch.zeros(1, self.num_drones, 1, device=self.device)
        target_height = self.config.flight_height * torch.ones(1, self.num_drones, 1, device=self.device)
        rotor_commands = self.controller.compute(
            root_state=drone_state,  # shape [1, N, 3]
            target_vel_xy=target_vel_xy,  # shape [1, N, 2]
            target_height=target_height,  # shape [1, N, 1]
            target_yaw=target_yaws  # shape [1, N]
        )
        self.drone.apply_action(rotor_commands)
        
        
    def _post_physics_step(self):
        # 更新state的内容，用于提供observations
        self._update_state_manager()

    def update_control(self, dt: float) -> torch.Tensor:
        """Update control for all drones in batch.
        
        Args:
            dt: Time step
            
        Returns:
            Batch rotor commands of shape [N, num_rotors]
        """
        if not self.is_initialized:
            raise RuntimeError("Drones must be initialized before updating control")
        
        if self.num_drones <= 0:
            return torch.empty(0, 4, device=self.device)  # Empty tensor for no drones
        drone_state = self.drone.get_state(env_frame=False)[..., :13]#[1, N, 13]
        # Get current positions - shape [1, N, 3]
        current_positions = self.drone.pos
        
        # Calculate directions to targets - shape [1, N, 3]
        current_targets = self.state.target_positions.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        directions = current_targets - current_positions
        distances = torch.norm(directions, dim=-1)  # shape [1, N]
        
        # Check which drones have arrived - shape [N]
        arrived_mask = distances.squeeze(0) < self.arrival_threshold
        self.is_at_target = arrived_mask

        # Reset velocity commands
        self.state.velocity_commands.zero_()
        
        # Calculate velocity commands for all drones (keep [1, N, 3] format)
        # Normalize directions (avoid division by zero)
        valid_movement = distances[0] > 1e-6  # shape [N]
        
        if valid_movement.any():
            # Calculate normalized directions for valid movements
            valid_directions = directions[0, valid_movement]  # shape [num_valid, 3]
            valid_distances = distances[0, valid_movement]    # shape [num_valid]
            
            # Normalize directions
            normalized_valid_dirs = valid_directions / valid_distances.unsqueeze(-1)
            
            # Calculate speeds with gradual slowdown
            # Get preferred speeds for valid moving drones
            valid_v_pref = self.state.v_pref[valid_movement]
            speeds = torch.clamp(
                torch.minimum(
                    valid_v_pref,
                    valid_distances * 0.5
                ),
                min=0.1
            )
            
            # Apply speed to directions
            velocity_commands_valid = normalized_valid_dirs * speeds.unsqueeze(-1)
            
            # Update velocity commands for valid movements (maintain [1, N, 3] format)
            self.state.velocity_commands[valid_movement] = velocity_commands_valid
        
        # Get root states for all drones - shape [1, N, 13]
        # root_states = self.drone.get_state(env_frame=False)
        
        # Calculate target velocities and yaws for Lee controller
        target_velocities = self.state.velocity_commands.unsqueeze(0)  # shape [1, N, 3]
        target_vel_xy = target_velocities[:, :, :2]
        target_height = self.config.flight_height * torch.ones(1, self.num_drones, 1, device=self.device)

        # Calculate target yaws based on velocity direction (atan2(vy, vx))
        # For xy-plane movement, yaw = atan2(velocity_y, velocity_x)
        # vel_x = target_velocities[0, :, 0]  # shape [N]
        # vel_y = target_velocities[0, :, 1]  # shape [N]
        # target_yaws = torch.atan2(vel_y, vel_x)  # shape [N]

        # # For stationary drones (zero velocity), set yaw to 0
        # zero_velocity_mask = torch.norm(target_velocities[0, :, :2], dim=-1) < 1e-6
        # target_yaws[zero_velocity_mask] = 0.0

        if self.policy is not None:
            self.policy.predict(self.state, self.evtol_states)

        target_yaws = torch.zeros(1, self.num_drones, 1, device=self.device)
        
        rotor_commands = self.controller.compute(
            root_state=drone_state,  # shape [1, N, 3]
            target_vel_xy=target_vel_xy,  # shape [1, N, 2]
            target_height=target_height,  # shape [1, N, 1]
            target_yaw=target_yaws  # shape [1, N]
        )
        
        return rotor_commands  # shape [N, num_rotors]
    
    def step(self, dt: float = 0.02):
        """Execute one step for drone traffic management.
        
        Args:
            dt: Time step for control update
        """
        if not self.is_initialized:
            self.logger.warning("Drones not initialized yet")
            return
        
        if self.num_drones <= 0:
            return
        
        # Check which drones need new targets and generate them
        # This uses the is_at_target status updated in update_control
        arrived_drones = self.is_at_target
        if arrived_drones.any():
            # 获取需要新目标的无人机索引
            drone_indices = torch.where(arrived_drones)[0]  # 返回需要更新的无人机索引
            num_arrived = len(drone_indices)
            
            if num_arrived > 0:
                # 批量生成新目标
                new_targets = self.target_generator.generate_targets(num_arrived)
                
                # 批量设置目标
                self.state.target_positions[drone_indices, :] = new_targets
                
                # 批量更新状态
                self.is_at_target[drone_indices] = False
                self.target_updated_times[drone_indices] = 0.0

        
        # Update control for all drones
        rotor_commands = self.update_control(dt)
        
        # Apply commands to the drone batch
        self.apply_actions(rotor_commands)
        
        # 更新状态管理器中的运动状态
        self._update_state_manager()
    
    def get_positions(self) -> torch.Tensor:
        """Get positions of all drones.
        
        Returns:
            Tensor of shape [1, N, 3] containing drone positions
        """
        if self.drone is None:
            return torch.empty(1, 0, 3, device=self.device)
        return self.drone.pos
    
    def get_velocities(self) -> torch.Tensor:
        """Get velocities of all drones.
        
        Returns:
            Tensor of shape [1, N, 3] containing drone linear velocities
        """
        if self.drone is None:
            return torch.empty(1, 0, 3, device=self.device)
        return self.drone.vel[:, :, :3]  # Get only linear velocities
    
    def get_states(self) -> torch.Tensor:
        """Get full states of all drones.
        
        Returns:
            Tensor of shape [1, N, 13] containing full drone states
        """
        if self.drone is None:
            return torch.empty(1, 0, 13, device=self.device)
        return self.drone.get_state(env_frame=False)
    
    def get_targets(self) -> torch.Tensor:
        """Get current targets for all drones.
        
        Returns:
            Tensor of shape [1, N, 3] containing target positions
        """
        return self.state.target_positions.unsqueeze(0)  # [N, 3] -> [1, N, 3]
    
    def _update_state_manager(self):
        """更新状态管理器中的运动状态"""
        if self.drone is None:
            return
        self.drone.get_state(env_frame=False)
        # 直接更新state中的运动状态
        self.state.positions = self.drone.pos.squeeze(0)  # [1, N, 3] -> [N, 3]
        self.state.velocities = self.drone.vel[:, :, :3].squeeze(0)  # [1, N, 3] -> [N, 3] 
        self.state.rotations = self.drone.rot.squeeze(0)  # [1, N, 4] -> [N, 4]
        self.state.angular_velocities = self.drone.vel[:, :, 3:].squeeze(0)  # [1, N, 3] -> [N, 3]
    
    def reset_positions(self, positions: torch.Tensor, rotations: Optional[torch.Tensor] = None):
        """Reset drone positions.
        
        Args:
            positions: Tensor of shape [N, 3] containing new positions
            rotations: Optional tensor of shape [N, 4] containing new rotations (quaternions)
        """
        if self.drone is None:
            return
            
        if rotations is None:
            rotations = torch.zeros(positions.shape[0], 4, device=self.device)
            rotations[:, 0] = 1.0  # Identity quaternion [w, x, y, z]
        
        self.drone.set_world_poses(positions.unsqueeze(0), rotations.unsqueeze(0))
        
        # Reset states
        env_ids = torch.arange(0, device=self.device)
        self.drone._reset_idx(env_ids)
    
    def apply_actions(self, rotor_commands: torch.Tensor):
        """Apply rotor commands to all drones.
        
        Args:
            rotor_commands: Tensor of shape [N, num_rotors] containing rotor commands
        """
        if self.drone is None:
            return
        
        # Convert [N, num_rotors] to [1, N, num_rotors] for apply_action
        if rotor_commands.dim() == 2:
            rotor_commands = rotor_commands.unsqueeze(0)  # [N, num_rotors] -> [1, N, num_rotors]
        
        self.drone.apply_action(rotor_commands)
    def reset_kinematics(self):
        env_ids = torch.arange(1, device=self.device)
        self.drone._reset_idx(env_ids)
    def reset(self):
        """Reset all traffic drones to new random positions."""
        if not self.is_initialized:
            self.logger.warning("Drones not initialized yet")
            return
        
        if self.num_drones <= 0:
            return
        
        self.target_generator.initialize_targets(self.config.drone.target_num)
        # Generate new positions for all drones
        new_positions = []
        for i in range(self.num_drones):
            new_pos = self._generate_random_position()
            new_positions.append(new_pos)
        
        # Set all positions at once
        if new_positions:
            positions_batch = torch.stack(new_positions)  # Shape [N, 3]
            self.reset_positions(positions_batch)
        
        self.reset_kinematics()
        # 重置速度参数为配置值
        safety_radius = [self.safety_radius] * self.num_drones
        max_speed = [self.max_speed] * self.num_drones
        min_speed = [self.min_speed] * self.num_drones
        v_pref = [self.v_pref] * self.num_drones
        self.state.max_speed = torch.tensor(max_speed, device=self.device)
        self.state.min_speed = torch.tensor(min_speed, device=self.device)
        self.state.v_pref = torch.tensor(v_pref, device=self.device)
        self.state.safety_radius = torch.tensor(safety_radius, device=self.device)
        self.random_attributes(self.config.drone.random_speed, self.config.drone.random_safety_radius)
        # Assign new targets using internal target generator
        self.set_initial_targets()
        
        self.logger.info(f"Reset {self.num_drones} traffic drones")
    
    def set_initial_targets(self):
        """Set initial targets for all drones using internal target generator."""
        if not self.is_initialized:
            self.logger.warning("Drones not initialized yet. Targets will be set when initialized.")
            return
        
        # Generate targets for all drones at once
        targets_batch = self.target_generator.generate_targets(self.num_drones)
        # Set all targets at once using batch processing
        
        # 直接设置state中的目标和起始位置
        self.state.target_positions = targets_batch
        self.state.start_positions = self.drone.pos.squeeze(0)  # [1, N, 3] -> [N, 3]
        
        self.is_at_target.fill_(False)
        self.target_updated_times.zero_()
        
        self.logger.info(f"Set initial targets for {self.num_drones} drones")
    
    def get_state_manager(self) -> TrafficState:
        """获取状态管理器"""
        return self.state
    
    def cleanup(self):
        """Cleanup resources."""
        # Reset any internal states if needed
        self.is_at_target.fill_(False)
        if hasattr(self.state, 'velocity_commands') and self.state.velocity_commands.numel() > 0:
            self.state.velocity_commands.zero_()
        self.target_updated_times.zero_()

    def get_safety_radius(self) -> torch.Tensor:
        """Get safety radius of all drones."""
        return self.state.safety_radius