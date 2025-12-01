
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
from isaac_lab_envs.traffic.utils.state import TrafficState
from isaac_lab_envs.traffic.utils.policy import ORCA
from isaac_lab_envs.traffic.utils.generator import DroneTargetGenerator
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner

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
        
        # Navigation parameters
        self.max_speed = config.drone.max_speed
        self.min_speed = config.drone.min_speed
        self.v_pref = config.drone.v_pref
        self.arrival_threshold = config.drone.arrival_threshold
        self.safety_radius = config.drone.safety_radius
        self.lookahead_distance = getattr(config.drone, "lookahead_distance", 10.0)
        self.safe_indices = None
        
        # Target generator - managed internally
        self.target_generator = DroneTargetGenerator(config, device)
        
        # Logging
        self.logger = logging.getLogger(__name__)

        # policy for collision avoidance
        self.policy = None
        self.evtol_states = None

        self.global_path_planner: GlobalPathPlanner | None = None
        # cached safe cell indices (grid coordinates [iy, ix]) for random safe sampling
        self.safe_indices: torch.Tensor | None = None
    
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
        
        # Generate initial positions for all drones (spawn off-stage then move on reset)
        initial_positions = []
        prim_paths = []
        init_2d = [1.5*self.config.area_bounds.xmin, 1.5*self.config.area_bounds.ymin]
        for i in range(self.num_drones):
            init_height = -20 - 5*i
            initial_positions.append((init_2d[0], init_2d[1], init_height))
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

        self.reset_drones()

        self.is_initialized = True
        self.logger.info(f"Initialized {self.num_drones} traffic drones")

        if getattr(self.config, 'orca', None) is not None and self.config.orca.enable:
            self.policy = ORCA(self.config)

    def update_grid_map(self, no_extended_grid: torch.Tensor, grid_size: float, bounds: tuple[float, float, float, float]):
        """Update internal occupancy map and optionally the global path planner."""
        from isaac_lab_envs.utils.map_utils import extend_occupancy_map, get_convex_hulls_from_grid
        # mirror into state for downstream checks
        self.state.occupancy_grid = no_extended_grid
        self.state.extended_occupancy_grid = extend_occupancy_map(no_extended_grid, self.safety_radius, grid_size)
        self.state.grid_size = float(grid_size)
        self.state.grid_bounds = bounds
        # invalidate cached safe indices
        self.safe_indices = None
        # update target generator and planner
        planner_cfg = getattr(self.config.drone, "global_path_planner", None)

        if planner_cfg is not None:
            if self.global_path_planner is None:
                self.global_path_planner = GlobalPathPlanner(planner_cfg)
            self.global_path_planner.update_grid_map(self.state.extended_occupancy_grid, self.state.grid_size, bounds)

        self.target_generator.update_grid_map(self.state.extended_occupancy_grid, self.state.grid_size, bounds, self.global_path_planner)


        if self.policy is not None:
            from isaac_lab_envs.utils.map_utils import get_convex_hulls_from_grid
            # 这里需要使用no_extended_grid, 因为ORCA算法本身有safety_space的考虑，如果用extended_grid, 会让行为变得非常保守
            hulls = get_convex_hulls_from_grid(no_extended_grid, bounds, grid_size)
            from isaac_lab_envs.utils.map_utils import plot_convex_hulls
            plot_convex_hulls(hulls, "static_obstacles.png")
            self.policy.set_static_obstacles(hulls)

    def update_global_path(self, drones_ids: torch.Tensor | None = None):
        """Compute and write waypoint paths from current starts to targets using the global planner.
        Fallback to straight line if planner unavailable or path not found.
        """
        # TODO：如果global path过长超过了max_waypoints，需要进行截断，并将截断后的last waypoint作为target position
        if self.state.target_positions.numel() == 0:
            return
        if drones_ids is None:
            drones_ids = torch.arange(self.num_drones, device=self.device)
        if drones_ids.numel() == 0:
            return
        max_wps = self.state.max_waypoints
        waypoints = self.state.waypoints
        waypoint_lengths = self.state.waypoint_lengths
        positions = self.state.start_positions
        targets = self.state.target_positions
        altitude = float(self.config.flight_height)
        use_planner = (self.global_path_planner is not None and self.global_path_planner.grid_map_np is not None)
        for i in drones_ids.tolist():
            s = positions[i, :2]
            g = targets[i, :2]
            pts_xy = []
            if use_planner:
                pts_xy = self.global_path_planner.plan_path((float(s[0].item()), float(s[1].item())), (float(g[0].item()), float(g[1].item())))
            if not pts_xy or len(pts_xy) < 2:
                pts_xy = [
                    (float(s[0].item()), float(s[1].item())),
                    (float(g[0].item()), float(g[1].item())),
                ]
            n = min(len(pts_xy), max_wps)
            if n > 0:
                pts = torch.tensor(pts_xy[:n], dtype=waypoints.dtype, device=self.device)
                waypoints[i, :n, 0:2] = pts[:, :2]
                waypoints[i, :n, 2] = altitude
                # set speed column
                speed = self.state.v_pref[i].clamp(min=self.state.min_speed[i], max=self.state.max_speed[i]).item() if self.state.v_pref.numel() else self.v_pref
                waypoints[i, :n, 3] = float(speed)
            waypoint_lengths[i] = n
        # from isaac_lab_envs.utils.map_utils import visualize_paths_on_grid
        # visualize_paths_on_grid(self.state.extended_occupancy_grid, waypoints, waypoint_lengths, self.state.grid_bounds, self.state.grid_size, "traffic_planned_paths.png")


    def set_targets_for_drones(self, drones_ids: torch.Tensor | None = None):
        """Generate and assign targets for specified drones; update start_positions and global path."""
        if drones_ids is None:
            drones_ids = torch.arange(self.num_drones, device=self.device)
        if drones_ids.numel() == 0:
            return
        k = drones_ids.numel()
        new_targets = self.target_generator.generate_targets(k)
        self.state.target_positions[drones_ids, :] = new_targets
        # 在外部更新start_positions
        # cur_pos = self.drone.pos.squeeze(0)
        # self.state.start_positions[drones_ids, :] = cur_pos[drones_ids, :].clone()
        # plan path if enabled
        self.update_global_path(drones_ids)

        

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

    def _generate_random_position(self, count: int = 1) -> torch.Tensor:
        """
        Generate one or many random positions.
        Prefer sampling safe cells from extended occupancy grid when available.
        Args:
            count: number of positions to sample
        Returns:
            Tensor of shape [count, 3]
        """
        grid = getattr(self.state, "extended_occupancy_grid", None)
        bounds = getattr(self.state, "grid_bounds", None)
        gs = getattr(self.state, "grid_size", None)
        if grid is not None and bounds is not None and gs is not None:
            if self.safe_indices is None:
                # compute safe cells (False in occupancy)
                safe_map = (~grid).to(torch.bool)
                self.safe_indices = torch.nonzero(safe_map, as_tuple=False)
            if self.safe_indices.numel() > 0:
                pick = torch.randint(0, self.safe_indices.shape[0], (int(count),), device=self.device)
                sel = self.safe_indices[pick]
                iy = sel[:, 0].float()
                ix = sel[:, 1].float()
                xmin, xmax, ymin, ymax = map(float, bounds)
                x = xmin + (ix + 0.5) * float(gs)
                y = ymin + (iy + 0.5) * float(gs)
                z = torch.full_like(x, float(self.config.flight_height))
                return torch.stack([x, y, z], dim=-1)
        # fallback: uniform area sampling
        ab = self.config.area_bounds
        x_range = ab.xmax - ab.xmin
        y_range = ab.ymax - ab.ymin
        x = torch.rand(int(count), device=self.device) * x_range + ab.xmin
        y = torch.rand(int(count), device=self.device) * y_range + ab.ymin
        z = torch.full((int(count),), float(self.config.flight_height), device=self.device)
        return torch.stack([x, y, z], dim=-1)                

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
        


        self._update_state_manager()
        directions = self.state.target_positions - self.state.positions # shape [N, 3]
        distances = torch.norm(directions, dim=-1)  # shape [N]
        
        # Check which drones have arrived - shape [N]
        arrived_drones = distances < self.arrival_threshold
        # Check which drones need new targets and generate them
        if arrived_drones.any():
            # 获取需要新目标的无人机索引
            drone_indices = torch.where(arrived_drones)[0]  # 返回需要更新的无人机索引
            num_arrived = len(drone_indices)
            self.state.start_positions[drone_indices] = self.state.positions[drone_indices].clone()
            if num_arrived > 0:
                # 批量生成并设置目标（含路径）
                self.set_targets_for_drones(drone_indices)

        # 使用 traffic state 的矢量化导航更新局部目标
        self.state.update_navigation_state_vectorized(self.lookahead_distance)
        # 基于 local goal 生成速度指令
        pos = self.state.positions
        goals = self.state.local_goals
        v_pref = self.state.v_pref if self.state.v_pref.numel() else torch.full((self.num_drones,), self.v_pref, device=self.device)
        vel_cmd = torch.zeros_like(pos)
        d = goals - pos
        dist = torch.norm(d, dim=-1)
        valid = dist > 1e-6
        if torch.any(valid):
            dir3 = d[valid] / dist[valid].unsqueeze(-1)
            # slowdown near final target
            final_d = torch.norm(self.state.target_positions[valid] - pos[valid], dim=-1)
            base_speed = v_pref[valid]
            speeds = torch.clamp(torch.minimum(base_speed, final_d * 0.5), min=0.1)
            vel_cmd[valid] = dir3 * speeds.unsqueeze(-1)
        self.state.velocity_commands = vel_cmd
        
        # Get root states for all drones - shape [1, N, 13]
        # root_states = self.drone.get_state(env_frame=False)
        
        # Calculate target velocities and yaws for Lee controller
        target_velocities = self.state.velocity_commands.unsqueeze(0)  # shape [1, N, 3]
        target_vel_xy = target_velocities[:, :, :2]
        

        try:
            if self.policy is not None:
                self.policy.predict(self.state, self.evtol_states, dt)
        except Exception as e:
            # 如果失败了就用原来的速度指令
            print(f"[ERROR][traffic]Error predicting in ORCA: {e}")
        # this change the state.velocity__commands

    def _apply_actions(self):
        drone_state = self.drone.get_state(env_frame=False)[..., :13]
        if torch.isnan(drone_state).any():
            logging.warning(f"TrafficDroneManager: drone_state is nan: {drone_state}")
            return
        target_vel_xy = self.state.velocity_commands.unsqueeze(0)
        target_vel_xy = target_vel_xy[:, :, :2]
        # target_yaws = torch.zeros(1, self.num_drones, 1, device=self.device)
        target_yaws = None
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
        need_reset = self._update_state_manager()
        if need_reset: # 有nan值，直接reset整个drone manager
            self.reset()
            return
        collided = self.detect_collision()
        # 对发生碰撞的无人机，重新生成目标并规划路径，同时暂时将速度指令清零以稳定
        if torch.any(collided):
            ids = torch.where(collided)[0]
            collided_num = ids.numel()
            new_positions = self._generate_random_position(collided_num)
            self.state.positions[ids] = new_positions
            self.state.start_positions[ids] = new_positions
            # 这个重置其实很危险，因为他重置了非碰撞飞机的位置，但是暂时没有合适的API重置部分飞机的位置
            # self.drone.set_world_poses(self.state.positions.unsqueeze(0), self.state.rotations.unsqueeze(0))
            self.reset_positions(positions=self.state.positions, rotations=self.state.rotations)
            self.reset_drones(ids)
            self.set_targets_for_drones(ids)
            self.state.velocity_commands[ids] = 0.0
            self._update_state_manager()

    def detect_collision(self):
        # 碰撞检测：与占据图、与EVTOL、与其他drones
        collided = torch.zeros(self.num_drones, dtype=torch.bool, device=self.device)
        # occupancy map
        if self.state.extended_occupancy_grid is not None:
            safe = self.state.are_positions_safe(self.state.positions)
            collided |= ~safe
            num_collided = collided.sum()
            # if num_collided > 0:
            #     self.logger.warning(f"TrafficDroneManager: detect collision with occupancy map: {num_collided}")
        # EVTOL collisions
        if self.evtol_states is not None and self.evtol_states.positions.numel() > 0:
            dmat = torch.cdist(self.state.positions, self.evtol_states.positions)
            thr = self.state.safety_radius.unsqueeze(1) + self.evtol_states.safety_radius.unsqueeze(0)
            collided |= (dmat < thr).any(dim=1)
            num_collided = collided.sum()
            # if num_collided > 0:
            #     self.logger.warning(f"TrafficDroneManager: detect collision with EVTOL: {num_collided}")
        # Drone-drone collisions
        if self.num_drones > 1:
            dmat = torch.cdist(self.state.positions, self.state.positions)
            thr = self.state.safety_radius.unsqueeze(1) + self.state.safety_radius.unsqueeze(0)
            # ignore self
            eye = torch.eye(self.num_drones, dtype=torch.bool, device=self.device)
            pair_collide = (dmat < thr) & (~eye)
            collided |= pair_collide.any(dim=1)
            num_collided = collided.sum()
            # if num_collided > 0:
            #     self.logger.warning(f"TrafficDroneManager: detect collision with drones: {num_collided}")
        # Height abnormality: treat as collision when leaving the allowed band
        # band: [flight_height - 3 * safety_radius, flight_height + 3 * safety_radius]
        z = self.state.positions[:, 2]
        tol = self.config.drone.safety_radius * 4.0
        fh = float(self.config.flight_height)
        height_abnormal = (z < (fh - tol)) | (z > (fh + tol))
        num_collided = height_abnormal.sum()
        if num_collided > 0:
            self.logger.warning(f"TrafficDroneManager: detect height abnormality: {num_collided}")
        collided |= height_abnormal
        # update state
        self.state.has_collided = collided.clone()
        return collided

    def get_positions(self) -> torch.Tensor:
        """获取无人机位置 [1, N, 3]"""
        return self.state.positions.unsqueeze(0)
    
    def get_velocities(self) -> torch.Tensor:
        """获取无人机速度 [1, N, 3]"""
        return self.state.velocities.unsqueeze(0)
    
    def get_rotations(self) -> torch.Tensor:
        """获取无人机姿态 [1, N, 4]"""
        return self.state.rotations.unsqueeze(0)

    def get_states(self) -> torch.Tensor:
        """获取完整状态 [1, N, 13]"""
        # 将位置、姿态、速度、角速度拼接成13维状态
        states = torch.cat([
            self.state.positions,          # [N, 3]
            self.state.rotations,          # [N, 4]
            self.state.velocities,         # [N, 3]
            self.state.angular_velocities  # [N, 3]
        ], dim=-1)  # [N, 13]
        
        return states.unsqueeze(0)  # [1, N, 13]
    
    def get_targets(self) -> torch.Tensor:
        """Get current targets for all drones.
        
        Returns:
            Tensor of shape [1, N, 3] containing target positions
        """
        return self.state.target_positions.unsqueeze(0)  # [N, 3] -> [1, N, 3]
    
    def _update_state_manager(self):
        """更新状态管理器中的运动状态"""
        need_reset = False
        if self.drone is None:
            need_reset = True
            return need_reset
        drone_state = self.drone.get_state(env_frame=False)
        if torch.isnan(drone_state).any():
            logging.warning(f"TrafficDroneManager: drone_state is nan: {drone_state}")
            need_reset = True
            return need_reset
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
        self.state.positions = positions.clone()
        self.drone.set_world_poses(positions.unsqueeze(0), rotations.unsqueeze(0))
        zero_velocities = torch.zeros(positions.shape[0], 6, device=self.device)
        self.drone.set_velocities(zero_velocities.unsqueeze(0))

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


    def reset_drones(self, drones_ids: torch.Tensor = None):
        env_ids = torch.arange(1, device=self.device)
        if drones_ids is None:
            # full env reset (randomization etc.)
            self.drone._reset_idx(env_ids)
            return
        # normalize ids
        if not isinstance(drones_ids, torch.Tensor):
            drones_ids = torch.as_tensor(drones_ids, device=self.device, dtype=torch.long)
        drones_ids = drones_ids.to(self.device).long().flatten()
        if drones_ids.numel() == 0:
            return
        # zero kinematic buffers for selected drones
        self.drone.thrusts[env_ids, drones_ids, :] = 0.0
        self.drone.torques[env_ids, drones_ids, :] = 0.0
        self.drone.vel[env_ids, drones_ids, :] = 0.0
        self.drone.acc[env_ids, drones_ids, :] = 0.0
        if hasattr(self.drone, "forces"):
            self.drone.forces[env_ids, drones_ids, :] = 0.0
        # reinitialize per-drone throttle to hover equilibrium to avoid sudden drop/rise
        init_throttle = self.drone.gravity[env_ids, drones_ids] / self.drone.KF[env_ids, drones_ids].sum(-1, keepdim=True)
        self.drone.throttle[env_ids, drones_ids, :] = self.drone.rotors.f_inv(init_throttle)
        self.drone.throttle_difference[env_ids, drones_ids] = 0.0

        
    def reset(self):
        """Reset all traffic drones to new random positions."""
        if not self.is_initialized:
            self.logger.warning("Drones not initialized yet")
            return
        
        if self.num_drones <= 0:
            return
        # 重置速度参数为配置值
        # 初始化状态管理器, 防止有nan值被继续传递
        names = [f"traffic_drone_{i}" for i in range(self.num_drones)]
        aircraft_types = ["drone"] * self.num_drones
        safety_radius = [self.safety_radius] * self.num_drones
        max_speed = [self.max_speed] * self.num_drones
        min_speed = [self.min_speed] * self.num_drones
        v_pref = [self.v_pref] * self.num_drones
        self.state.initialize_aircraft(names, aircraft_types, safety_radius, max_speed, min_speed, v_pref, self.device)

        self.random_attributes(self.config.drone.random_speed, self.config.drone.random_safety_radius)

        self.target_generator.initialize_targets(self.config.drone.target_num)
        # Generate new positions for all drones (batch)
        positions_batch = self._generate_random_position(self.num_drones)  # [N, 3]
        self.reset_positions(positions_batch)
        self.state.start_positions = positions_batch.clone()
        
        self.reset_drones()


        # Assign new targets using internal target generator
        self.set_targets_for_drones()
        self._update_state_manager()
        
        self.logger.info(f"Reset {self.num_drones} traffic drones")
    

    def get_state_manager(self) -> TrafficState:
        """获取状态管理器"""
        return self.state
    
    def cleanup(self):
        """Cleanup resources."""
        # Reset any internal states if needed
        if hasattr(self.state, 'velocity_commands') and self.state.velocity_commands.numel() > 0:
            self.state.velocity_commands.zero_()

    def get_safety_radius(self) -> torch.Tensor:
        """Get safety radius of all drones."""
        return self.state.safety_radius

    # ---------------- Prediction APIs ----------------
    def predict_future_positions_constant_velocity(self, predict_steps: int, pred_timestep: float) -> torch.Tensor:
        """
        Predict future positions of traffic drones assuming constant velocity.
        Returns:
            Tensor of shape [N, predict_steps + 1, 3]
        """
        # current states
        if self.state.positions.numel() == 0:
            return torch.empty(0, predict_steps + 1, 3, device=self.device)
        positions = self.state.positions  # [N, 3]
        velocities = self.state.velocities if self.state.velocities.numel() > 0 else torch.zeros_like(positions)
        # time vector [S+1]
        times = torch.arange(0, predict_steps + 1, device=self.device, dtype=torch.float32) * float(pred_timestep)
        # broadcast to [S+1, N, 3]
        disp = times.view(-1, 1, 1) * velocities.view(1, -1, 3)
        preds = positions.view(1, -1, 3) + disp  # [S+1, N, 3]
        return preds.permute(1, 0, 2).contiguous()  # [N, S+1, 3]