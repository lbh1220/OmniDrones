from __future__ import annotations
import gymnasium as gym

import numpy as np
from typing import Dict, Any, List
from types import SimpleNamespace
from typing import TYPE_CHECKING

# Optional typing import for standalone usage
try:
    from airsim_nav_env import EnvironmentState  # type: ignore
except Exception:  # pragma: no cover - fallback stub for typing only
    class EnvironmentState:  # type: ignore
        pass

# Ensure ROS dependencies are not required in this standalone module
TRAFFIC_MSGS_AVAILABLE = True

class IsaacLikeObservationProcessor:
    """Isaac-style observation processor for SB3 environment.
    - Inherits ComplexObservationProcessor to keep the same observation dict schema
    - Changes spatial_edges normalization and padding: zero-padding instead of +inf
    - Optional angle-distance encoding; default disabled to match AirSimNavSB3Env shapes
    """

    def __init__(self, config=None, sb3_config=None):
        # Persist config with sensible defaults for standalone usage
        if config is None:
            config = SimpleNamespace(
                observation_radius=1000.0,
                vehicle_name="ego",
                max_aircraft_in_obs=20,
                obs_noise_level=0.0,
                use_global_planner=False,
            )
        self.config = config

        # Additional normalization/encoding options (compatible with sb3_config.obs)
        obs_cfg = getattr(sb3_config, 'obs', None) if sb3_config is not None else None
        # Whether to use angle+distance encoding: (sin(theta), cos(theta), 1/(d+1))
        # Note: AirSimNavSB3Env expects 2*(steps+1) per neighbor by default; keep disabled for shape compatibility
        self.use_angle_distance_obs = getattr(obs_cfg, 'use_angle_distance_obs', True)
        # 观测归一化缩放
        self.observation_norm_scale = getattr(obs_cfg, 'norm_scale', getattr(obs_cfg, 'scale', 1.0))
        self.use_global_planner = getattr(config, 'use_global_planner', False)
        # Sensor range and area scale parameters
        self.sensor_range = self.config.observation_radius
        self.human_num = getattr(config, 'max_aircraft_in_obs', 20)
        # 若未在sim配置circle_radius，则回退到飞行区域估算

        self.circle_radius = getattr(obs_cfg, 'circle_radius', 113.137085)

        # Prediction related defaults (standalone)
        self.predict_steps = getattr(config, 'predict_steps', 4)
        self.pred_timestep = getattr(config, 'pred_timestep', 0.1)
        self.last_human_states = None
        self.state = None

        if self.use_global_planner:
            self.robot_node_dim = 5 + 2 + 2
        else:
            self.robot_node_dim = 5
        if self.use_angle_distance_obs:
            self.robot_node_dim += 1
            if self.use_global_planner:
                self.robot_node_dim += 2
        self.temporal_edges_dim = 2
        self.spatial_point_dim = 3 if self.use_angle_distance_obs else 2
        self.spatial_dim = self.spatial_point_dim * (self.predict_steps + 1) + 1 
    def _encode_relative_xy(self, rel_xy: np.ndarray) -> np.ndarray:
        """Encode relative positions.
        - Default: linear normalization of (dx, dy) by area scale
        - Optional: (sin(theta), cos(theta), 1/(d+1)); disabled by default to keep shape
        """
        if not self.use_angle_distance_obs:
            return rel_xy / (2.0 * self.circle_radius) * self.observation_norm_scale

        dx = rel_xy[..., 0]
        dy = rel_xy[..., 1]
        theta = np.arctan2(dy, dx)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        dist = np.linalg.norm(rel_xy, axis=-1)
        inv_dist = 1.0 / (dist + 1.0)
        return np.stack([sin_theta, cos_theta, inv_dist], axis=-1)

    def _get_robot_node(self, state):
        """Get robot node from state"""
        # Robot state
        robot_pos = state.vehicle.get_position_array()  # [x, y, z]
        robot_vel = state.vehicle.get_velocity_array()  # [vx, vy, vz]
        robot_yaw = np.arctan2(robot_vel[1], robot_vel[0])
        robot_radius = state.vehicle.radius
        robot_v_pref = state.vehicle.max_velocity

        # Goal
        goal_pos = state.navigation.goal_position.to_array()  # [x, y, z]

        # Robot node (Isaac style but shape-compatible with SB3 spaces)
        rel_goal = goal_pos[:2] - robot_pos[:2]  # [dx, dy]
        enc_goal = self._encode_relative_xy(rel_goal)  # [2] (angle mode disabled by default)
        # Assemble as [rel_goal_x, rel_goal_y, radius, v_pref, yaw]
        if self.use_global_planner:
            local_goal = state.navigation.local_goal_position.to_array()
            closest_point = state.navigation.closest_point_on_path.to_array()
            rel_local_goal = local_goal[:2] - closest_point[:2]
            rel_closest_point = closest_point[:2] - robot_pos[:2]
            enc_rel_local_goal = self._encode_relative_xy(rel_local_goal)
            enc_rel_closest_point = self._encode_relative_xy(rel_closest_point)

        robot_node = np.concatenate([enc_goal, [robot_radius, robot_v_pref, robot_yaw]])
        if self.use_global_planner:
            robot_node = np.concatenate([robot_node, enc_rel_local_goal, enc_rel_closest_point])
        return robot_node.reshape(1, -1)
    def _get_temporal_edges(self, state):
        """Get temporal edges from state"""
        robot_vel = state.vehicle.get_velocity_array()  # [vx, vy, vz]
        temporal_edges = np.array([robot_vel[0], robot_vel[1]], dtype=np.float32)
        return temporal_edges.reshape(1, -1)

    def _get_spatial_edges(self, state):

        # Robot state
        robot_pos = state.vehicle.get_position_array()  # [x, y, z]
        robot_vel = state.vehicle.get_velocity_array()  # [vx, vy, vz]
        # Other aircraft states from AirSim traffic
        # Gather aircraft states (already filtered by observation_radius upstream)
        all_states = self._extract_aircraft_states(state)

        # Predict future trajectories (const-vel)
        self._update_human_states(all_states)
        predicted_states = self._calc_human_future_traj(all_states, robot_pos, robot_vel)

        # Spatial edges allocation matching SB3 spaces (Isaac-style direct build + pad)
        human_num = self.human_num
        predict_steps = self.predict_steps
        spatial_point_dim = 3 if getattr(self, 'use_angle_distance_obs', False) else 2
        base_len = int(spatial_point_dim * (predict_steps + 1))
        spatial_len = base_len + 1  # +1 for safety radius per neighbor, per Isaac design

        spatial_edges = np.zeros((human_num, spatial_len), dtype=np.float32)
        visible_masks = np.zeros((human_num,), dtype=np.float32)
        spatial_types = np.zeros((human_num,), dtype=np.int64)

        if predicted_states is not None and len(all_states) > 0:
            # [steps+1, N, 2] -> [N, steps+1, 2]
            predicted_pos = predicted_states[:, :, :2]
            rel_pos = np.transpose(predicted_pos, (1, 0, 2)) - robot_pos[:2]

            if self.use_angle_distance_obs:
                # Encode to (sin, cos, 1/(d+1)) per step
                enc = self._encode_relative_xy(rel_pos)
                flattened = enc.reshape((enc.shape[0], -1))
            else:
                # Linear normalization by circle_radius and observation_norm_scale
                normed = rel_pos / (2.0 * self.circle_radius) * self.observation_norm_scale
                flattened = normed.reshape((normed.shape[0], -1))

            num_aircraft = min(len(all_states), human_num)
            write_len = min(base_len, flattened.shape[1])

            if num_aircraft > 0 and write_len > 0:
                spatial_edges[:num_aircraft, :write_len] = flattened[:num_aircraft, :write_len]

            # Append safety radius per neighbor as last column
            for i in range(num_aircraft):
                spatial_edges[i, base_len] = float(all_states[i].get('radius', 1.0))

            # Visible mask based on current step distance within sensor range
            sensor_range = 1000
            current_dist = np.linalg.norm(rel_pos[:, 0, :], axis=-1) if rel_pos.shape[1] > 0 else np.zeros((num_aircraft,), dtype=float)
            visible = (current_dist <= sensor_range).astype(np.float32)
            visible_masks[:num_aircraft] = visible

            # Spatial types: 1=drone(uav), 2=evtol, 0=dummy
            for i in range(num_aircraft):
                t = all_states[i].get('aircraft_type', 'uav')
                spatial_types[i] = 2 if t == 'evtol' else 1

            # Zero out spatial edges for non-visible rows
            # for i in range(num_aircraft):
            #     if visible_masks[i] < 0.5:
            #         spatial_edges[i, :] = 0.0

            return spatial_edges, visible_masks, spatial_types
    def process_observation(self, state):
        """Build Isaac-Lab style observation using numpy arrays and return under key 'policy'.
        Fields: robot_node, temporal_edges, spatial_edges, visible_masks, spatial_types
        """
        self.state = state

        policy: Dict[str, Any] = {}


        policy['robot_node'] = self._get_robot_node(state)


        policy['temporal_edges'] = self._get_temporal_edges(state)

        spatial_edges, visible_masks, spatial_types = self._get_spatial_edges(state)
        policy['spatial_edges'] = spatial_edges
        policy['visible_masks'] = visible_masks
        policy['spatial_types'] = spatial_types

        # Return nested dict per Isaac design
        return policy

    def generate_policy_obs_dict(self):
        # 计算traffic数量配置
        
        # 1. 创建内层字典 "policy" 的内容
        policy_space_dict = {
            # robot_node维度: 相对目标(2或3) + 安全半径1 + v_pref 1 + yaw 1
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.robot_node_dim), dtype=np.float32),
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.temporal_edges_dim), dtype=np.float32),
            'spatial_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, self.spatial_dim), dtype=np.float32),
            'visible_masks': gym.spaces.Box(low=0.0, high=1.0, shape=(self.human_num,), dtype=np.float32),
            'spatial_types': gym.spaces.Box(low=0, high=2, shape=(self.human_num,), dtype=np.int64),
        }
        return policy_space_dict

    def _calc_human_future_traj(self, all_states, robot_pos, robot_vel):
        """计算人类未来轨迹"""
        if not all_states:
            return np.zeros((self.predict_steps + 1, 0, 4))
        
        human_num = len(all_states)
        self.human_future_traj = np.zeros((self.predict_steps + 1, human_num, 4))
        
        # 初始化当前状态
        for i, aircraft in enumerate(all_states):
            pos = aircraft['position']
            vel = aircraft['velocity']
            self.human_future_traj[0, i] = [pos[0], pos[1], vel[0], vel[1]]
        
        # 使用恒速模型预测
        self.pred_method = 'const_vel'
        if self.pred_method == 'const_vel':
            time_step = self.pred_timestep  # 假设时间步长
            for step in range(1, self.predict_steps + 1):
                self.human_future_traj[step] = self.human_future_traj[0].copy()
                # 更新位置
                self.human_future_traj[step, :, 0] += self.human_future_traj[0, :, 2] * step * time_step
                self.human_future_traj[step, :, 1] += self.human_future_traj[0, :, 3] * step * time_step
        
        return self.human_future_traj

    def _extract_aircraft_states(self, state: EnvironmentState) -> List[Dict[str, Any]]:
        """提取其他飞机状态"""
        aircraft_states = []
        if not TRAFFIC_MSGS_AVAILABLE or not state.raw_aircraft_states:
            return aircraft_states
        
        current_pos = state.vehicle.get_position_array()
        noise_level = self.config.obs_noise_level
        
        distances = []
        for aircraft in state.raw_aircraft_states:
            if aircraft.name == self.config.vehicle_name:
                continue
            
            aircraft_pos = np.array([aircraft.position.x, aircraft.position.y, aircraft.position.z])
            dist = np.linalg.norm(aircraft_pos - current_pos)
            
            if dist <= self.config.observation_radius:
                aircraft_dict = {
                    'aircraft_type': aircraft.aircraft_type,
                    'position': aircraft_pos.tolist(),
                    'velocity': [aircraft.linear_velocity.x, aircraft.linear_velocity.y, aircraft.linear_velocity.z],
                    'radius': aircraft.radius,
                    'name': aircraft.name,
                    'distance': dist
                }
                
                # 添加观测噪声
                if noise_level > 0.0:
                    aircraft_type = getattr(aircraft, 'aircraft_type', 'uav')
                    aircraft_dict = self._add_observation_noise(aircraft_dict, aircraft_type, noise_level, current_pos)
                
                distances.append((aircraft_dict['distance'], aircraft_dict))
        
        # 按距离排序并取最近的几个
        distances.sort(key=lambda x: x[0])
        num_aircraft = min(len(distances), self.config.max_aircraft_in_obs)
        aircraft_states = [distances[i][1] for i in range(num_aircraft)]
        
        return aircraft_states

    def _update_human_states(self, all_states):
        """更新人类状态历史"""
        if self.last_human_states is None:
            self.last_human_states = np.zeros((len(all_states), 5))
        
        # 更新可见飞机的状态
        for i, aircraft in enumerate(all_states):
            if i < len(self.last_human_states):
                pos = aircraft['position']
                vel = aircraft['velocity']
                radius = aircraft.get('radius', 1.0)
                self.last_human_states[i] = [pos[0], pos[1], vel[0], vel[1], radius]