from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple

import torch
import gymnasium as gym

from airsim_nav_env import EnvironmentConfig, EnvironmentState
# Import airsim_traffic_ros messages
try:
    from airsim_traffic_ros.msg import AircraftStateArray, AircraftState
    from airsim_traffic_ros.srv import ResetSimulation, ResetSimulationRequest, GetGridMap
    TRAFFIC_MSGS_AVAILABLE = True
except ImportError:
    TRAFFIC_MSGS_AVAILABLE = False


class BaseObservationProcessor:
    """观测处理基类，便于扩展"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
 
    def process_observation(self, state: EnvironmentState) -> Dict[str, Any]:
        """处理观测数据 - 子类需要实现"""
        raise NotImplementedError
    
    def _extract_self_state(self, state: EnvironmentState) -> Dict[str, Any]:
        """提取自身状态"""
        return {
            'position': state.vehicle.get_position_array().tolist(),
            'velocity': state.vehicle.get_velocity_array().tolist(),
            'yaw': state.vehicle.get_yaw(),
            'radius': state.vehicle.radius,
            'v_pref': state.vehicle.max_velocity,
            'quaternion': state.vehicle.pose.orientation.to_array().tolist() # [x, y, z, w]
        }
    
    def _extract_goal_relative(self, state: EnvironmentState) -> List[float]:
        """提取目标相对位置"""
        goal_relative = [0.0, 0.0, 0.0]
        goal_pos = state.navigation.goal_position
        vehicle_pos = state.vehicle.pose.position
        goal_relative = [
            goal_pos.x - vehicle_pos.x,
            goal_pos.y - vehicle_pos.y,
            goal_pos.z - vehicle_pos.z
        ]
        return goal_relative


class DefaultObservationProcessor(BaseObservationProcessor):
    """默认观测处理器"""
    
    def process_observation(self, state: EnvironmentState) -> Dict[str, Any]:
        """处理观测数据"""
        # 自身状态
        self_state = self._extract_self_state(state)
        
        # 目标相对位置
        goal_relative = self._extract_goal_relative(state)
        
        # 其他飞机状态
        aircraft_states = self._extract_aircraft_states(state)
        
        # 局部控制信息
        if state.navigation.has_local_goal:
            local_control = (
                state.navigation.local_goal_velocity.vx,
                state.navigation.local_goal_velocity.vy,
                state.navigation.local_goal_velocity.vz,
                state.navigation.local_goal_yaw
            )
            local_goal = (
                state.navigation.local_goal_position.x,
                state.navigation.local_goal_position.y,
                state.navigation.local_goal_position.z,
                state.navigation.local_goal_yaw
            )
        else:
            local_control = None
            local_goal = None
        
        return {
            'self_state': self_state,
            'goal_relative': goal_relative,
            'aircraft_states': aircraft_states,
            'local_control': local_control,
            'local_goal': local_goal
        }
    
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
    
    def _add_observation_noise(self, aircraft_dict: Dict, aircraft_type: str, 
                              noise_level: float, current_pos: np.ndarray) -> Dict:
        """为观测添加噪声"""
        # 噪声添加逻辑保持不变，只是参数传递方式调整
        if noise_level <= 0.0:
            return aircraft_dict
        
        px, py, pz = aircraft_dict['position']
        vx, vy, vz = aircraft_dict['velocity']
        radius = aircraft_dict['radius']
        
        if aircraft_type == 'evtol':
            px_noisy, py_noisy = self._add_evtol_position_noise(px, py, vx, vy, radius, noise_level)
            vx_noisy, vy_noisy = self._add_evtol_velocity_noise(vx, vy, radius, noise_level)
        else:
            px_noisy, py_noisy = self._add_uav_position_noise(px, py, radius, noise_level)
            vx_noisy, vy_noisy = self._add_uav_velocity_noise(vx, vy, radius, noise_level)
        
        noisy_aircraft_dict = aircraft_dict.copy()
        noisy_aircraft_dict['position'] = [px_noisy, py_noisy, pz]
        noisy_aircraft_dict['velocity'] = [vx_noisy, vy_noisy, vz]
        
        noisy_pos = np.array([px_noisy, py_noisy, pz])
        noisy_aircraft_dict['distance'] = np.linalg.norm(noisy_pos - current_pos)
        
        return noisy_aircraft_dict
    
    def _add_evtol_position_noise(self, px: float, py: float, vx: float, vy: float, 
                                 radius: float, noise_level: float) -> Tuple[float, float]:
        """为eVTOL添加位置噪声"""
        current_speed = np.sqrt(vx**2 + vy**2)
        
        if current_speed > 1e-6:
            velocity_unit_x = vx / current_speed
            velocity_unit_y = vy / current_speed
            
            forward_range = 0.25 * radius * noise_level
            backward_range = 0.5 * radius * noise_level
            along_velocity_noise = np.random.uniform(-backward_range, forward_range)
            
            perpendicular_noise_std = noise_level * radius * 0.1
            perpendicular_noise = np.random.normal(0, perpendicular_noise_std)
            perpendicular_noise = np.clip(perpendicular_noise, -3*perpendicular_noise_std, 3*perpendicular_noise_std)
            
            px_noise = along_velocity_noise * velocity_unit_x + perpendicular_noise * (-velocity_unit_y)
            py_noise = along_velocity_noise * velocity_unit_y + perpendicular_noise * velocity_unit_x
        else:
            position_noise_std = noise_level * radius * 0.1
            px_noise = np.random.normal(0, position_noise_std)
            py_noise = np.random.normal(0, position_noise_std)
            max_noise = 3 * position_noise_std
            px_noise = np.clip(px_noise, -max_noise, max_noise)
            py_noise = np.clip(py_noise, -max_noise, max_noise)
        
        return px + px_noise, py + py_noise
    
    def _add_evtol_velocity_noise(self, vx: float, vy: float, radius: float, 
                                 noise_level: float) -> Tuple[float, float]:
        """为eVTOL添加速度噪声"""
        current_speed = np.sqrt(vx**2 + vy**2)
        v_pref = max(0.1, current_speed) if current_speed > 1e-6 else 1.0
        
        if current_speed > 1e-6:
            current_angle = np.arctan2(vy, vx)
            
            max_angle_noise = noise_level * np.pi / 18  # 10度
            angle_noise = np.random.normal(0, max_angle_noise / 3)
            angle_noise = np.clip(angle_noise, -max_angle_noise, max_angle_noise)
            noisy_angle = current_angle + angle_noise
            
            speed_noise_std = noise_level * current_speed * 0.1
            speed_noise = np.random.normal(0, speed_noise_std)
            max_speed_noise = 3 * speed_noise_std
            speed_noise = np.clip(speed_noise, -max_speed_noise, max_speed_noise)
            noisy_speed = max(0, current_speed + speed_noise)
            noisy_speed = min(noisy_speed, 3 * v_pref)
            
            vx_noisy = noisy_speed * np.cos(noisy_angle)
            vy_noisy = noisy_speed * np.sin(noisy_angle)
        else:
            v_noise_std = noise_level * v_pref * 0.05
            vx_noise = np.random.normal(0, v_noise_std)
            vy_noise = np.random.normal(0, v_noise_std)
            max_v_noise = 3 * v_noise_std
            vx_noise = np.clip(vx_noise, -max_v_noise, max_v_noise)
            vy_noise = np.clip(vy_noise, -max_v_noise, max_v_noise)
            vx_noisy = vx + vx_noise
            vy_noisy = vy + vy_noise
        
        return vx_noisy, vy_noisy
    
    def _add_uav_position_noise(self, px: float, py: float, radius: float, 
                               noise_level: float) -> Tuple[float, float]:
        """为UAV添加位置噪声"""
        position_noise_std = noise_level * radius
        px_noise = np.random.normal(0, position_noise_std)
        py_noise = np.random.normal(0, position_noise_std)
        max_pos_noise = 3 * position_noise_std
        px_noise = np.clip(px_noise, -max_pos_noise, max_pos_noise)
        py_noise = np.clip(py_noise, -max_pos_noise, max_pos_noise)
        return px + px_noise, py + py_noise
    
    def _add_uav_velocity_noise(self, vx: float, vy: float, radius: float, 
                               noise_level: float) -> Tuple[float, float]:
        """为UAV添加速度噪声"""
        current_speed = np.sqrt(vx**2 + vy**2)
        v_pref = max(0.1, current_speed) if current_speed > 1e-6 else 1.0
        
        if current_speed > 1e-6:
            current_angle = np.arctan2(vy, vx)
            
            max_angle_noise = noise_level * np.pi / 4  # 45度
            angle_noise = np.random.normal(0, max_angle_noise / 3)
            angle_noise = np.clip(angle_noise, -max_angle_noise, max_angle_noise)
            noisy_angle = current_angle + angle_noise
            
            speed_noise_std = noise_level * current_speed * 0.2
            speed_noise = np.random.normal(0, speed_noise_std)
            max_speed_noise = 3 * speed_noise_std
            speed_noise = np.clip(speed_noise, -max_speed_noise, max_speed_noise)
            noisy_speed = max(0, current_speed + speed_noise)
            noisy_speed = min(noisy_speed, 3 * v_pref)
            
            vx_noisy = noisy_speed * np.cos(noisy_angle)
            vy_noisy = noisy_speed * np.sin(noisy_angle)
        else:
            v_noise_std = noise_level * v_pref * 0.1
            vx_noise = np.random.normal(0, v_noise_std)
            vy_noise = np.random.normal(0, v_noise_std)
            max_v_noise = 3 * v_noise_std
            vx_noise = np.clip(vx_noise, -max_v_noise, max_v_noise)
            vy_noise = np.clip(vy_noise, -max_v_noise, max_v_noise)
            vx_noisy = vx + vx_noise
            vy_noisy = vy + vy_noise
        
        return vx_noisy, vy_noisy

class ComplexObservationProcessor(DefaultObservationProcessor):
    """复杂观测处理器，基于 airspace_sim_large.py 的观测逻辑"""
    
    def __init__(self, config=None, sb3_config=None):
        super().__init__(config)
        self.sb3_config = sb3_config
        
        # 从 sb3_config 读取观测相关配置
        if sb3_config:
            self._configure_from_sb3(sb3_config)
        else:
            self._set_default_obs_config()
        
        # 预测轨迹相关
        self.human_future_traj = None
        self.prev_human_pos = None
        self.last_human_states = None
        self.state = None
        self.evtol_index = []
        self.uav_index = []
        
    def _configure_from_sb3(self, config):
        """从 sb3_config 配置观测参数"""
        # 基础参数
        self.circle_radius = getattr(config.sim, 'circle_radius', 100.0)
        self.human_num = getattr(config.sim, 'human_num', 20)
        self.evtol_num = getattr(config.sim, 'evtol_num', 10)
        self.uav_num = getattr(config.sim, 'uav_num', 10)
        self.predict_steps = getattr(config.sim, 'predict_steps', 4)
        self.pred_timestep = getattr(config.data, 'pred_timestep', 0.1)
        
        # 观测配置
        obs_config = getattr(config, 'obs', None)
        if obs_config:
            self.use_norm_dis = getattr(obs_config, 'use_norm', False)
            self.norm_scale = getattr(obs_config, 'scale', 1.0)
            self.use_relative_pos = getattr(obs_config, 'use_relative_pos', False)
            self.use_norm_radius = getattr(obs_config, 'use_norm_radius', False)
            self.use_norm_vel = getattr(obs_config, 'use_norm_vel', False)
            self.extend_obs_radius = getattr(obs_config, 'extend_obs_radius', False)
            self.extend_obs_vel = getattr(obs_config, 'extend_obs_vel', False)
            self.obs_noise_level = getattr(obs_config, 'obs_noise_level', 0.0)
        else:
            self._set_default_obs_config()
        
        # 传感器范围
        self.uav_sensor_range = getattr(config.uav, 'sensor_range', 50.0)
        self.evtol_sensor_range = getattr(config.evtol, 'sensor_range', 100.0)
        
        # 预测方法
        self.pred_method = getattr(config.sim, 'predict_method', 'const_vel')
        
        # 计算预测间隔
        
    def _set_default_obs_config(self):
        """设置默认观测配置"""
        self.circle_radius = 100.0
        self.human_num = 20
        self.evtol_num = 10
        self.uav_num = 10
        self.predict_steps = 4
        self.use_norm_dis = False
        self.norm_scale = 1.0
        self.use_relative_pos = False
        self.use_norm_radius = False
        self.use_norm_vel = False
        self.extend_obs_radius = False
        self.extend_obs_vel = False
        self.obs_noise_level = 0.0
        self.uav_sensor_range = 50.0
        self.evtol_sensor_range = 100.0
        self.pred_method = 'const_vel'
        self.pred_timestep = 0.1
    
    def process_observation(self, state):
        """处理观测数据，返回复杂的字典结构观测"""
        ob = {}
        self.state = state
        # 获取机器人状态
        robot_pos = state.vehicle.get_position_array()
        robot_vel = state.vehicle.get_velocity_array()
        robot_yaw = state.vehicle.get_yaw()
        robot_radius = state.vehicle.radius
        robot_v_pref = state.vehicle.max_velocity
        
        # 获取目标位置
        goal_pos = state.navigation.goal_position.to_array()
        # yaw应该是速度组成的yaw，而非姿态yaw
        robot_yaw = np.arctan2(robot_vel[1], robot_vel[0])

        # 生成机器人节点观测
        if self.use_relative_pos:
            ob['robot_node'] = [
                goal_pos[0] - robot_pos[0],  # 相对目标位置
                goal_pos[1] - robot_pos[1],
                robot_radius,
                robot_v_pref,
                robot_yaw
            ]
            if self.use_norm_dis:
                ob['robot_node'][0] = (ob['robot_node'][0] + self.circle_radius*2) / (self.circle_radius*4) * (self.norm_scale*2) - self.norm_scale
                ob['robot_node'][1] = (ob['robot_node'][1] + self.circle_radius*2) / (self.circle_radius*4) * (self.norm_scale*2) - self.norm_scale
            if self.use_norm_vel:
                ob['robot_node'][3] = ob['robot_node'][3] / robot_v_pref
        else:
            ob['robot_node'] = [
                robot_pos[0],
                robot_pos[1],
                robot_radius,
                goal_pos[0],
                goal_pos[1],
                robot_v_pref,
                robot_yaw
            ]
            if self.use_norm_dis:
                ob['robot_node'][0] = (ob['robot_node'][0] + self.circle_radius) / (self.circle_radius*2) * (self.norm_scale*2) - self.norm_scale
                ob['robot_node'][1] = (ob['robot_node'][1] + self.circle_radius) / (self.circle_radius*2) * (self.norm_scale*2) - self.norm_scale
                ob['robot_node'][3] = (ob['robot_node'][3] + self.circle_radius) / (self.circle_radius*2) * (self.norm_scale*2) - self.norm_scale
                ob['robot_node'][4] = (ob['robot_node'][4] + self.circle_radius) / (self.circle_radius*2) * (self.norm_scale*2) - self.norm_scale
            if self.use_norm_vel:
                ob['robot_node'][5] = ob['robot_node'][5] / robot_v_pref
        
        ob['robot_node'] = np.array(ob['robot_node'], dtype=np.float32)
        ob['robot_node'] = np.expand_dims(ob['robot_node'], axis=0)
        
        # 时间边信息（速度）
        ob['temporal_edges'] = np.array([robot_vel[0], robot_vel[1]], dtype=np.float32)
        if self.use_norm_vel:
            ob['temporal_edges'][0] = ob['temporal_edges'][0] / robot_v_pref
            ob['temporal_edges'][1] = ob['temporal_edges'][1] / robot_v_pref
        ob['temporal_edges'] = np.expand_dims(ob['temporal_edges'], axis=0)
        
        # 获取其他飞机状态并分类
        aircraft_states = self._extract_aircraft_states(state)
        evtol_states, uav_states, all_states = self._classify_aircraft_states(aircraft_states, robot_pos)
        
        # 计算预测轨迹
        self._update_human_states(all_states)
        predicted_states = self._calc_human_future_traj(all_states, robot_pos, robot_vel)
        
        # 生成空间边观测
        spatial_edges_length = int(2*(self.predict_steps+1))
        if self.extend_obs_radius:
            spatial_edges_length += 1
        if self.extend_obs_vel:
            spatial_edges_length += 2
        
        # 初始化观测数组
        ob['spatial_edges'] = np.ones((self.human_num, spatial_edges_length), dtype=np.float32) * np.inf
        ob['spatial_edges_evtol'] = np.ones((self.human_num, spatial_edges_length), dtype=np.float32) * np.inf
        ob['spatial_edges_uav'] = np.ones((self.human_num, spatial_edges_length), dtype=np.float32) * np.inf
        
        # 填充观测数据
        self._fill_spatial_edges(ob, all_states, evtol_states, uav_states, predicted_states, robot_pos, robot_vel, robot_v_pref)
        
        # 代理类型
        ob['agent_types'] = self._generate_agent_types(all_states)
        
        # 检测到的飞机数量
        ob['detected_human_num'] = np.array([len(all_states)], dtype=np.int32)
        ob['detected_evtol_num'] = np.array([len(evtol_states)], dtype=np.int32)
        ob['detected_uav_num'] = np.array([len(uav_states)], dtype=np.int32)
        
        # 确保最少有一个检测到的对象（为了兼容性）
        if ob['detected_human_num'][0] == 0:
            ob['detected_human_num'] = np.array([1], dtype=np.int32)
        if ob['detected_evtol_num'][0] == 0:
            ob['detected_evtol_num'] = np.array([1], dtype=np.int32)
        if ob['detected_uav_num'][0] == 0:
            ob['detected_uav_num'] = np.array([1], dtype=np.int32)
        
        # 填充无限值
        self._fill_infinity_values(ob)
        
        return ob
    
    def _classify_aircraft_states(self, aircraft_states, robot_pos):
        """将飞机状态按类型分类"""
        evtol_states = []
        uav_states = []
        all_states = []
        
        for aircraft in aircraft_states:
            # 检查是否在感知范围内
            aircraft_pos = np.array(aircraft['position'])
            distance = np.linalg.norm(aircraft_pos - robot_pos)
            
            # 根据飞机类型确定感知范围
            aircraft_type = aircraft.get('aircraft_type', 'uav')
            # if aircraft_type == 'evtol':
            #     sensor_range = self.evtol_sensor_range
            # else:
            #     sensor_range = self.uav_sensor_range
            sensor_range = self.config.observation_radius
            if distance <= sensor_range:
                all_states.append(aircraft)
                if aircraft_type == 'evtol':
                    evtol_states.append(aircraft)
                else:
                    uav_states.append(aircraft)
        
        return evtol_states, uav_states, all_states
    
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
        if self.pred_method == 'const_vel':
            time_step = self.pred_timestep  # 假设时间步长
            for step in range(1, self.predict_steps + 1):
                self.human_future_traj[step] = self.human_future_traj[0].copy()
                # 更新位置
                self.human_future_traj[step, :, 0] += self.human_future_traj[0, :, 2] * step * time_step
                self.human_future_traj[step, :, 1] += self.human_future_traj[0, :, 3] * step * time_step
        
        return self.human_future_traj
    
    def _fill_spatial_edges(self, ob, all_states, evtol_states, uav_states, predicted_states, robot_pos, robot_vel, robot_v_pref):
        """填充空间边观测"""
        if predicted_states is None or len(all_states) == 0:
            return
        
        # 计算相对位置
        predicted_pos = predicted_states[:, :, :2]  # [steps, humans, 2]
        predicted_pos = np.transpose(predicted_pos, (1, 0, 2)) - robot_pos[:2]  # [humans, steps, 2]
        
        # 归一化处理
        if self.use_norm_dis:
            normalized_pos = (predicted_pos + self.circle_radius*2) / (self.circle_radius*4) * (self.norm_scale*2) - self.norm_scale
        elif self.use_norm_radius:
            # 使用半径归一化
            human_radii = np.array([state.get('radius', 1.0) for state in all_states])
            human_radii = human_radii[:, np.newaxis, np.newaxis]
            normalized_pos = predicted_pos / human_radii
        else:
            normalized_pos = predicted_pos
        
        # 填充主要的空间边
        num_aircraft = min(len(all_states), self.human_num)
        spatial_data = normalized_pos[:num_aircraft].reshape((num_aircraft, -1))
        data_length = min(spatial_data.shape[1], int(2*(self.predict_steps+1)))
        ob['spatial_edges'][:num_aircraft, :data_length] = spatial_data[:, :data_length]
        
        # 扩展观测
        curr_idx = int(2*(self.predict_steps+1))
        if self.extend_obs_radius and curr_idx < ob['spatial_edges'].shape[1]:
            for i in range(num_aircraft):
                if i < len(all_states):
                    radius = all_states[i].get('radius', 1.0)
                    if self.use_norm_radius:
                        ob['spatial_edges'][i, curr_idx] = radius / (radius + self.state.vehicle.radius)
                    else:
                        ob['spatial_edges'][i, curr_idx] = radius
            curr_idx += 1
        
        if self.extend_obs_vel and curr_idx + 1 < ob['spatial_edges'].shape[1]:
            for i in range(num_aircraft):
                if i < len(all_states):
                    vel = all_states[i]['velocity']
                    rel_vel_x = vel[0] - robot_vel[0]
                    rel_vel_y = vel[1] - robot_vel[1]
                    if self.use_norm_vel:
                        ob['spatial_edges'][i, curr_idx] = rel_vel_x / robot_v_pref
                        ob['spatial_edges'][i, curr_idx+1] = rel_vel_y / robot_v_pref
                    else:
                        ob['spatial_edges'][i, curr_idx] = rel_vel_x
                        ob['spatial_edges'][i, curr_idx+1] = rel_vel_y
        
        # 分别填充 evtol 和 uav 的观测
        idx_evtol = 0
        idx_uav = 0
        self.evtol_index = []
        self.uav_index = []
        for i in range(len(all_states)):
            if all_states[i]['aircraft_type'] == 'evtol':
                ob['spatial_edges_evtol'][idx_evtol, :] = ob['spatial_edges'][i, :]
                self.evtol_index.append(i)
                idx_evtol += 1
            else:
                ob['spatial_edges_uav'][idx_uav, :] = ob['spatial_edges'][i, :]
                self.uav_index.append(i)
                idx_uav += 1
    
    
    def _generate_agent_types(self, all_states):
        """生成代理类型数组"""
        agent_types = np.zeros(self.human_num, dtype=np.int32)
        for i, state in enumerate(all_states):
            if i < self.human_num:
                aircraft_type = state.get('aircraft_type', 'uav')
                agent_types[i] = 0 if aircraft_type == 'evtol' else 1
        return agent_types
    
    def _fill_infinity_values(self, ob):
        """填充无限值"""
        if self.use_norm_dis:
            fill_value = self.norm_scale
        elif self.use_norm_radius:
            fill_value = max(self.uav_sensor_range / (self.state.vehicle.radius + self.sb3_config.uav.radius), 
                           self.evtol_sensor_range / (self.state.vehicle.radius + self.sb3_config.evtol.radius))
        else:
            fill_value = self.circle_radius * 2
        
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = fill_value
        ob['spatial_edges_evtol'][np.isinf(ob['spatial_edges_evtol'])] = fill_value
        ob['spatial_edges_uav'][np.isinf(ob['spatial_edges_uav'])] = fill_value



class IsaacLikeObservationProcessor(ComplexObservationProcessor):
    """Isaac-style observation processor for SB3 environment.
    - Inherits ComplexObservationProcessor to keep the same observation dict schema
    - Changes spatial_edges normalization and padding: zero-padding instead of +inf
    - Optional angle-distance encoding; default disabled to match AirSimNavSB3Env shapes
    """

    def __init__(self, config=None, sb3_config=None):
        super().__init__(config, sb3_config)

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
        # 若未在sim配置circle_radius，则回退到飞行区域估算

        self.circle_radius = getattr(obs_cfg, 'circle_radius', 113.137085)
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