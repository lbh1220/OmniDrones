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
TrafficEVTOLManager - 管理交通EVTOL的类

实现外部动力学的EVTOL仿真，包括航线生成、轨迹平滑和位置更新
"""

import torch
import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from omni.isaac.core.utils import prims as prim_utils

from omni_drones.robots.evtol import EVTOLBase
from omni_drones.traffic.utils.state import TrafficState, Waypoint, Waypoint_ex
from omni_drones.traffic.utils.generator import EVTOLTargetGenerator





class TrafficEVTOLManager:
    """EVTOL交通管理器，实现外部动力学仿真"""
    
    def __init__(self, config, device: str = "cuda", traffic_prim_path: str = "/World/Traffic"):
        self.config = config
        self.device = device
        self.num_evtols = config.num_evtols
        self.traffic_prim_path = traffic_prim_path
        
        # 创建traffic命名空间
        if not prim_utils.is_prim_path_valid(self.traffic_prim_path):
            prim_utils.define_prim(self.traffic_prim_path)
        
        # EVTOL基础类
        self.evtol: Optional[EVTOLBase] = None
        
        # 初始化状态
        self.is_created = False
        self.is_initialized = False
        
        # 状态管理器 - 使用新的状态管理架构
        self.state = TrafficState(device)
        
        # 预生成的航线数据 - 修改为预计算模式
        self.all_courses = []  # 所有可用的航线
        self.all_smooth_waypoints = []  # 所有可用的平滑轨迹
        self.evtol_course_assignments = []  # 每个EVTOL被分配的course索引
        
        # EVTOL参数
        self.max_speed = config.evtol.max_speed
        self.arrival_threshold = config.evtol.arrival_threshold
        self.turn_radius = config.evtol.turn_radius
        
        # 目标生成器
        self.target_generator = EVTOLTargetGenerator(config, device)
        
        # 时间管理
        self.last_update_time = 0.0
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def create_evtols(self):
        """创建EVTOL primitives"""
        if self.is_created:
            self.logger.warning("EVTOLs already created")
            return
        
        if self.num_evtols <= 0:
            self.logger.info("No EVTOLs to create")
            self.is_created = True
            return
        
        # 创建EVTOL基础对象
        self.evtol = EVTOLBase(device=self.device)
        
        # 预生成所有可用的航线
        self._pregenerate_all_courses(self.config.evtol.course_num)
        
        # 为每个EVTOL分配航线
        initial_positions = []
        prim_paths = []
        scales = []
        
        for i in range(self.num_evtols):
            # 随机分配一条course
            course_idx = random.randint(0, len(self.all_courses) - 1)
            self.evtol_course_assignments.append(course_idx)            
            # 设置初始位置为航线起点
            smooth_waypoints = self.all_smooth_waypoints[course_idx]
            initial_positions.append((smooth_waypoints[0].x, smooth_waypoints[0].y, smooth_waypoints[0].z))
            prim_paths.append(f"{self.traffic_prim_path}/traffic_evtol_{i}")
            scales.append((self.config.evtol.safety_radius, self.config.evtol.safety_radius, 1.0))  # EVTOL通常比较大
        
        # 创建primitives - traffic 内部不碰撞
        created_prims = self.evtol.spawn(
            translations=initial_positions,
            prim_paths=prim_paths,
            scales=scales,
            asset_name="Sphere",  # 使用球体作为占位符
            enable_collision=False  # traffic 内部不碰撞
        )
        
        self.is_created = True
        self.logger.info(f"Created {len(created_prims)} EVTOL primitives")
    
    def _pregenerate_all_courses(self, num_courses: int):
        """预生成所有可用的航线"""
        self.all_courses = []
        self.all_smooth_waypoints = []
        for i in range(num_courses):
            # 生成航线并获取平滑后的轨迹
            course, smooth_waypoints = self.target_generator.generate_course_with_smooth_trajectory()
            self.all_courses.append(course)
            self.all_smooth_waypoints.append(smooth_waypoints)
        self.logger.info(f"Pre-generated {num_courses} courses")
    
    def initialize(self):
        """初始化EVTOL管理器"""
        if self.is_initialized:
            self.logger.warning("EVTOLs already initialized")
            return
        
        if not self.is_created:
            self.logger.error("EVTOLs not created yet. Call create_evtols() first.")
            return
        
        if self.num_evtols <= 0:
            self.logger.info("No EVTOLs to initialize")
            self.is_initialized = True
            return
        
        # 初始化EVTOL视图
        self.evtol.initialize(prim_paths_expr=f"{self.traffic_prim_path}/traffic_evtol_*")
        
        # 初始化状态管理器
        names = [f"traffic_evtol_{i}" for i in range(self.num_evtols)]
        aircraft_types = ["evtol"] * self.num_evtols
        safety_radius = [self.config.evtol.safety_radius] * self.num_evtols  # EVTOL通常比较大
        max_speed = [self.max_speed] * self.num_evtols
        self.state.initialize_aircraft(names, aircraft_types, safety_radius, max_speed, self.device)
        
        # 为每个EVTOL在state中设置对应的平滑后轨迹
        for i in range(self.num_evtols):
            course_idx = self.evtol_course_assignments[i]
            course_waypoints = self.all_courses[course_idx]
            
            # 转换smooth waypoints为新格式
            # waypoints = []
            # for wp in smooth_waypoints:
            #     waypoints.append(Waypoint(wp.x, wp.y, wp.z, self.max_speed))
            
            self.state.set_waypoints_for_aircraft(i, course_waypoints)
        
        # 设置初始状态
        self._update_initial_states()
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.num_evtols} EVTOLs")
    
    
    def _update_initial_states(self):
        """更新初始状态"""
        if not self.evtol or not self.evtol.is_valid:
            return
        for i in range(self.num_evtols):
            course_idx = random.randint(0, len(self.all_smooth_waypoints) - 1)
            smooth_waypoints = self.all_smooth_waypoints[course_idx]
            start_idx = random.randint(0, len(smooth_waypoints) - 1)
            self.state.positions[i] = torch.tensor([smooth_waypoints[start_idx].x, smooth_waypoints[start_idx].y, smooth_waypoints[start_idx].z], device=self.device)
            self.state.rotations[i] = torch.tensor([smooth_waypoints[start_idx].qw, smooth_waypoints[start_idx].qx, smooth_waypoints[start_idx].qy, smooth_waypoints[start_idx].qz], device=self.device)
            self.state.velocities[i] = torch.tensor([smooth_waypoints[start_idx].vx, smooth_waypoints[start_idx].vy, smooth_waypoints[start_idx].vz], device=self.device)
            self.state.angular_velocities[i] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            self.state.current_waypoint_indices[i] = start_idx
            self.state.target_positions[i] = torch.tensor([smooth_waypoints[-1].x, smooth_waypoints[-1].y, smooth_waypoints[-1].z], device=self.device)
            self.state.start_positions[i] = torch.tensor([smooth_waypoints[0].x, smooth_waypoints[0].y, smooth_waypoints[0].z], device=self.device)
    

        self.evtol.set_world_poses(self.state.positions.unsqueeze(0), self.state.rotations.unsqueeze(0))
        
        
    

    
    def step(self, dt: float = 0.02):
        """执行一个仿真步骤"""
        if not self.is_initialized:
            self.logger.warning("EVTOLs not initialized yet")
            return
        
        if self.num_evtols <= 0:
            return
        
        # 更新每个EVTOL的位置和姿态
        self.evtol.update_states()
        for i in range(self.num_evtols):
            self._update_evtol_state(i, dt)
        
        # 应用新的位置和姿态到Isaac Sim
        self.evtol.set_world_poses(self.state.positions.unsqueeze(0), self.state.rotations.unsqueeze(0))
    
    def _update_evtol_state(self, evtol_idx: int, dt: float):
        """更新单个EVTOL的状态 - 使用基于距离的连续移动和插值"""
        # 获取平滑后的轨迹waypoints
        course_idx = self.evtol_course_assignments[evtol_idx]
        smooth_waypoints = self.all_smooth_waypoints[course_idx]
        
        if not smooth_waypoints:
            return
        
        # 获取当前位置和航路点索引
        current_pos = self.state.positions[evtol_idx].cpu().numpy()
        current_waypoint_idx = self.state.current_waypoint_indices[evtol_idx].item()
        
        # 计算这一步要前进的距离
        move_distance = self.max_speed * dt
        
        # 如果当前航路点索引超出范围，需要重新设置航路
        if current_waypoint_idx >= len(smooth_waypoints):
            self._reassign_course_for_evtol(evtol_idx)
            return
        
        # 循环直到消耗完所有距离或者到达终点
        remaining_dist = move_distance
        while remaining_dist > 0 and current_waypoint_idx < len(smooth_waypoints):
            # 获取当前目标点
            target = smooth_waypoints[current_waypoint_idx]
            target_pos = np.array([target.x, target.y, target.z])
            
            # 计算到目标点的向量和距离
            direction = target_pos - current_pos
            distance_to_target = np.linalg.norm(direction)
            
            # 如果距离为0，直接前进到下一个点
            if distance_to_target < 1e-6:
                current_waypoint_idx += 1
                continue
                
            # 单位方向向量
            direction = direction / distance_to_target
            
            # 如果剩余距离足够到达目标点
            if remaining_dist >= distance_to_target:
                # 移动到目标点
                current_pos = target_pos
                
                # 更新剩余距离
                remaining_dist -= distance_to_target
                
                # 前进到下一个航路点
                current_waypoint_idx += 1
            else:
                # 沿方向前进剩余距离
                current_pos = current_pos + direction * remaining_dist
                remaining_dist = 0
        
        # 更新当前位置
        self.state.positions[evtol_idx] = torch.tensor([current_pos[0], current_pos[1], current_pos[2]], device=self.device)
        
        # 更新航路点索引
        self.state.current_waypoint_indices[evtol_idx] = current_waypoint_idx
        
        # 更新姿态和速度
        if current_waypoint_idx < len(smooth_waypoints):
            # 还有下一个点，在相邻航路点之间插值
            if current_waypoint_idx > 0:
                prev_waypoint = smooth_waypoints[current_waypoint_idx - 1]
                next_waypoint = smooth_waypoints[current_waypoint_idx]
                
                prev_pos = np.array([prev_waypoint.x, prev_waypoint.y, prev_waypoint.z])
                next_pos = np.array([next_waypoint.x, next_waypoint.y, next_waypoint.z])
                
                dist_to_next = np.linalg.norm(next_pos - current_pos)
                dist_to_prev = np.linalg.norm(prev_pos - current_pos)
                total_dist = dist_to_next + dist_to_prev
                
                if total_dist < 1e-6:
                    ratio = 1.0
                else:
                    ratio = dist_to_prev / total_dist
                
                # 在prev_waypoint和next_waypoint之间插值姿态和速度
                qx = prev_waypoint.qx * (1.0 - ratio) + next_waypoint.qx * ratio
                qy = prev_waypoint.qy * (1.0 - ratio) + next_waypoint.qy * ratio
                qz = prev_waypoint.qz * (1.0 - ratio) + next_waypoint.qz * ratio
                qw = prev_waypoint.qw * (1.0 - ratio) + next_waypoint.qw * ratio
                
                # 归一化四元数
                norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                if norm > 1e-6:
                    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
                
                self.state.rotations[evtol_idx] = torch.tensor([qw, qx, qy, qz], device=self.device)
                
                vx = prev_waypoint.vx * (1.0 - ratio) + next_waypoint.vx * ratio
                vy = prev_waypoint.vy * (1.0 - ratio) + next_waypoint.vy * ratio
                vz = prev_waypoint.vz * (1.0 - ratio) + next_waypoint.vz * ratio
                
                self.state.velocities[evtol_idx] = torch.tensor([vx, vy, vz], device=self.device)
                self.state.velocity_commands[evtol_idx] = torch.tensor([vx, vy, vz], device=self.device)
            else:
                # 使用第一个航路点的姿态和速度
                first_wp = smooth_waypoints[0]
                self.state.rotations[evtol_idx] = torch.tensor([first_wp.qw, first_wp.qx, first_wp.qy, first_wp.qz], device=self.device)
                self.state.velocities[evtol_idx] = torch.tensor([first_wp.vx, first_wp.vy, first_wp.vz], device=self.device)
                self.state.velocity_commands[evtol_idx] = torch.tensor([first_wp.vx, first_wp.vy, first_wp.vz], device=self.device)
        else:
            # 已到达终点，需要重新设置航路
            self._reassign_course_for_evtol(evtol_idx)
    
    def _reassign_course_for_evtol(self, evtol_idx: int):
        """为特定EVTOL重新分配预生成的航线"""
        # 随机选择一条新的航线
        new_course_idx = random.randint(0, len(self.all_courses) - 1)
        self.evtol_course_assignments[evtol_idx] = new_course_idx
        
        # 重置航路点索引
        self.state.current_waypoint_indices[evtol_idx] = 0
        
        # 设置新的起始位置 - 使用第一个平滑轨迹点
        smooth_waypoints = self.all_smooth_waypoints[new_course_idx]
        first_smooth_wp = smooth_waypoints[0]
        self.state.positions[evtol_idx] = torch.tensor([first_smooth_wp.x, first_smooth_wp.y, first_smooth_wp.z], device=self.device)
        self.state.start_positions[evtol_idx] = torch.tensor([first_smooth_wp.x, first_smooth_wp.y, first_smooth_wp.z], device=self.device)
        # 更新目标位置 - 使用最后一个平滑轨迹点
        last_smooth_wp = smooth_waypoints[-1]
        self.state.target_positions[evtol_idx] = torch.tensor([last_smooth_wp.x, last_smooth_wp.y, last_smooth_wp.z], device=self.device)
        
        # 更新state中的waypoints（用于兼容性）
        waypoints = []
        for wp in smooth_waypoints:
            waypoints.append(Waypoint(wp.x, wp.y, wp.z, self.max_speed))
        self.state.set_waypoints_for_aircraft(evtol_idx, waypoints)
        
        # self.logger.debug(f"Reassigned course {new_course_idx} for EVTOL {evtol_idx}")
    

    
    def get_positions(self) -> torch.Tensor:
        """获取EVTOL位置 [1, N, 3]"""
        return self.state.positions.unsqueeze(0)
    
    def get_velocities(self) -> torch.Tensor:
        """获取EVTOL速度 [1, N, 3]"""
        return self.state.velocities.unsqueeze(0)
    
    def get_rotations(self) -> torch.Tensor:
        """获取EVTOL姿态 [1, N, 4]"""
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
    
    def get_state_manager(self) -> TrafficState:
        """获取状态管理器"""
        return self.state
    
    def reset(self):
        """重置所有EVTOL"""
        if not self.is_initialized:
            return
        
        # 初始化状态管理器
        names = [f"traffic_evtol_{i}" for i in range(self.num_evtols)]
        aircraft_types = ["evtol"] * self.num_evtols
        safety_radius = [self.config.evtol.safety_radius] * self.num_evtols  # EVTOL通常比较大
        max_speed = [self.max_speed] * self.num_evtols
        self.state.initialize_aircraft(names, aircraft_types, safety_radius, max_speed, self.device)
        
        # 为每个EVTOL在state中设置对应的平滑后轨迹
        for i in range(self.num_evtols):
            course_idx = self.evtol_course_assignments[i]
            course_waypoints = self.all_courses[course_idx]
            
            # 转换smooth waypoints为新格式
            # waypoints = []
            # for wp in smooth_waypoints:
            #     waypoints.append(Waypoint(wp.x, wp.y, wp.z, self.max_speed))
            
            self.state.set_waypoints_for_aircraft(i, course_waypoints)
        
        # 设置初始状态
        self._update_initial_states()
        
        self.is_initialized = True
        
        self.logger.info(f"Reset {self.num_evtols} EVTOLs")
    
    def cleanup(self):
        """清理资源"""
        if self.evtol:
            self.evtol.cleanup()
            self.evtol = None
        
        self.is_initialized = False
        self.is_created = False
        
        self.logger.info("EVTOL manager cleanup complete")
    
    def get_safety_radius(self) -> torch.Tensor:
        """Get safety radius of all EVTOLs."""
        return self.state.safety_radius