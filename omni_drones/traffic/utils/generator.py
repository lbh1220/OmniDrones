import numpy as np
import random
from typing import List, Tuple
import torch

from omni_drones.traffic.utils.state import Waypoint_ex, Waypoint
import omni.isaac.lab.utils.math as math_utils

class DroneTargetGenerator_simple:
    """Generate flight targets for traffic drones."""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.bounds = config.area_bounds
        self.device = device
    
    def generate_target(self) -> torch.Tensor:
        """Generate a random target position within bounds."""
        x_range = self.bounds.xmax -  self.bounds.xmin
        y_range = self.bounds.ymax - self.bounds.ymin
        
        x = torch.rand(1, device=self.device) * x_range + self.bounds.xmin
        y = torch.rand(1, device=self.device) * y_range + self.bounds.ymin
        z = torch.tensor([self.config.flight_height], device=self.device)
        return torch.cat([x, y, z])
    
    def generate_waypoint_targets(self, num_waypoints: int = 3) -> List[torch.Tensor]:
        """Generate a series of waypoint targets for more complex flight patterns."""
        targets = []
        for _ in range(num_waypoints):
            targets.append(self.generate_target())
        return targets
class DroneTargetGenerator:
    """Enhanced target generator with candidate points and batch generation with random offsets."""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.bounds = config.area_bounds
        self.device = device
        self.flight_height = config.flight_height
        
        # 候选目标点
        self.candidate_targets = None
        self.num_candidates = config.drone.target_num
        
        # 偏移参数
        self.min_offset_radius = 0.0  # 最小偏移半径
        self.max_offset_radius = getattr(config, 'max_offset_radius', 8.0)  # 最大偏移半径
        
    def initialize_targets(self, num_candidates: int = 20):
        """Initialize candidate target points within bounds, distributed relatively evenly.
        
        Args:
            num_candidates: Number of candidate target points to generate
        """
        self.num_candidates = num_candidates
        
        # 计算网格大小以实现相对均匀的分布
        grid_size_x = (self.bounds.xmax - self.bounds.xmin) / (self.num_candidates ** 0.5)
        grid_size_y = (self.bounds.ymax - self.bounds.ymin) / (self.num_candidates ** 0.5)
        
        # 生成候选目标点
        candidate_targets = []
        
        for i in range(self.num_candidates):
            # 使用网格索引计算基础位置
            grid_x = i % int(self.num_candidates ** 0.5)
            grid_y = i // int(self.num_candidates ** 0.5)
            
            # 在网格中心添加随机偏移
            base_x = self.bounds.xmin + grid_x * grid_size_x + grid_size_x * 0.5
            base_y = self.bounds.ymin + grid_y * grid_size_y + grid_size_y * 0.5
            
            # 添加随机偏移以避免完全对齐
            offset_x = (torch.rand(1, device=self.device) - 0.5) * grid_size_x * 0.3
            offset_y = (torch.rand(1, device=self.device) - 0.5) * grid_size_y * 0.3
            
            x = base_x + offset_x
            y = base_y + offset_y
            z = torch.tensor([self.flight_height], device=self.device)
            
            target = torch.cat([x, y, z])
            candidate_targets.append(target)
        
        # 转换为张量 [num_candidates, 3]
        self.candidate_targets = torch.stack(candidate_targets)
        
        print(f"Initialized {self.num_candidates} candidate targets")
    
    def generate_targets(self, num_drones: int, max_offset = None) -> torch.Tensor:
        """Batch generate targets for multiple drones with random offsets.
        
        Args:
            num_drones: Number of drones to generate targets for
            
        Returns:
            Tensor of shape [num_drones, 3] containing target positions
        """
        if max_offset is not None:
            self.max_offset_radius = max_offset
        
        if self.candidate_targets is None:
            raise RuntimeError("Must call initialize_targets() first")
        
        if num_drones <= 0:
            return torch.empty(0, 3, device=self.device)
        
        # Step 1: Randomly select candidate targets for each drone
        # 为每个无人机随机选择一个候选目标点
        drone_indices = torch.randint(0, self.num_candidates, (num_drones,), device=self.device)
        selected_targets = self.candidate_targets[drone_indices]  # [num_drones, 3]
        
        # Step 2: Generate random offsets around selected targets
        # 生成随机半径和角度
        radii = torch.rand(num_drones, device=self.device) * (self.max_offset_radius - self.min_offset_radius) + self.min_offset_radius
        angles = torch.rand(num_drones, device=self.device) * 2 * torch.pi
        
        # 计算偏移量 [num_drones, 3]
        # 只在xy平面上添加偏移，z保持不变
        offsets = torch.zeros(num_drones, 3, device=self.device)
        offsets[:, 0] = radii * torch.cos(angles)  # x偏移
        offsets[:, 1] = radii * torch.sin(angles)  # y偏移
        offsets[:, 2] = 0.0  # z偏移为0
        
        # 应用偏移量
        final_targets = selected_targets + offsets
        
        # 确保目标点在边界内
        final_targets[:, 0] = torch.clamp(final_targets[:, 0], 
                                         self.bounds.xmin, self.bounds.xmax)
        final_targets[:, 1] = torch.clamp(final_targets[:, 1], 
                                         self.bounds.ymin, self.bounds.ymax)
        
        return final_targets
    
    def generate_single_target(self) -> torch.Tensor:
        """Generate a single target for one drone."""
        return self.generate_targets(1).squeeze(0)
    
    def get_candidate_targets(self) -> torch.Tensor:
        """Get the current candidate target points.
        
        Returns:
            Tensor of shape [num_candidates, 3] containing candidate targets
        """
        if self.candidate_targets is None:
            raise RuntimeError("Must call initialize_targets() first")
        return self.candidate_targets.clone()
    
    def update_candidate_targets(self, new_candidates: torch.Tensor):
        """Update candidate target points.
        
        Args:
            new_candidates: Tensor of shape [N, 3] containing new candidate targets
        """
        if new_candidates.dim() != 2 or new_candidates.shape[1] != 3:
            raise ValueError("new_candidates must be of shape [N, 3]")
        
        self.candidate_targets = new_candidates.clone()
        self.num_candidates = new_candidates.shape[0]
        print(f"Updated to {self.num_candidates} candidate targets")

class EVTOLTargetGenerator:
    """EVTOL目标和航线生成器"""
    
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.bounds = config.area_bounds
        self.device = device
        self.flight_height = config.flight_height
        self.course_num = config.evtol.course_num
        
        # 预生成的航线
        self.courses = []
        
        # EVTOL飞行参数
        self.max_speed = config.evtol.max_speed
        self.turn_radius = config.evtol.turn_radius
        self.arrival_threshold = config.evtol.arrival_threshold
        self.grid_size = config.area_bounds.grid_size
        
        # 当前正在处理的航线数据
        self.waypoints = []
        self.smooth_waypoints = []
    
    def _generate_random_point(self) -> Tuple[float, float, float]:
        """生成随机位置点"""
        x_range = self.bounds.xmax - self.bounds.xmin
        y_range = self.bounds.ymax - self.bounds.ymin
        
        x = random.uniform(self.bounds.xmin, self.bounds.xmax)
        y = random.uniform(self.bounds.ymin, self.bounds.ymax)
        z = self.flight_height
        
        return (x, y, z)
    
    def _generate_intermediate_waypoints(self, start: Tuple[float, float, float], 
                                       end: Tuple[float, float, float], 
                                       num_points: int = 3) -> List[Tuple[float, float, float]]:
        """在起点和终点之间生成中间航路点"""
        waypoints = []
        
        start_pos = np.array(start)
        end_pos = np.array(end)
        
        for i in range(1, num_points + 1):
            # 基本插值
            alpha = i / (num_points + 1)
            base_point = start_pos + alpha * (end_pos - start_pos)
            
            # 添加随机偏移以创建更有趣的路径
            deviation = min(10.0, np.linalg.norm(end_pos - start_pos) * 0.2)
            offset_x = random.uniform(-deviation, deviation)
            offset_y = random.uniform(-deviation, deviation)
            
            waypoint = (
                base_point[0] + offset_x,
                base_point[1] + offset_y,
                self.flight_height
            )
            waypoints.append(waypoint)
        
        return waypoints
    
    def generate_course_with_smooth_trajectory(self):
        """生成航线并返回平滑后的轨迹"""
        # 随机选择起点和终点
        area_radius_x = self.bounds.xmax - self.bounds.xmin
        area_radius_y = self.bounds.ymax - self.bounds.ymin
        area_radius = max(area_radius_x, area_radius_y)/2.0
        area_center = torch.tensor([(self.bounds.xmin + self.bounds.xmax) / 2, 
                                    (self.bounds.ymin + self.bounds.ymax) / 2, 
                                    self.flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)
        start_tensor = math_utils.sample_cylinder(area_radius, (self.flight_height, self.flight_height), 1, self.device)
        goal_tensor = start_tensor.clone()
        goal_tensor[:,:2] = -goal_tensor[:,:2]
        start_tensor[:,:2] = start_tensor[:,:2] + area_center[:,:2]
        goal_tensor[:,:2] = goal_tensor[:,:2] + area_center[:,:2]
        start = start_tensor.cpu().numpy()[0].astype(np.float64)
        goal = goal_tensor.cpu().numpy()[0].astype(np.float64)
        
        # 生成中间航路点（3-4个点）
        inter_points = math_utils.sample_cylinder(area_radius/2.0, (self.flight_height, self.flight_height), 1, self.device)
        inter_points[:,:2] = inter_points[:,:2] + area_center[:,:2]
        inter_points = inter_points.cpu().numpy()[0].astype(np.float64)
        waypoints = [start, inter_points, goal]
        
        course = {
            "start":start,
            "end":goal,
            "waypoints":waypoints
        }
        
        # 生成基础waypoints并进行平滑处理
        self.set_course(course)
        if len(self.smooth_waypoints) > 0:
            return self.waypoints, self.smooth_waypoints
        else:
            raise ValueError("course is not valid")
            
    def set_course(self, course):
        """
        设置航路点并生成平滑轨迹 - 基于evtol.py的实现
        这里的waypoints需要包含起点终点
        """
        waypoints = course['waypoints']
        
        self.waypoints: List[Waypoint] = []
        
        # 添加起点
        self.waypoints.append(Waypoint(course['start'][0], course['start'][1], self.flight_height, self.max_speed))
        
        if len(waypoints) >= 2:
            # 使用向量夹角判断是否为关键点
            for i in range(0, len(waypoints) - 1):
                # 计算前后两个向量
                current_point = self.waypoints[-1]
                v1 = np.array([waypoints[i][0] - current_point.x, waypoints[i][1] - current_point.y])
                v2 = np.array([waypoints[i+1][0] - waypoints[i][0], waypoints[i+1][1] - waypoints[i][1]])
                
                # 归一化向量
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # 计算夹角余弦值
                    cos_angle = np.dot(v1, v2)
                    # 如果夹角足够大（余弦值小于阈值），认为是关键点
                    if cos_angle < 0.99 and cos_angle > -0.707:  # 约5.7度的阈值
                        self.waypoints.append(Waypoint(waypoints[i][0], waypoints[i][1], self.flight_height, self.max_speed))
            
            # 添加最后一个点
            last_idx = len(waypoints) - 1
            self.waypoints.append(Waypoint(waypoints[last_idx][0], waypoints[last_idx][1], self.flight_height, self.max_speed))
        
        # 判断end是否距离最后一个点足够远
        if np.linalg.norm(np.array([course['end'][0], course['end'][1]]) - np.array([self.waypoints[-1].x, self.waypoints[-1].y])) > self.grid_size:
            self.waypoints.append(Waypoint(course['end'][0], course['end'][1], self.flight_height, self.max_speed))
        
        # 生成平滑轨迹
        if len(self.waypoints) >= 2:
            self.smooth_trajectory()
        else:
            raise ValueError("Waypoints length is less than 2")
            # 按理说应该处理一下self.waypoints，但是应该不会出现这种情况，因为算上起点终点，self.waypoints至少有2个点

        
    def generate_arc_points(self, start, end, center, radius, roll_angle, num_points=10):
        """生成圆弧上的等间隔点，包括姿态和速度信息
        Args:
            start: 圆弧起点
            end: 圆弧终点
            center: 圆弧中心
            radius: 圆弧半径
            roll_angle: roll角度(度)
            num_points: 圆弧上的点数
        Returns:
            arc_points: 圆弧上的点列表
        """
        # 计算圆弧起点和终点向量
        v1 = start - center
        v2 = end - center
        
        # 计算圆弧角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 判断圆弧方向（顺时针或逆时针）
        cross_prod = np.cross(v1[:2], v2[:2])  # 只考虑x-y平面
        if cross_prod < 0:
            angle = -angle
            
        # 生成圆弧上的点
        arc_points = []
        roll_rad = np.radians(roll_angle)
        
        for i in range(num_points + 1):
            t = i / num_points
            # 计算当前角度
            current_angle = t * angle
            
            # 计算旋转后的向量
            cos_t = np.cos(current_angle)
            sin_t = np.sin(current_angle)
            
            # 旋转向量v1得到当前位置
            x = center[0] + cos_t * v1[0] - sin_t * v1[1]
            y = center[1] + sin_t * v1[0] + cos_t * v1[1]
            z = start[2]  # 保持z不变
            
            # 计算前进方向（切线方向）
            tangent_x = -sin_t * v1[0] - cos_t * v1[1]
            tangent_y = cos_t * v1[0] - sin_t * v1[1]
            tangent = np.array([tangent_x, tangent_y, 0])
            tangent = tangent / np.linalg.norm(tangent)
            
            # 计算姿态四元数
            # 前进方向作为x轴
            x_axis = tangent
            
            # 计算z轴（向上）与roll倾斜
            if np.abs(angle) > 1e-6:  # 有转弯角度时添加roll
                # 计算圆心到当前点的向量
                radial = np.array([x - center[0], y - center[1], 0])
                radial = radial / np.linalg.norm(radial)
                
                # z轴应该在与圆心相反的方向倾斜
                z_axis = np.array([0, 0, 1])
                if angle > 0:  # 逆时针转弯，向左倾斜
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    z_axis = np.cos(roll_rad) * z_axis - np.sin(roll_rad) * radial
                else:  # 顺时针转弯，向右倾斜
                    y_axis = np.cross(x_axis, z_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    z_axis = np.cos(roll_rad) * z_axis + np.sin(roll_rad) * radial
            else:
                z_axis = np.array([0, 0, 1])
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
            
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # 重新正交化y轴
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # 构建旋转矩阵
            R = np.column_stack((x_axis, y_axis, z_axis))
            
            # 旋转矩阵转四元数
            qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
            
            # 计算速度
            vel = np.array([self.max_speed, 0, 0])  # 机体坐标系中的速度
            world_vel = R @ vel  # 世界坐标系中的速度
            
            # 创建包含姿态和速度的路径点
            point = Waypoint_ex(
                x=x, y=y, z=z,
                vx=world_vel[0], vy=world_vel[1], vz=world_vel[2],
                qx=qx, qy=qy, qz=qz, qw=qw
            )
            
            arc_points.append(point)
            
        return arc_points
    
    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        # 确保四元数是单位四元数
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        return qw/norm, qx/norm, qy/norm, qz/norm
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """将欧拉角转换为四元数"""
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) + np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        # quaternion = tf.quaternion_from_euler(roll, pitch, yaw) # qx,qy,qz,qw
        # 对比一下两个结果
        #error = np.linalg.norm(np.array([qw - quaternion[3], qx - quaternion[0], qy - quaternion[1], qz - quaternion[2]]))
        #print(f"euler_to_quaternion: {error}")
        return qw, qx, qy, qz
    # 将欧拉角转换为旋转矩阵
    def quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """将四元数转换为旋转矩阵"""
        R = np.eye(3)
        R[0, 0] = 1 - 2*(qy*qy + qz*qz)
        R[0, 1] = 2*(qx*qy - qw*qz)
        R[0, 2] = 2*(qx*qz + qw*qy)
        R[1, 0] = 2*(qx*qy + qw*qz)
        R[1, 1] = 1 - 2*(qx*qx + qz*qz)
        R[1, 2] = 2*(qy*qz - qw*qx)
        R[2, 0] = 2*(qx*qz - qw*qy)
        R[2, 1] = 2*(qy*qz + qw*qx)
        R[2, 2] = 1 - 2*(qx*qx + qy*qy)
        # R_tf = tf.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
        # print(f"quaternion_to_rotation_matrix: {np.linalg.norm(R - R_tf)}")
        return R
        
    def smooth_trajectory(self):
        """生成平滑轨迹"""
        self.smooth_waypoints: List[Waypoint_ex] = []
        if len(self.waypoints) < 2:
            return False
        # 添加第一个点上去
        v1 = np.array([self.waypoints[1].x - self.waypoints[0].x, self.waypoints[1].y - self.waypoints[0].y, self.waypoints[1].z - self.waypoints[0].z])
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 1e-6:
            return False
        v1 = v1 / v1_norm
        yaw = np.arctan2(v1[1], v1[0])
        pitch = np.arctan2(v1[2], np.linalg.norm(v1[:2]))
        roll = 0.0
        # convert euler to R
        qw, qx, qy, qz = self.euler_to_quaternion(roll, pitch, yaw)
        R_matrix = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
        vel = np.array([self.max_speed, 0, 0])
        world_vel = R_matrix @ vel
        wp = Waypoint_ex(
            x=self.waypoints[0].x, y=self.waypoints[0].y, z=self.waypoints[0].z,
            vx=world_vel[0], vy=world_vel[1], vz=world_vel[2],
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        self.smooth_waypoints.append(wp)
        
        if len(self.waypoints) == 2:
            wp = Waypoint_ex(
                x=self.waypoints[1].x, y=self.waypoints[1].y, z=self.waypoints[1].z,
                vx=world_vel[0], vy=world_vel[1], vz=world_vel[2],
                qx=qx, qy=qy, qz=qz, qw=qw
            )
            self.smooth_waypoints.append(wp)
            return True
        # 大于两个点就可以构建过渡路径

        for i in range(len(self.waypoints) - 2):
            start_point = np.array([self.smooth_waypoints[-1].x, 
                                    self.smooth_waypoints[-1].y, 
                                    self.smooth_waypoints[-1].z])
            # v1可能要用start_point来计算
            v1 = np.array([self.waypoints[i+1].x - start_point[0], 
                           self.waypoints[i+1].y - start_point[1], 
                           self.waypoints[i+1].z - start_point[2]])
            v2 = np.array([self.waypoints[i+2].x - self.waypoints[i+1].x, 
                           self.waypoints[i+2].y - self.waypoints[i+1].y, 
                           self.waypoints[i+2].z - self.waypoints[i+1].z])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                continue
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            # 计算转弯点
            second_point = np.array([self.waypoints[i+1].x, 
                                   self.waypoints[i+1].y, 
                                   self.waypoints[i+1].z])
            third_point = np.array([self.waypoints[i+2].x, 
                                   self.waypoints[i+2].y, 
                                   self.waypoints[i+2].z])
            start_turn, end_turn, turn_radius = self.calculate_turning_points(
                start_point, second_point, third_point, self.turn_radius
            )
            self.insert_ex_waypoints_straight(start_point, start_turn, num_points=10)

            # 计算圆弧中心
            self.insert_ex_waypoints_arc(start_turn, end_turn, v1, v2, turn_radius, num_points=20)
            
        # last waypoint
        start_point = np.array([self.smooth_waypoints[-1].x, 
                                self.smooth_waypoints[-1].y, 
                                self.smooth_waypoints[-1].z])
        end_point = np.array([self.waypoints[-1].x, 
                              self.waypoints[-1].y, 
                              self.waypoints[-1].z])
        self.insert_ex_waypoints_straight(start_point, end_point, num_points=10)

    def insert_ex_waypoints_straight(self,start_point, end_point, num_points=10):
        """在直线段上插入点"""
        v1 = end_point - start_point
        v1_norm = np.linalg.norm(v1)

        if v1_norm < 1e-6:
            return False
        
        v1 = v1 / v1_norm
        yaw = np.arctan2(v1[1], v1[0])
        pitch = -np.arctan2(v1[2], np.linalg.norm(v1[:2]))
        roll = 0.0
        qw, qx, qy, qz = self.euler_to_quaternion(roll, pitch, yaw)
        R_matrix = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
        vel = np.array([self.max_speed, 0, 0])
        world_vel = R_matrix @ vel

        if v1_norm < 1.0:
            num_points = 1

        delta_dist = v1_norm / num_points

        for i in range(num_points):
            pos = start_point + delta_dist * (i+1) * v1
            wp = Waypoint_ex(
                x=pos[0], y=pos[1], z=pos[2],
                vx=world_vel[0], vy=world_vel[1], vz=world_vel[2],
                qx=qx, qy=qy, qz=qz, qw=qw
            )
            self.smooth_waypoints.append(wp)
        return True
    def insert_ex_waypoints_arc(self,start_point, end_point, v1_unit, v2_unit, turn_radius=10.0, num_points=10):
        # 判断转弯方向
        turn_dist = np.linalg.norm(end_point - start_point)
        if turn_dist < 1e-6:
            # 说明calculate_turning_points输出的是两个相同位置的点
            return True
        theta_1 = np.arctan2(v1_unit[1], v1_unit[0])
        theta_2 = np.arctan2(v2_unit[1], v2_unit[0])
        delta_yaw = theta_2 - theta_1
        # 将delta_yaw转换到[-pi, pi]范围内
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
        is_turn_left = 1
        if delta_yaw < 0:
            is_turn_left = -1

        # 计算圆心
        dir_start_center = theta_1 + np.pi/2.0 * is_turn_left

        
        circle_center_2D = np.array([start_point[0] + turn_radius * np.cos(dir_start_center),
                                  start_point[1] + turn_radius * np.sin(dir_start_center)])
        
        delta_height = end_point[2] - start_point[2]
        dist_arc = np.abs(turn_radius * delta_yaw)
        pitch = -np.arctan2(delta_height, dist_arc)

        roll = np.arctan2(self.max_speed*self.max_speed, 9.81*turn_radius)
        if end_point[2] < -1.0:
            # NED frame
            roll = is_turn_left * roll
        else:
            # ENU frame
            roll = - is_turn_left * roll


        vel = np.array([self.max_speed, 0, 0])
        yaw = theta_1
        qw, qx, qy, qz = self.euler_to_quaternion(roll, pitch, yaw)
        R_matrix = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
        world_vel = R_matrix @ vel
        self.smooth_waypoints[-1].qx = qx
        self.smooth_waypoints[-1].qy = qy
        self.smooth_waypoints[-1].qz = qz
        self.smooth_waypoints[-1].qw = qw
        for i in range(num_points):
            # delta_yaw是有正负的，所以不同is_turn_left
            pos_2D = circle_center_2D + turn_radius * np.array([np.cos(dir_start_center + np.pi + delta_yaw * (i+1) / num_points),
                                                          np.sin(dir_start_center + np.pi + delta_yaw * (i+1) / num_points)])
            pos = np.array([pos_2D[0], pos_2D[1], start_point[2] + delta_height * (i+1) / num_points])
            yaw = theta_1 + delta_yaw * (i+1) / num_points
            qw, qx, qy, qz = self.euler_to_quaternion(roll, pitch, yaw)
            R_matrix = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
            world_vel = R_matrix @ vel
            wp = Waypoint_ex(
                x=pos[0], y=pos[1], z=pos[2],
                vx=world_vel[0], vy=world_vel[1], vz=world_vel[2],
                qx=qx, qy=qy, qz=qz, qw=qw
            )
            self.smooth_waypoints.append(wp)
        return True

    def calculate_turning_points(self, p1, p2, p3, radius):
        """计算转弯点
        Args:
            p1: 第一个点 (x1, y1, z1)
            p2: 拐角点 (x2, y2, z2)
            p3: 第三个点 (x3, y3, z3)
            radius: 转弯半径
        Returns:
            (start_turn, end_turn): 开始转弯和结束转弯的点
        """
        # TODO:这里是三维的一个判断，如果pitch太大，会有不良行为
        # 计算向量p1->p2和p2->p3
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
        
        # normalize
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            # 如果有两点太近，直接返回p2作为转弯点
            return p2, p2, radius
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # 计算转弯角度(弧度)
        cos_angle = np.dot(v1, v2)
        # 裁剪到[-1, 1]范围内
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 如果角度太小，不需要转弯
        if angle < np.radians(10) or angle > np.radians(170):
            return p2, p2, radius
            
        # 计算圆弧起点和终点到p2的距离
        turn_dist = min(radius * np.tan(angle/2), v1_norm * 0.45, v2_norm * 0.45)

        first_dist = np.linalg.norm(p2 - p1)
        if turn_dist > first_dist:
            turn_dist = first_dist
        second_dist = np.linalg.norm(p2 - p3)
        if turn_dist > second_dist/2.0:
            turn_dist = second_dist/2.0
        turn_radius = turn_dist / np.tan(angle/2)
        
        # 计算圆弧起点和终点
        start_turn = p2 - v1 * turn_dist
        end_turn = p2 + v2 * turn_dist
        
        return start_turn, end_turn, turn_radius