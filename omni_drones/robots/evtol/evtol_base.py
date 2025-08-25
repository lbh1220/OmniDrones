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
EVTOL基础类，实现外部动力学的EVTOL仿真
不考虑EVTOL的动力学，只在脚本中计算位置和姿态，然后移动到相应位置
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

from omni.isaac.core.utils import prims as prim_utils
from omni.isaac.core.utils import stage as stage_utils
from omni_drones.views import RigidPrimView


class EVTOLBase:
    """
    EVTOL基础类，管理外部动力学的EVTOL
    
    使用RigidPrimView来管理多个EVTOL作为刚体
    不进行动力学仿真，直接通过脚本控制位置和姿态
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.shape = (1,)  # 仿真环境数量，与Isaac Lab保持一致
        
        # EVTOL视图
        self.evtol_view: Optional[RigidPrimView] = None
        
        # 初始化状态
        self.is_created = False
        self.is_initialized = False
        
        # 状态张量
        self.pos = torch.empty(1, 0, 3, device=device)  # [1, N, 3]
        self.rot = torch.empty(1, 0, 4, device=device)  # [1, N, 4] quaternions
        self.vel = torch.empty(1, 0, 6, device=device)  # [1, N, 6] linear + angular
        
        # 用于存储Isaac Sim内置的一些模型名称
        self.available_assets = [
            "Cube",  # 简单立方体作为占位符
            "Sphere",  # 球体
            "Cylinder"  # 圆柱体
        ]
        self.default_asset = "Cube"
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def spawn(self, translations: List[Tuple[float, float, float]], 
              prim_paths: List[str],
              scales: Optional[List[Tuple[float, float, float]]] = None,
              asset_name: str = None,
              enable_collision: bool = False) -> List[str]:
        """
        生成EVTOL primitives
        
        Args:
            translations: 初始位置列表
            prim_paths: USD路径列表
            scales: 缩放比例列表
            asset_name: 使用的资产名称
            enable_collision: 是否启用碰撞检测（默认False，traffic内部不碰撞）
            
        Returns:
            成功创建的prim路径列表
        """
        if self.is_created:
            self.logger.warning("EVTOLs already created")
            return []
        
        if not translations or not prim_paths:
            self.logger.info("No EVTOLs to create")
            self.is_created = True
            return []
        
        if len(translations) != len(prim_paths):
            raise ValueError("Number of translations must match number of prim_paths")
        
        if scales and len(scales) != len(translations):
            raise ValueError("Number of scales must match number of translations")
        
        if not scales:
            scales = [(1.0, 1.0, 1.0)] * len(translations)
        
        if not asset_name:
            asset_name = self.default_asset
        
        created_prims = []
        
        # 为每个EVTOL创建primitive
        for i, (translation, prim_path, scale) in enumerate(zip(translations, prim_paths, scales)):
            try:
                # 创建基础几何体作为EVTOL表示
                if asset_name == "Cube":
                    prim = prim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type="Cube",
                        translation=translation,
                        scale=scale
                    )
                elif asset_name == "Sphere":
                    prim = prim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type="Sphere", 
                        translation=translation,
                        scale=scale
                    )
                elif asset_name == "Cylinder":
                    prim = prim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type="Cylinder",
                        translation=translation,
                        scale=scale
                    )
                else:
                    # 尝试作为USD资产加载
                    prim = prim_utils.create_prim(
                        prim_path=prim_path,
                        usd_path=asset_name,
                        translation=translation,
                        scale=scale
                    )
                
                if prim:
                    # 添加物理组件使其成为rigid body
                    from pxr import UsdPhysics
                    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    
                    # 根据需要设置碰撞
                    if enable_collision:
                        UsdPhysics.CollisionAPI.Apply(prim)
                    else:
                        # traffic 内部不碰撞：不添加 CollisionAPI 或禁用碰撞
                        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                        prim.GetAttribute("physics:collisionEnabled").Set(False)
                    
                    # 设置基本物理属性
                    massAPI = UsdPhysics.MassAPI.Apply(prim)
                    massAPI.CreateMassAttr().Set(10.0)  # 10kg质量
                    
                    # 禁用重力影响
                    from pxr import PhysxSchema
                    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_rigid_body_api.CreateDisableGravityAttr().Set(True)
                    
                    created_prims.append(prim_path)
                    self.logger.debug(f"Created EVTOL primitive at {prim_path}")
                else:
                    self.logger.warning(f"Failed to create EVTOL primitive at {prim_path}")
                    
            except Exception as e:
                self.logger.error(f"Error creating EVTOL at {prim_path}: {e}")
                continue
        
        self.is_created = True
        self.logger.info(f"Successfully created {len(created_prims)} EVTOL primitives")
        return created_prims
    
    def initialize(self, prim_paths_expr: str):
        """
        初始化EVTOL视图
        
        Args:
            prim_paths_expr: USD路径表达式，如"/World/Traffic/evtol_*"
        """
        if self.is_initialized:
            self.logger.warning("EVTOLs already initialized")
            return
        
        if not self.is_created:
            self.logger.error("EVTOLs not created yet. Call spawn() first.")
            return
        
        # 创建RigidPrimView
        self.evtol_view = RigidPrimView(
            prim_paths_expr=prim_paths_expr,
            reset_xform_properties=False
        )
        
        # 初始化视图
        self.evtol_view.initialize()
        
        # 获取EVTOL数量并初始化状态张量
        num_evtols = self.evtol_view.count
        
        self.pos = torch.zeros(1, num_evtols, 3, device=self.device)
        self.rot = torch.zeros(1, num_evtols, 4, device=self.device)
        self.rot[:, :, 0] = 1.0  # 初始化为单位四元数 [w, x, y, z]
        self.vel = torch.zeros(1, num_evtols, 6, device=self.device)
        
        # 获取初始位置和姿态
        self.pos[:], self.rot[:] = self.get_world_poses(clone=True)
        
        self.is_initialized = True
        self.logger.info(f"Initialized {num_evtols} EVTOLs")
    
    def get_world_poses(self, clone: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取世界坐标系下的位置和姿态
        
        Args:
            clone: 是否克隆张量
            
        Returns:
            位置张量 [1, N, 3] 和姿态张量 [1, N, 4]
        """
        if not self.is_initialized:
            return self.pos, self.rot
        
        positions, rotations = self.evtol_view.get_world_poses(clone=clone)
        
        # 确保形状为 [1, N, 3] 和 [1, N, 4]
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if rotations.dim() == 2:
            rotations = rotations.unsqueeze(0)
        
        return positions, rotations
    
    def set_world_poses(self, positions: torch.Tensor, orientations: torch.Tensor):
        """
        设置世界坐标系下的位置和姿态
        
        Args:
            positions: 位置张量 [1, N, 3] 或 [N, 3]
            orientations: 姿态张量 [1, N, 4] 或 [N, 4] (quaternions)
        """
        if not self.is_initialized:
            self.logger.warning("EVTOLs not initialized yet")
            return
        
        self.evtol_view.set_world_poses(positions, orientations)
    
    def get_velocities(self, clone: bool = True) -> torch.Tensor:
        """
        获取速度 (线速度 + 角速度)
        
        Args:
            clone: 是否克隆张量
            
        Returns:
            速度张量 [1, N, 6]
        """
        if not self.is_initialized:
            return self.vel
        
        velocities = self.evtol_view.get_velocities(clone=clone)
        
        # 确保形状为 [1, N, 6]
        if velocities.dim() == 2:
            velocities = velocities.unsqueeze(0)
        
        return velocities
    
    def set_velocities(self, velocities: torch.Tensor):
        """
        设置速度
        
        Args:
            velocities: 速度张量 [1, N, 6] 或 [N, 6]
        """
        if not self.is_initialized:
            self.logger.warning("EVTOLs not initialized yet")
            return
        
        # 确保形状正确
        if velocities.dim() == 3:
            velocities = velocities.squeeze(0)  # [N, 6]
        
        self.evtol_view.set_velocities(velocities)
        
        # 更新内部状态
        self.vel = velocities.unsqueeze(0)  # [1, N, 6]
    
    def get_linear_velocities(self) -> torch.Tensor:
        """获取线速度 [1, N, 3]"""
        return self.vel[:, :, :3]
    
    def get_angular_velocities(self) -> torch.Tensor:
        """获取角速度 [1, N, 3]"""
        return self.vel[:, :, 3:]
    
    def update_states(self):
        """更新所有状态信息"""
        if not self.is_initialized:
            return
        
        self.pos[:], self.rot[:] = self.get_world_poses(clone=True)
        self.vel[:] = self.get_velocities(clone=True)

    def reset_positions(self, positions: torch.Tensor, rotations: Optional[torch.Tensor] = None):
        """
        重置EVTOL位置和姿态
        
        Args:
            positions: 新位置 [N, 3]
            rotations: 新姿态 [N, 4]，如果为None则保持当前姿态
        """
        if not self.is_initialized:
            self.logger.warning("EVTOLs not initialized yet")
            return
        
        if rotations is None:
            rotations = self.rot.squeeze(0)  # 使用当前姿态
        
        self.set_world_poses(positions, rotations)
        
        # 重置速度
        zero_velocities = torch.zeros(positions.shape[0], 6, device=self.device)
        self.set_velocities(zero_velocities)
    
    @property
    def count(self) -> int:
        """获取EVTOL数量"""
        if self.evtol_view is not None:
            return self.evtol_view.count
        return 0
    
    @property
    def is_valid(self) -> bool:
        """检查EVTOL是否有效"""
        return self.is_initialized and self.evtol_view is not None
    
    def cleanup(self):
        """清理资源"""
        if self.evtol_view is not None:
            del self.evtol_view
            self.evtol_view = None
        
        self.is_initialized = False
        self.is_created = False
        
        self.logger.info("EVTOL cleanup complete")
