# MIT License
#
# Copyright (c) 2025
#
# A lightweight subclass for EVTOL with a custom USD asset as default.
from __future__ import annotations
import torch
from typing import List, Tuple, Optional
from omni.isaac.core.utils import prims as prim_utils
from pxr import UsdPhysics, PhysxSchema
from .evtol_base import EVTOLBase
from omni_drones.views import RigidPrimView
from omni_drones.robots.robot import ASSET_PATH
class EVTOL_TYPE1(EVTOLBase):
    """
    EVTOL variant whose default asset is a custom USD model.
    Usage:
        ev = EVTOL_TYPE1(device="cuda")
        ev.spawn(..., asset_name=None)  # will use the default USD
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        # Default to repository-local usd if not provided
        self.default_asset = ASSET_PATH + "/usd/evtol_type1.usd"

    def spawn(self, translations: List[Tuple[float, float, float]], 
                prim_paths: List[str],
                scales: Optional[List[Tuple[float, float, float]]] = None,
                asset_name: str = None, # 这里保留接口，但通常不需要传
                enable_collision: bool = True) -> List[str]:
            
            if self.is_created:
                self.logger.warning("EVTOLs already created")
                return []
            
            # 如果没有指定 asset_name，就使用初始化时传入的 usd_path
            
            # 处理缩放默认值
            if not scales:
                scales = [(1.0, 1.0, 1.0)] * len(translations)

            created_prims = []

            for i, (translation, prim_path, scale) in enumerate(zip(translations, prim_paths, scales)):
                try:
                    # 1. 加载 USD 资产
                    # 注意：这里不再支持 "Cube" 等字符串，专门处理 USD
                    prim = prim_utils.create_prim(
                        prim_path=prim_path,
                        usd_path=self.default_asset,
                        translation=translation,
                        scale=scale
                    )
                    
                    if prim:
                        # 2. 设置物理属性 (关键差异点)
                        # 我的baselink中是有物理属性的，取消掉他即可
                        
                        # # 应用刚体 API
                        # rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                        
                        # # 【核心设置】强制开启运动学模式 (Kinematic)
                        # # 这样飞机就会完全听从 set_world_poses 的指令，无视重力，也不会掉下去
                        # rigid_body_api.CreateKinematicEnabledAttr().Set(True)

                        # # 3. 处理碰撞 (关键差异点)
                        # # 我们【不】在 Root 节点应用 CollisionAPI
                        # # 因为你的 USD 内部已经有一个 collision_mesh 子节点负责碰撞了
                        # # 如果在这里加 CollisionAPI，物理引擎会报错找不到 Root 的 Mesh
                        
                        # # 如果用户显式要求禁用碰撞 (enable_collision=False)，
                        # # 我们可以在这里尝试禁用，但这通常比较复杂。
                        # # 对于交通流，默认保持 USD 内部的碰撞设置即可。
                        
                        # # 4. 双重保险：禁用重力 (虽然 Kinematic 已经隐含了这点)
                        # physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        # physx_rigid_body_api.CreateDisableGravityAttr().Set(True)
                        
                        # # 5. 设置质量 (可选，防止意外变成动态刚体时质量为0)
                        # massAPI = UsdPhysics.MassAPI.Apply(prim)
                        # massAPI.CreateMassAttr().Set(10.0)

                        created_prims.append(prim_path)
                    else:
                        self.logger.warning(f"Failed to create USD EVTOL at {prim_path}")

                except Exception as e:
                    self.logger.error(f"Error spawning USD EVTOL at {prim_path}: {e}")
                    continue
            
            self.is_created = True
            self.logger.info(f"Successfully spawned {len(created_prims)} USD EVTOLs")
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
        # 与父类的区别，现在的evtol自己有一个节点base_link,所以需要指定这个节点
        self.evtol_view = RigidPrimView(
            prim_paths_expr=f"{prim_paths_expr}/base_link",
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
    