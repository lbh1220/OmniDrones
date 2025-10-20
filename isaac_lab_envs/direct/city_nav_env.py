from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict

from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
import omni.isaac.lab.utils.math as math_utils
import os

from isaac_lab_envs.direct.nav_env import NavEnv, NavEnvCfg
from isaac_lab_envs.direct.mdp.observations import CityNavObservationProcessor


@dataclass
class NavCityEnvCfg(NavEnvCfg):
    # Use height-field generator terrain with discrete obstacles under /World/ground (global)
    episode_length_s = 500.0
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(20.0, 20.0),
            border_width=20.0,
            num_rows=3,
            num_cols=3,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(20.0, 20.0),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=0.0,
                    num_obstacles=4,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(3.0, 5.0),
                    obstacle_height_range=(30.0, 50.0),
                    platform_width=0.0,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5


    lidar_range: float = 15.0
    lidar_resolution: tuple[int, int] = (36, 4)
    lidar_attach_yaw_only: bool = True
    grid_resolution: float = 0.5
    grid_top_z: float = 100.0
    # global point cloud config
    point_cloud_resolution: float = 0.25  # step for sampling rays in XY
    point_cloud_top_z: float = 100.0      # ray start Z
    point_cloud_margin: float = 0.0       # optional margin added around terrain bounds


class NavCityEnv(NavEnv):
    cfg: NavCityEnvCfg

    def __init__(self, cfg: NavCityEnvCfg, render_mode: str | None = None, **kwargs):
        # override obs processor before parent init hooks use it
        self.global_height_map = None
        self.global_point_cloud_xy = None  # [N, 2] XY of sampled grid
        self.global_point_cloud_z = None   # [N] Z hits (inf for miss)
        self.global_pc_shape_hw = None     # (H, W) for reshaping
        self.global_pc_bounds = None       # (xmin, xmax, ymin, ymax)
        
        super().__init__(cfg, render_mode, **kwargs)

    def _init_mdp_components(self, cfg: NavEnvCfg):
        """初始化模块化组件"""
        from isaac_lab_envs.direct.mdp.observations import CityNavObservationProcessor
        from isaac_lab_envs.direct.mdp.rewards import CityNavRewardCalculator
        self.obs_processor = CityNavObservationProcessor(cfg)
        self.reward_calculator = CityNavRewardCalculator(cfg)

    def _setup_scene(self):
        # Terrain config is already set to generator under /World/ground; parent will create it
        self.cfg.terrain.terrain_generator.seed = self.cfg.seed
        super()._setup_scene()
        # global point cloud will be created after sensors are initialized in _post_init_setup
    
    def _post_init_setup(self):
        super()._post_init_setup()
        # 构建全局点云（一次性）
        try:
            self._create_global_point_cloud()
        except Exception as e:
            print(f"WARNING: Global point cloud generation failed: {e}")

    def _setup_lidar(self):
        lidar_vfov_rad = (
            max(-89.0, self.cfg.lidar_vfov[0]) * math.pi / 180.0,
            min(89.0, self.cfg.lidar_vfov[1]) * math.pi / 180.0
        )
        vertical_ray_angles = torch.linspace(lidar_vfov_rad[0], lidar_vfov_rad[1], self.cfg.lidar_resolution[1])
        mesh_paths = ["/World/ground"]
        ray_caster_cfg = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/{self.cfg.drone_model.capitalize()}_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=self.cfg.lidar_attach_yaw_only,
            pattern_cfg=patterns.BpearlPatternCfg(
                vertical_ray_angles=vertical_ray_angles
            ),
            debug_vis=self.cfg.debug_vis,
            mesh_prim_paths=mesh_paths,
            max_distance=self.cfg.lidar_range,
        )
        self._lidar = RayCaster(ray_caster_cfg)

    def _post_physics_step(self, env_ids: torch.Tensor = None):
        super()._post_physics_step(env_ids)
        # compute lidar scan
        self._lidar.update(self.step_dt)
        try:
            lidar_range = self.cfg.lidar_range
            h, w = self.cfg.lidar_resolution
            hits = self._lidar.data.ray_hits_w
            origins = self._lidar.data.pos_w
            scan = lidar_range - (
                (hits - origins.unsqueeze(1)).norm(dim=-1).clamp_max(lidar_range)
            ).reshape(self.num_envs, 1, h, w)
            self.state.perception.lidar_scan = scan
        except Exception:
            self.state.perception.lidar_scan = torch.zeros(self.num_envs, 1, self.cfg.lidar_resolution[0], self.cfg.lidar_resolution[1], device=self.device)
        collision_mask = self._detect_collisions()
        self.state.collision.collision_mask = collision_mask.clone()

        
    def _detect_collisions(self) -> torch.Tensor:
        """检测与障碍物的碰撞"""
        scan = self.state.perception.lidar_scan
        scan = scan.reshape(self.num_envs, -1)
        # scan in [0, lidar_range]; physical distance = lidar_range - scan
        distances = self.cfg.lidar_range - scan
        # 碰撞条件：任一射线距离 <= safety_radius
        collision_mask = (distances <= self.cfg.safety_radius).any(dim=1)
        return collision_mask


    def _generate_crossing_task(self, num_env: int = 1, flight_height: float = 20.0):
        if num_env <= 0:
            raise ValueError("num_aircraft must be greater than 0")
        area_center = torch.tensor([(self.cfg.area_bounds.xmin + self.cfg.area_bounds.xmax) / 2, 
                                    (self.cfg.area_bounds.ymin + self.cfg.area_bounds.ymax) / 2, 
                                    flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)
        start_tensor = math_utils.sample_cylinder(self.circle_radius, (0, 0), num_env, self.device)
        goal_tensor = start_tensor.clone()
        goal_tensor = -goal_tensor
        start_tensor = start_tensor + area_center
        goal_tensor = goal_tensor + area_center

        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1)
    
    def _generate_crossing_task_with_waypoints(self, num_env: int = 1, flight_height: float = 20.0):
        if num_env <= 0:
            raise ValueError("num_aircraft must be greater than 0")
        area_center = torch.tensor([(self.cfg.area_bounds.xmin + self.cfg.area_bounds.xmax) / 2, 
                                    (self.cfg.area_bounds.ymin + self.cfg.area_bounds.ymax) / 2, 
                                    flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)
        start_tensor = math_utils.sample_cylinder(self.circle_radius, (0, 0), num_env, self.device)
        goal_tensor = start_tensor.clone()
        goal_tensor = -goal_tensor
        start_tensor = start_tensor + area_center
        goal_tensor = goal_tensor + area_center

        inter_points = math_utils.sample_cylinder(self.circle_radius/2.0, (0, 0), num_env, self.device)
        inter_points = inter_points + area_center
        waypoints = torch.cat([start_tensor.unsqueeze(1), inter_points.unsqueeze(1), goal_tensor.unsqueeze(1)], dim=1)

        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), waypoints








    def _create_global_occupancy_map(self):
        """
        在环境初始化时运行一次，生成并缓存整个静态地形的全局高度图。
        """
        # Deprecated in favor of _create_global_point_cloud
        raise NotImplementedError("_create_global_occupancy_map is replaced by _create_global_point_cloud")

    def _create_global_point_cloud(self):
        """
        在环境初始化时运行一次，基于场景静态mesh自上而下投射，生成并缓存全局点云（致密栅格）。
        - 使用 warp.raycast_mesh (同 LiDAR) 进行一次性批量射线查询。
        - 使用 cfg.point_cloud_resolution 控制XY采样密度。
        - 射线自 (x, y, top_z) 向下 (0,0,-1) 发射。
        """
        print("INFO: Generating and caching the global static point cloud...")
        # 读取地形总尺寸（生成器模式）
        gen_cfg = self.cfg.terrain.terrain_generator
        total_width = gen_cfg.num_cols * gen_cfg.size[1]
        total_length = gen_cfg.num_rows * gen_cfg.size[0]
        margin = float(self.cfg.point_cloud_margin)
        # 计算采样范围与网格参数
        xmin = -total_length / 2 - margin
        xmax = total_length / 2 + margin
        ymin = -total_width / 2 - margin
        ymax = total_width / 2 + margin
        resolution = float(self.cfg.point_cloud_resolution)
        z_top = float(self.cfg.point_cloud_top_z)
        # 生成均匀网格
        num_x = max(1, int(math.ceil((xmax - xmin) / resolution)))
        num_y = max(1, int(math.ceil((ymax - ymin) / resolution)))
        x_coords = torch.linspace(xmin, xmax, num_x, device=self.device)
        y_coords = torch.linspace(ymin, ymax, num_y, device=self.device)
        grid_x, grid_y = torch.meshgrid(y_coords, x_coords, indexing="ij")  # 注意 Isaac Lab 中 x,y 对应关系
        # 组装射线
        num_rays = num_x * num_y
        ray_starts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.full((num_rays,), z_top, device=self.device)], dim=-1)
        ray_dirs = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_rays, 3)
        # 独立加载 /World/ground 的 mesh 并进行 raycast（不依赖 RayCaster 实例）
        from omni.isaac.lab.utils.warp import raycast_mesh as warp_raycast_mesh, convert_to_warp_mesh
        from pxr import UsdGeom
        import numpy as np
        import omni.isaac.lab.sim as sim_utils
        # 查找 Mesh prim（若是 Plane 则创建无限平面）
        mesh_prim = sim_utils.get_first_matching_child_prim("/World/ground", lambda prim: prim.GetTypeName() == "Plane")
        if mesh_prim is not None:
            from omni.isaac.lab.terrains.trimesh.utils import make_plane
            plane = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(plane.vertices, plane.faces, device=self.device)
        else:
            mesh_prim = sim_utils.get_first_matching_child_prim("/World/ground", lambda prim: prim.GetTypeName() == "Mesh")
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError("Invalid mesh prim under /World/ground")
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(usd_mesh.GetPointsAttr().Get())
            indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
        # 运行 raycast（一次性）
        hits, _, _, _ = warp_raycast_mesh(ray_starts, ray_dirs, wp_mesh, max_dist=z_top + 5.0)
        z_hits = hits[:, 2].contiguous()
        # 未命中视为地面 z=0
        inf_mask = torch.isinf(z_hits)
        if torch.any(inf_mask):
            z_hits[inf_mask] = 0.0
        # 缓存点云信息
        self.state.map.point_cloud_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).contiguous()
        self.state.map.point_cloud_z = z_hits
        self.state.map.pc_shape_hw = (num_y, num_x)
        self.state.map.pc_bounds = (xmin, xmax, ymin, ymax)
        self.state.map.pc_resolution = resolution
        # 同时保留高度图视图（便于快速阈值化）
        self.state.map.height_map = z_hits.reshape(num_y, num_x).contiguous()
        self._save_height_map(output_path="height_map.png")
        print(f"INFO: Global point cloud cached: shape(H,W)=({num_y},{num_x}), bounds=({xmin},{xmax},{ymin},{ymax})")

    def _save_height_map(self, output_path: str, cmap: str = "viridis") -> None:
        """
        将 state.map.height_map 保存到本地：优先保存为 PNG，若缺依赖则保存为 NPY。
        """
        hm = self.state.map.height_map
        if hm is None:
            print("WARNING: height_map is None; skip saving")
            return
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        arr = hm.detach().cpu().numpy()
        # 优先使用 matplotlib 直接保存彩色图
        try:
            import matplotlib.pyplot as plt
            plt.imsave(output_path, arr, cmap=cmap)
            print(f"INFO: Saved height_map PNG to {output_path}")
            return
        except Exception as e:
            print(f"WARNING: matplotlib save failed ({e}); fallback to PIL/NPY")
        # 尝试 PIL 保存灰度
        try:
            from PIL import Image
            import numpy as np
            vmin = float(arr.min()) if arr.size > 0 else 0.0
            vmax = float(arr.max()) if arr.size > 0 else 1.0
            if vmax <= vmin:
                norm = (arr - vmin)
            else:
                norm = (arr - vmin) / (vmax - vmin)
            img = (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")
            Image.fromarray(img).save(output_path)
            print(f"INFO: Saved height_map PNG (PIL) to {output_path}")
            return
        except Exception as e:
            print(f"WARNING: PIL save failed ({e}); fallback to NPY")
        # 最后退化为 NPY
        try:
            import numpy as np
            np.save(output_path if output_path.endswith('.npy') else output_path + '.npy', arr)
            print(f"INFO: Saved height_map NPY to {output_path}")
        except Exception as e:
            print(f"ERROR: Failed to save height_map to {output_path}: {e}")

    # 新API：从全局点云派生任意范围与分辨率的占据网格
    def get_occupancy_grid_at_height(self, bounds: tuple[float, float, float, float], height_z: float, grid_size: float) -> torch.Tensor:
        """
        基于全局点云，返回指定范围(bxmin, bxmax, bymin, bymax)、高度阈值和分辨率的2D占据网格。
        - 若请求范围超出点云范围：超出部分视为未占据（False）。
        - grid_size 必须 >= point_cloud_resolution，方可保证无信息丢失。
        返回张量形状：[1, H, W]，布尔类型。
        """
        assert self.state.map.height_map is not None, "Global point cloud not built"
        bxmin, bxmax, bymin, bymax = map(float, bounds)
        pc_xmin, pc_xmax, pc_ymin, pc_ymax = self.state.map.pc_bounds
        base_res = float(self.state.map.pc_resolution)
        gs = float(grid_size)
        if gs < base_res - 1e-6:
            raise ValueError(f"grid_size({gs}) must be >= point_cloud_resolution({base_res})")
        # 计算输出栅格维度
        out_w = max(1, int(math.ceil((bxmax - bxmin) / gs)))
        out_h = max(1, int(math.ceil((bymax - bymin) / gs)))
        # 计算与全局高度图的重叠区域（索引空间）
        H, W = self.state.map.pc_shape_hw
        # 全局栅格坐标步长
        step = base_res
        # 将物理坐标映射到全局索引范围
        def world_to_index_x(x):
            return (x - pc_xmin) / step
        def world_to_index_y(y):
            return (y - pc_ymin) / step
        ix0 = math.floor(world_to_index_x(max(bxmin, pc_xmin)))
        ix1 = math.ceil(world_to_index_x(min(bxmax, pc_xmax)))
        iy0 = math.floor(world_to_index_y(max(bymin, pc_ymin)))
        iy1 = math.ceil(world_to_index_y(min(bymax, pc_ymax)))
        # 裁切到全局索引有效范围
        ix0_clamp = max(0, min(W, ix0))
        ix1_clamp = max(0, min(W, ix1))
        iy0_clamp = max(0, min(H, iy0))
        iy1_clamp = max(0, min(H, iy1))
        # 若无重叠，返回全 False
        if ix0_clamp >= ix1_clamp or iy0_clamp >= iy1_clamp:
            return torch.zeros(1, out_h, out_w, dtype=torch.bool, device=self.device)
        # 从全局高度图截取重叠子图
        sub_height = self.state.map.height_map[iy0_clamp:iy1_clamp, ix0_clamp:ix1_clamp]
        # 对子图按高度阈值生成占据（inf 表示未命中，应视为未占据，这里比较会为 False）
        occ_sub = sub_height >= float(height_z)
        # 计算子图的物理范围
        sub_xmin = pc_xmin + ix0_clamp * step
        sub_xmax = pc_xmin + ix1_clamp * step
        sub_ymin = pc_ymin + iy0_clamp * step
        sub_ymax = pc_ymin + iy1_clamp * step
        # 将子图重采样到目标分辨率与尺寸：
        # 简单做法：块最大/任一为True聚合，以保证保守占据。
        # 计算目标网格每个cell在子图索引空间的覆盖区间
        out = torch.zeros(out_h, out_w, dtype=torch.bool, device=self.device)
        for oy in range(out_h):
            y0 = bymin + oy * gs
            y1 = min(bymin + (oy + 1) * gs, bymax)
            # 映射到子图索引
            sy0 = max(0.0, (y0 - sub_ymin) / step)
            sy1 = max(0.0, (y1 - sub_ymin) / step)
            sy0_i = int(math.floor(sy0))
            sy1_i = int(math.ceil(sy1))
            sy0_i = max(0, min(occ_sub.shape[0], sy0_i))
            sy1_i = max(0, min(occ_sub.shape[0], sy1_i))
            if sy0_i >= sy1_i:
                continue
            for ox in range(out_w):
                x0 = bxmin + ox * gs
                x1 = min(bxmin + (ox + 1) * gs, bxmax)
                sx0 = max(0.0, (x0 - sub_xmin) / step)
                sx1 = max(0.0, (x1 - sub_xmin) / step)
                sx0_i = int(math.floor(sx0))
                sx1_i = int(math.ceil(sx1))
                sx0_i = max(0, min(occ_sub.shape[1], sx0_i))
                sx1_i = max(0, min(occ_sub.shape[1], sx1_i))
                if sx0_i >= sx1_i:
                    continue
                if occ_sub[sy0_i:sy1_i, sx0_i:sx1_i].any():
                    out[oy, ox] = True
        return out.unsqueeze(0)

