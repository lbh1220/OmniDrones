from abc import ABC, abstractmethod
from omni.isaac.lab.utils import configclass
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner, GlobalPathPlannerCfg
from isaac_lab_envs.direct.mdp.state import EnvState
import torch
import math

@configclass
class MapManagerCfg:
    # global point cloud config
    point_cloud_resolution: float = 1.0  # step for sampling rays in XY
    point_cloud_top_z: float = 200.0      # ray start Z
    point_cloud_margin: float = 0.0       # optional margin added around terrain bounds

@configclass
class UrbanTerrainCfg:
    terrain_type: str = "plane" # plane, hfdiscrete or some mesh in the future
    prim_path: str = "/World/ground"
    size: tuple[float, float] = (50.0, 50.0)
    num_rows: int = 2
    num_cols: int = 2
    obstacle_width_range: tuple[float, float] = (5.0, 12.0)
    obstacle_height_range: tuple[float, float] = (25.0, 40.0)
    platform_width: float = 0.0
    num_obstacles: int = 4
    border_width: float = 3.0

def convert_urban_terrain_cfg_to_terrain_importer_cfg(input_cfg: UrbanTerrainCfg, seed: int):
    from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
    import omni.isaac.lab.sim as sim_utils
    if input_cfg.terrain_type == "plane":
        return TerrainImporterCfg(
            prim_path=input_cfg.prim_path,
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                )
            )
    elif input_cfg.terrain_type == "hfdiscrete":
        return TerrainImporterCfg(
            prim_path=input_cfg.prim_path,
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=tuple(input_cfg.size),
                border_width=0.0,
                num_rows=input_cfg.num_rows,
                num_cols=input_cfg.num_cols,
                horizontal_scale=0.5,
                vertical_scale=1.0,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={"obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=input_cfg.size,
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=input_cfg.border_width,
                    num_obstacles=input_cfg.num_obstacles,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=tuple(input_cfg.obstacle_width_range),
                    obstacle_height_range=tuple(input_cfg.obstacle_height_range),
                    platform_width=input_cfg.platform_width,

                )},
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )
    else:
        raise ValueError(f"Invalid terrain type: {input_cfg.terrain_type}")


class MapManager(ABC):
    def __init__(self, cfg: MapManagerCfg, env):
        self.cfg = cfg
        self.env = env


    def create_global_point_cloud(self):
        """
        在环境初始化时运行一次，基于场景静态mesh自上而下投射，生成并缓存全局点云（致密栅格）。
        - 使用 warp.raycast_mesh (同 LiDAR) 进行一次性批量射线查询。
        - 使用 cfg.point_cloud_resolution 控制XY采样密度。
        - 射线自 (x, y, top_z) 向下 (0,0,-1) 发射。
        """
        print("INFO: Generating and caching the global static point cloud...")
        # 读取地形总尺寸（生成器模式）
        gen_cfg = self.env.cfg.terrain.terrain_generator
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
        x_coords = torch.linspace(xmin, xmax, num_x, device=self.env.device)
        y_coords = torch.linspace(ymin, ymax, num_y, device=self.env.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # 注意 Isaac Lab 中 x,y 对应关系
        # 组装射线
        num_rays = num_x * num_y
        ray_starts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.full((num_rays,), z_top, device=self.env.device)], dim=-1)
        ray_dirs = torch.tensor([0.0, 0.0, -1.0], device=self.env.device).expand(num_rays, 3)
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
            wp_mesh = convert_to_warp_mesh(plane.vertices, plane.faces, device=self.env.device)
        else:
            mesh_prim = sim_utils.get_first_matching_child_prim("/World/ground", lambda prim: prim.GetTypeName() == "Mesh")
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError("Invalid mesh prim under /World/ground")
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(usd_mesh.GetPointsAttr().Get())
            indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=self.env.device)
        # 运行 raycast（一次性）
        hits, _, _, _ = warp_raycast_mesh(ray_starts, ray_dirs, wp_mesh, max_dist=z_top + 5.0)
        z_hits = hits[:, 2].contiguous()
        # 未命中视为地面 z=0
        inf_mask = torch.isinf(z_hits)
        if torch.any(inf_mask):
            z_hits[inf_mask] = 0.0
        # 缓存点云信息
        self.env.state.map.point_cloud_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).contiguous()
        self.env.state.map.point_cloud_z = z_hits
        self.env.state.map.pc_shape_hw = (num_y, num_x)
        self.env.state.map.pc_bounds = (xmin, xmax, ymin, ymax)
        self.env.state.map.pc_resolution = resolution
        # 同时保留高度图视图（便于快速阈值化）
        self.env.state.map.height_map = z_hits.reshape(num_y, num_x).contiguous()

        print(f"INFO: Global point cloud cached: shape(H,W)=({num_y},{num_x}), bounds=({xmin},{xmax},{ymin},{ymax})")

    def create_occupancy_grid(self):
        bounds = (
            self.env.cfg.area_bounds.xmin,
            self.env.cfg.area_bounds.xmax,
            self.env.cfg.area_bounds.ymin,
            self.env.cfg.area_bounds.ymax,
        )
        height_z = self.env.cfg.flight_height
        grid_size = self.env.cfg.area_bounds.grid_size
        self.env.state.map.occupancy_grid = self.get_occupancy_grid_at_height(bounds=bounds, height_z=height_z, grid_size=grid_size)
        self.env.state.map.grid_bounds = bounds
        self.env.state.map.grid_size = grid_size
        from isaac_lab_envs.utils.map_utils import extend_occupancy_map
        self.env.state.map.extended_occupancy_grid = extend_occupancy_map(self.env.state.map.occupancy_grid, self.env.cfg.safety_radius, grid_size)


    def _create_occupancy_grid_for_traffic(self):
        bounds = (
            self.env.cfg.traffic_sim.area_bounds.xmin,
            self.env.cfg.traffic_sim.area_bounds.xmax,
            self.env.cfg.traffic_sim.area_bounds.ymin,
            self.env.cfg.traffic_sim.area_bounds.ymax,
        )
        height_z = self.env.cfg.traffic_sim.flight_height
        grid_size = self.env.cfg.traffic_sim.area_bounds.grid_size
        occupancy_grid_for_traffic = self.get_occupancy_grid_at_height(bounds=bounds, height_z=height_z, grid_size=grid_size)
        self.env.traffic_sim.update_grid_map(no_extended_grid=occupancy_grid_for_traffic, 
                                        grid_size=grid_size,
                                        bounds=bounds)
    
    def are_positions_safe(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check whether one or a batch of world positions are safe using the extended occupancy grid.

        Args:
            positions: Tensor of shape [N, 3] or [3]. Positions are in world frame (x, y, z).

        Returns:
            Bool tensor of shape [N]: True means the position is safe (not occupied or out-of-bounds),
            False means the position is inside an occupied cell.
        """
        grid = getattr(self.env.state.map, "extended_occupancy_grid", None)
        bounds = getattr(self.env.state.map, "grid_bounds", None)
        grid_size = getattr(self.env.state.map, "grid_size", None)
        if grid is None or bounds is None or grid_size is None:
            # No grid available -> treat as safe
            if positions.ndim == 1:
                return torch.ones(1, dtype=torch.bool, device=self.env.device)
            return torch.ones(positions.shape[0], dtype=torch.bool, device=self.env.device)

        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        # Ensure computations on the same device as the grid
        device = grid.device
        positions = positions.to(device)

        # Extract XY components
        x = positions[:, 0]
        y = positions[:, 1]

        xmin, xmax, ymin, ymax = map(float, bounds)
        gs = float(grid_size)
        H, W = grid.shape[-2], grid.shape[-1]

        # Compute discrete indices without clamping to detect out-of-bounds
        ix = torch.floor((x - xmin) / gs).long()
        iy = torch.floor((y - ymin) / gs).long()

        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        safe = torch.ones(positions.shape[0], dtype=torch.bool, device=device)
        if torch.any(in_bounds):
            ix_in = ix[in_bounds]
            iy_in = iy[in_bounds]
            occupied = grid[iy_in, ix_in]
            safe[in_bounds] = ~occupied

        return safe
    
    # 新API：从全局点云派生任意范围与分辨率的占据网格
    def get_occupancy_grid_at_height(self, bounds: tuple[float, float, float, float], height_z: float, grid_size: float) -> torch.Tensor:
        """
        基于全局点云，返回指定范围(bxmin, bxmax, bymin, bymax)、高度阈值和分辨率的2D占据网格。
        - 若请求范围超出点云范围：超出部分视为未占据（False）。
        - grid_size 必须 >= point_cloud_resolution，方可保证无信息丢失。
        返回张量形状：[1, H, W]，布尔类型。
        """
        assert self.env.state.map.height_map is not None, "Global point cloud not built"
        bxmin, bxmax, bymin, bymax = map(float, bounds)
        pc_xmin, pc_xmax, pc_ymin, pc_ymax = self.env.state.map.pc_bounds
        base_res = float(self.env.state.map.pc_resolution)
        gs = float(grid_size)
        if gs < base_res - 1e-6:
            raise ValueError(f"grid_size({gs}) must be >= point_cloud_resolution({base_res})")
        # 计算输出栅格维度
        out_w = max(1, int(math.ceil((bxmax - bxmin) / gs)))
        out_h = max(1, int(math.ceil((bymax - bymin) / gs)))
        # 计算与全局高度图的重叠区域（索引空间）
        H, W = self.env.state.map.pc_shape_hw
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
            return torch.zeros(1, out_h, out_w, dtype=torch.bool, device=self.env.device)
        # 从全局高度图截取重叠子图
        sub_height = self.env.state.map.height_map[iy0_clamp:iy1_clamp, ix0_clamp:ix1_clamp]
        # 对子图按高度阈值生成占据（inf 表示未命中，应视为未占据，这里比较会为 False）
        occ_sub = sub_height >= float(height_z)
        # 1. 创建最终输出的“画布”，尺寸为 (out_h, out_w)，初始值全为 False
        final_out = torch.zeros(out_h, out_w, dtype=torch.bool, device=self.env.device)

        # 2. 计算重叠区域 `occ_sub` 在 `final_out` 画布中应该占据的像素尺寸
        #    这需要考虑从 base_res 到 gs 的分辨率变化
        sub_h, sub_w = occ_sub.shape
        interp_h = max(1, round(sub_h * base_res / gs))
        interp_w = max(1, round(sub_w * base_res / gs))

        # 3. 将布尔图转换为浮点图，并添加维度以符合 interpolate 的输入要求
        occ_sub_float = occ_sub.float().unsqueeze(0).unsqueeze(0)

        # 4. 使用 'area' 模式将子图重采样到我们刚刚计算出的、正确的中间尺寸
        interpolated_sub = torch.nn.functional.interpolate(occ_sub_float, size=(interp_h, interp_w), mode='area')

        # 5. 将插值结果转换回布尔类型
        out = (interpolated_sub > 0).bool().squeeze(0).squeeze(0)

        # 6. 计算 `out` 这块内容应该被粘贴到 `final_out` 画布的哪个位置
        #    首先计算重叠区域的物理起始点 (sub_xmin, sub_ymin)
        sub_xmin = pc_xmin + ix0_clamp * base_res
        sub_ymin = pc_ymin + iy0_clamp * base_res
        #    然后计算这个物理点在最终输出网格中的索引位置
        paste_x_start = max(0, math.floor((sub_xmin - bxmin) / gs))
        paste_y_start = max(0, math.floor((sub_ymin - bymin) / gs))

        # 7. 【核心修复】定义切片的终点，由 `out` 的实际形状决定
        paste_x_end = paste_x_start + out.shape[1]
        paste_y_end = paste_y_start + out.shape[0]

        # 8. 执行粘贴操作，现在源和目标的尺寸保证匹配
        final_out[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = out
        # out put is [H,W]
        return final_out

