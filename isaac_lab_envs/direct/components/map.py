from abc import ABC, abstractmethod
from omni.isaac.lab.utils import configclass
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner, GlobalPathPlannerCfg
from isaac_lab_envs.direct.mdp.state import EnvState
import torch
import math
import os
import pickle

@configclass
class MapManagerCfg:
    # global point cloud config
    point_cloud_resolution: float = 1.0  # step for sampling rays in XY
    point_cloud_top_z: float = 200.0      # ray start Z
    point_cloud_margin: float = 0.0       # optional margin added around terrain bounds

@configclass
class UrbanTerrainCfg:
    # 修改默认值为 "usd" 以便测试，或者保持 "plane"
    terrain_type: str = "usd"  # Options: "plane", "hfdiscrete", "usd"
    
    # 新增：你的城市 USD 文件的绝对路径
    usd_path: str = "/home/liang/Projects/isaac_sim_projects/Brushify/mini_center_city_layout.usd" 
    
    # 保持原有的 prim_path，通常是 "/World/ground"
    # Isaac Lab 会把你的城市加载在这个路径下面
    prim_path: str = "/World/ground"
    
    # 下面这些参数在使用 usd 模式时可能暂时用不到，但保留着无妨
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
    # === 新增：处理 USD 类型的地形 ===
    elif input_cfg.terrain_type == "usd":
        return TerrainImporterCfg(
                prim_path=input_cfg.prim_path,
                terrain_type="plane", # <--- 回归最简模式
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
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
        初始化地图数据。
        - 如果是 USD 模式：加载预计算的 voxel pkl 文件。
        - 其他模式：运行实时 Raycast 生成 height map。
        """
        terrain_type = self.env.cfg.urban_terrain.terrain_type
        
        if terrain_type == "usd":
            # === 分支 1: 加载预处理的 Voxel Map ===
            print(f"INFO: [MapManager] Loading pre-computed voxel map for USD terrain...")
            self._load_usd_voxel_map()
        else:
            # === 分支 2: 传统的 Raycast 生成 Height Map ===
            self._create_raycast_height_map()

    def _load_usd_voxel_map(self):
        """
        加载与 USD 文件同名的 _voxel.pkl 文件，并初始化 env.state.map
        """
        usd_path = self.env.cfg.urban_terrain.usd_path
        base_name = os.path.splitext(usd_path)[0]
        pkl_path = base_name + "_voxel.pkl"

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Voxel map not found at {pkl_path}. Please run the voxelizer tool first.")

        print(f"INFO: [MapManager] Loading voxel data from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 1. 提取元数据
        bounds_min = data["bounds_min"] # [min_x, min_y, min_z]
        bounds_max = data["bounds_max"] # [max_x, max_y, max_z]
        resolution = float(data["resolution"])
        
        # 2. 提取点云 (N, 3) 或 (N, 4)
        np_points = data["point_cloud"]
        
        # 3. 转为 Tensor 并移至 GPU
        # 我们主要关心 XYZ 坐标用于碰撞检测
        # 注意：这里我们保存的是稀疏的 3D 点，而不是密集的 2D grid
        voxel_points = torch.tensor(np_points[:, :3], dtype=torch.float32, device=self.env.device)
        
        # 4. 存入 env.state.map
        # 为了兼容性，我们尽可能填充相关字段，但核心是 voxel_points
        self.env.state.map.voxel_points = voxel_points
        self.env.state.map.pc_resolution = resolution
        self.env.state.map.pc_bounds = (bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1])
        
        # 记录一下 Z 范围，虽然后续逻辑主要靠 query
        self.env.state.map.z_min = float(bounds_min[2])
        self.env.state.map.z_max = float(bounds_max[2])

        # *可选*: 如果你还需要兼容旧的 self.env.state.map.height_map (2.5D DSM)
        # 可以写一个逻辑把 3D 点投影成 2D Max-Height Map，防止某些可视化代码报错
        # 但既然你打算重写 get_occupancy_grid，这里可以先置空或者做个简单的 placeholder
        self.env.state.map.height_map = None 
        
        print(f"INFO: Voxel map loaded. Points: {voxel_points.shape[0]}, Resolution: {resolution}m")

    def _create_raycast_height_map(self):
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
    
    def get_occupancy_grid_at_height(self, bounds: tuple[float, float, float, float], height_z: float, grid_size: float) -> torch.Tensor:
        """
        根据指定的 3D 范围和分辨率生成 2D 占据网格。
        自动兼容 USD Voxel 模式 (3D Slicing) 和 传统 HeightMap 模式 (2.5D Thresholding)。
        """
        bxmin, bxmax, bymin, bymax = map(float, bounds)
        
        # 计算输出网格尺寸
        out_w = max(1, int(math.ceil((bxmax - bxmin) / float(grid_size))))
        out_h = max(1, int(math.ceil((bymax - bymin) / float(grid_size))))
        
        # === 模式 A: 3D Voxel 模式 (USD) ===
        if hasattr(self.env.state.map, "voxel_points") and self.env.state.map.voxel_points is not None:
            points = self.env.state.map.voxel_points # Tensor (N, 3)
            res = float(self.env.state.map.pc_resolution)
            
            # 1. 高度过滤 (Z Slicing)
            # 我们的 voxel 代表该空间被占据。如果无人机在 height_z 飞行，
            # 我们需要检查是否有 voxel 覆盖了这个高度。
            # Voxel 中心为 z，范围是 [z - res/2, z + res/2]
            # 只要 voxel 范围包含 height_z，或者两者距离小于 res 即可视为碰撞风险
            z_threshold = res / 2.0 + 0.1 # 加一点 epsilon
            mask_z = torch.abs(points[:, 2] - float(height_z)) <= z_threshold
            
            # 2. XY 范围过滤
            # 为了加速，先筛选 Z，通常 Z 轴截面点数会少很多
            subset = points[mask_z]
            
            if subset.shape[0] == 0:
                return torch.zeros(out_h, out_w, dtype=torch.bool, device=self.env.device)
            
            mask_xy = (subset[:, 0] >= bxmin) & (subset[:, 0] <= bxmax) & \
                      (subset[:, 1] >= bymin) & (subset[:, 1] <= bymax)
            subset = subset[mask_xy]
            
            if subset.shape[0] == 0:
                return torch.zeros(1, out_h, out_w, dtype=torch.bool, device=self.env.device)
            
            # 3. 投影到 2D Grid
            # 计算索引
            idx_x = torch.floor((subset[:, 0] - bxmin) / float(grid_size)).long()
            idx_y = torch.floor((subset[:, 1] - bymin) / float(grid_size)).long()
            
            # 边界保护 (以防浮点误差导致 idx 越界)
            idx_x = torch.clamp(idx_x, 0, out_w - 1)
            idx_y = torch.clamp(idx_y, 0, out_h - 1)
            
            # 4. 填充网格
            # 创建全 False 网格
            final_out = torch.zeros(out_h, out_w, dtype=torch.bool, device=self.env.device)
            # 将占据位置设为 True
            final_out[idx_y, idx_x] = True
            
            return final_out
            
        # === 模式 B: 2.5D Height Map 模式 (Legacy) ===
        else:
            # 原有的逻辑：从全局 height_map 切片并插值
            assert self.env.state.map.height_map is not None, "Global point cloud not built"
            
            pc_xmin, pc_xmax, pc_ymin, pc_ymax = self.env.state.map.pc_bounds
            base_res = float(self.env.state.map.pc_resolution)
            gs = float(grid_size)
            
            # 全局栅格坐标步长与索引映射
            def world_to_index_x(x): return (x - pc_xmin) / base_res
            def world_to_index_y(y): return (y - pc_ymin) / base_res
            
            ix0 = math.floor(world_to_index_x(max(bxmin, pc_xmin)))
            ix1 = math.ceil(world_to_index_x(min(bxmax, pc_xmax)))
            iy0 = math.floor(world_to_index_y(max(bymin, pc_ymin)))
            iy1 = math.ceil(world_to_index_y(min(bymax, pc_ymax)))
            
            H, W = self.env.state.map.pc_shape_hw
            ix0_clamp = max(0, min(W, ix0))
            ix1_clamp = max(0, min(W, ix1))
            iy0_clamp = max(0, min(H, iy0))
            iy1_clamp = max(0, min(H, iy1))
            
            if ix0_clamp >= ix1_clamp or iy0_clamp >= iy1_clamp:
                return torch.zeros(1, out_h, out_w, dtype=torch.bool, device=self.env.device)
            
            # 截取与阈值化 (2.5D 假设：高度 > height_z 即为障碍)
            sub_height = self.env.state.map.height_map[iy0_clamp:iy1_clamp, ix0_clamp:ix1_clamp]
            occ_sub = sub_height >= float(height_z)
            
            # 插值重采样逻辑 (保持你原有的 robust 实现)
            final_out = torch.zeros(out_h, out_w, dtype=torch.bool, device=self.env.device)
            sub_h, sub_w = occ_sub.shape
            interp_h = max(1, round(sub_h * base_res / gs))
            interp_w = max(1, round(sub_w * base_res / gs))
            
            occ_sub_float = occ_sub.float().unsqueeze(0).unsqueeze(0)
            interpolated_sub = torch.nn.functional.interpolate(occ_sub_float, size=(interp_h, interp_w), mode='area')
            out = (interpolated_sub > 0).bool().squeeze(0).squeeze(0)
            
            sub_xmin = pc_xmin + ix0_clamp * base_res
            sub_ymin = pc_ymin + iy0_clamp * base_res
            
            paste_x_start = max(0, math.floor((sub_xmin - bxmin) / gs))
            paste_y_start = max(0, math.floor((sub_ymin - bymin) / gs))
            paste_x_end = paste_x_start + out.shape[1]
            paste_y_end = paste_y_start + out.shape[0]
            
            # 边界保护
            target_h, target_w = final_out.shape
            # 裁剪源 out 如果它超出了目标边界
            if paste_x_end > target_w:
                out = out[:, :-(paste_x_end - target_w)]
                paste_x_end = target_w
            if paste_y_end > target_h:
                out = out[:-(paste_y_end - target_h), :]
                paste_y_end = target_h
                
            final_out[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = out
            return final_out
