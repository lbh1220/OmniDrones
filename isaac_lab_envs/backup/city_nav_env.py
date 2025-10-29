from __future__ import annotations

from inspect import BoundArguments
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
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner, GlobalPathPlannerCfg

@dataclass
class NavCityEnvCfg(NavEnvCfg):
    # Use height-field generator terrain with discrete obstacles under /World/ground (global)
    episode_length_s = 500.0
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50.0, 50.0),
            border_width=0.0,
            num_rows=2,
            num_cols=2,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(50.0, 50.0),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=10.0,
                    num_obstacles=3,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 15),
                    obstacle_height_range=(25.0, 40.0),
                    platform_width=0.0,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )





    # observation config
    use_angle_distance_obs: bool = True
    # reward config
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5


    lidar_range: float = 4.0
    lidar_vfov: tuple[float, float] = (-10.0, 20.0)  # degrees
    lidar_resolution: tuple[int, int] = (36, 4)  # horizontal x vertical
    lidar_attach_yaw_only: bool = True


    # global point cloud config
    point_cloud_resolution: float = 1.0  # step for sampling rays in XY
    point_cloud_top_z: float = 200.0      # ray start Z
    point_cloud_margin: float = 0.0       # optional margin added around terrain bounds



    # if use global path
    use_global_path: bool = True
    max_waypoints: int = 20
    lookahead_distance: float = 10.0
    rew_cross_track_coeff: float = 0.0
    rew_cross_track_alpha: float = 1.0
    global_path_planner: GlobalPathPlannerCfg = GlobalPathPlannerCfg(
        algorithm="astar",
        smooth_method="shortcut"
    )

class NavCityEnv(NavEnv):
    cfg: NavCityEnvCfg

    def __init__(self, cfg: NavCityEnvCfg, render_mode: str | None = None, **kwargs):
        # override obs processor before parent init hooks use it
        self.global_path_planner = None
        
        super().__init__(cfg, render_mode, **kwargs)

    def _init_mdp_components(self, cfg: NavEnvCfg):
        """初始化模块化组件"""
        from isaac_lab_envs.direct.mdp.observations import CityNavObservationProcessor,CityNavObservationProcessorWithPath
        from isaac_lab_envs.direct.mdp.rewards import CityNavRewardCalculator,CityNavRewardCalculatorWithPath
        if cfg.use_global_path:
            self.obs_processor = CityNavObservationProcessorWithPath(cfg)
            self.reward_calculator = CityNavRewardCalculatorWithPath(cfg)
        else:
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
        # 应该不用担心顺序的问题，因为现在获取pc的方法是检测的world/ground这个mesh,即使有飞机也不会hit
        self._create_global_point_cloud()

        if self.cfg.use_global_path:
            self.global_path_planner = GlobalPathPlanner(self.cfg.global_path_planner)

        self._create_occupancy_grid()



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
        collision_mask = super()._detect_collisions()
        scan = self.state.perception.lidar_scan
        scan = scan.reshape(self.num_envs, -1)
        # scan in [0, lidar_range]; physical distance = lidar_range - scan
        distances = self.cfg.lidar_range - scan
        # 碰撞条件：任一射线距离 <= safety_radius
        lidar_collision_mask = (distances <= self.cfg.safety_radius).any(dim=1)
        collision_mask = collision_mask | lidar_collision_mask
        # If extended occupancy grid exists, also check collision by occupancy map
        grid = getattr(self.state.map, "extended_occupancy_grid", None)
        bounds = getattr(self.state.map, "grid_bounds", None)
        grid_size = getattr(self.state.map, "grid_size", None)
        if grid is not None and bounds is not None and grid_size is not None:
            # Current drone positions: [num_envs, 1, 3] -> [num_envs, 3]
            positions = self.state.ego_drone.positions
            if positions is not None:
                positions_3d = positions[:, 0, :]
                safe_mask = self._are_positions_safe(positions_3d)
                occ_collision = ~safe_mask
                collision_mask = collision_mask | occ_collision
        return collision_mask

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

        start_tensor = start_tensor.unsqueeze(1)
        goal_tensor = goal_tensor.unsqueeze(1)

        # Prepare outputs
        max_wps = self.state.navigation.waypoints.shape[1]  # reasonable upper bound for A* path points
        waypoints = torch.zeros(num_env, max_wps, 3, device=self.device)
        waypoints_length = torch.zeros(num_env, dtype=torch.long, device=self.device)

        # Ensure planner ready and grid available
        planner = self.global_path_planner if self.cfg.use_global_path else None
        grid = getattr(self.state.map, "extended_occupancy_grid", None)
        bounds = getattr(self.state.map, "grid_bounds", None)
        grid_size = getattr(self.state.map, "grid_size", None)
        if planner is not None and planner.grid_map_np is None:
            # update grid in case re-generated
            planner.update_grid_map(grid, grid_size, bounds)

        # Iterate each env to plan individually
        for i in range(num_env):
            s = start_tensor[i, 0, :2]
            g = goal_tensor[i, 0, :2]

            # Retry sampling if path not found
            max_retries = 10
            path_xy = []
            for _ in range(max_retries):
                if planner is None:
                    break
                path_xy = planner.plan_path((float(s[0].item()), float(s[1].item())),
                                             (float(g[0].item()), float(g[1].item())))
                if len(path_xy) > 1:
                    break
                # resample start/goal around area_center circle if failed
                s_new = math_utils.sample_cylinder(self.circle_radius, (0, 0), 1, self.device)[0]
                g_new = -s_new
                s = s_new + area_center[0]
                g = g_new + area_center[0]

            if len(path_xy) <= 1:
                # fall back to straight-line 2-point path if planner unavailable or fail
                path_xy = [
                    (float(s[0].item()), float(s[1].item())),
                    (float(g[0].item()), float(g[1].item()))
                ]

            # Write into waypoints with fixed altitude
            altitude = float(flight_height)
            n = min(len(path_xy), max_wps)
            if n > 0:
                # vectorized write
                pts = torch.tensor(path_xy[:n], dtype=waypoints.dtype, device=self.device)
                waypoints[i, :n, :2] = pts[:, :2]
                waypoints[i, :n, 2] = altitude
            waypoints_length[i] = n

        from isaac_lab_envs.utils.map_utils import visualize_paths_on_grid
        if (grid is not None and bounds is not None) and (not getattr(self, "_paths_viz_done", False)):
            visualize_paths_on_grid(
                grid_map=self.state.map.occupancy_grid,
                waypoints=waypoints,
                waypoint_lengths=waypoints_length,
                bounds=bounds,
                grid_size=grid_size,
                output_path="planned_paths.png",
                max_trajs=min(10, num_env),
            )
            self._paths_viz_done = True


        return start_tensor, goal_tensor, waypoints, waypoints_length

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
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # 注意 Isaac Lab 中 x,y 对应关系
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

        print(f"INFO: Global point cloud cached: shape(H,W)=({num_y},{num_x}), bounds=({xmin},{xmax},{ymin},{ymax})")

    def _create_occupancy_grid(self):
        bounds = (
            self.cfg.area_bounds.xmin,
            self.cfg.area_bounds.xmax,
            self.cfg.area_bounds.ymin,
            self.cfg.area_bounds.ymax,
        )
        height_z = self.cfg.flight_height
        grid_size = self.cfg.area_bounds.grid_size
        self.state.map.occupancy_grid = self.get_occupancy_grid_at_height(bounds=bounds, height_z=height_z, grid_size=grid_size)
        self.state.map.grid_bounds = bounds
        self.state.map.grid_size = grid_size
        from isaac_lab_envs.utils.map_utils import extend_occupancy_map
        self.state.map.extended_occupancy_grid = extend_occupancy_map(self.state.map.occupancy_grid, self.cfg.safety_radius, grid_size)
        
        if self.cfg.use_global_path:
            if self.global_path_planner is None:
                self.global_path_planner = GlobalPathPlanner(self.cfg.global_path_planner)
            self.global_path_planner.update_grid_map(self.state.map.extended_occupancy_grid, 
                                                    grid_size, bounds=bounds)
        from isaac_lab_envs.utils.map_utils import save_height_map, get_convex_hulls_from_grid, plot_convex_hulls
        save_height_map(self.state.map.extended_occupancy_grid, "extended_occupancy_grid.png")
        hulls = get_convex_hulls_from_grid(self.state.map.extended_occupancy_grid, bounds, grid_size)
        plot_convex_hulls(hulls, "convex_hulls.png")
    
    def _are_positions_safe(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check whether one or a batch of world positions are safe using the extended occupancy grid.

        Args:
            positions: Tensor of shape [N, 3] or [3]. Positions are in world frame (x, y, z).

        Returns:
            Bool tensor of shape [N]: True means the position is safe (not occupied or out-of-bounds),
            False means the position is inside an occupied cell.
        """
        grid = getattr(self.state.map, "extended_occupancy_grid", None)
        bounds = getattr(self.state.map, "grid_bounds", None)
        grid_size = getattr(self.state.map, "grid_size", None)
        if grid is None or bounds is None or grid_size is None:
            # No grid available -> treat as safe
            if positions.ndim == 1:
                return torch.ones(1, dtype=torch.bool, device=self.device)
            return torch.ones(positions.shape[0], dtype=torch.bool, device=self.device)

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
        # 1. 创建最终输出的“画布”，尺寸为 (out_h, out_w)，初始值全为 False
        final_out = torch.zeros(out_h, out_w, dtype=torch.bool, device=self.device)

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

