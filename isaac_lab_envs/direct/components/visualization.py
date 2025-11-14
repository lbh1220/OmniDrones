import torch
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils

class VisualizationManager:
    """Unified visualization manager that binds to an env and renders available elements.

    - Uses self.env to access state/cfg/robot attributes
    - Provides modular setup/update for nav and traffic visuals
    - Public APIs: setup_debug_vis(debug_vis: bool), debug_vis_callback(event)
    """

    def __init__(self, env):
        self.env = env

    # ---------- Public API ----------
    def setup_debug_vis(self, debug_vis: bool):
        if self.env.sim.has_gui():
            from omni_drones.envs.isaac_env import DebugDraw
            self.debug_draw = DebugDraw()
        else:
            self.debug_draw = None
        if debug_vis:
            # Nav-related markers (targets / drone pos / local goals / projection)
            self._setup_nav_visuals()
            # Traffic-related markers (traffic positions / drone positions)
            self._setup_traffic_visuals()
        else:
            self._hide_all_visuals()

    def debug_vis_callback(self, event):
        # Update nav visuals if present
        try:
            if self.debug_draw is not None:
                self.debug_draw.clear()
            self._update_nav_visuals()
            # Update traffic visuals if present
            self._update_traffic_visuals()
            self._update_lidar_visuals()
            self._update_planned_path_visuals()
        except Exception as e:
            print(f"Warning: Could not update debug visuals: {e}")
    # ---------- Nav Visuals ----------
    def _setup_nav_visuals(self):
        env = self.env
        cfg = getattr(env, "cfg", None)
        state = getattr(env, "state", None)
        drone = getattr(env, "drone", None)
        if state is None or drone is None:
            return

        # target positions marker
        if not hasattr(self, "target_pos_visualizer"):
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
            marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
            marker_cfg.prim_path = "/Visuals/Command/target_position"
            self.target_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.target_pos_visualizer.set_visibility(True)

        # drone positions marker (green spheres)
        if not hasattr(self, "drone_pos_visualizer") and hasattr(drone, "pos"):
            drone_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/drone_position",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                            opacity=0.5,
                        ),
                    )
                },
            )
            self.drone_pos_visualizer = VisualizationMarkers(drone_marker_cfg)
            self.drone_pos_visualizer.set_visibility(True)

        # local goal & projection (only when using global path)
        use_global_path = bool(getattr(cfg, "use_global_path", False))
        if use_global_path:
            if not hasattr(self, "local_goal_visualizer"):
                local_goal_marker_cfg = CUBOID_MARKER_CFG.copy()
                local_goal_marker_cfg.markers["cuboid"].size = (0.18, 0.18, 0.18)
                local_goal_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                local_goal_marker_cfg.prim_path = "/Visuals/Command/local_goal"
                self.local_goal_visualizer = VisualizationMarkers(local_goal_marker_cfg)
                self.local_goal_visualizer.set_visibility(True)

            if not hasattr(self, "projection_point_visualizer"):
                projection_point_marker_cfg = CUBOID_MARKER_CFG.copy()
                projection_point_marker_cfg.markers["cuboid"].size = (0.12, 0.12, 0.12)
                projection_point_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
                projection_point_marker_cfg.prim_path = "/Visuals/Command/projection_point"
                self.projection_point_visualizer = VisualizationMarkers(projection_point_marker_cfg)
                self.projection_point_visualizer.set_visibility(True)

    def _update_nav_visuals(self):
        env = self.env
        cfg = getattr(env, "cfg", None)
        state = getattr(env, "state", None)
        drone = getattr(env, "drone", None)
        if state is None:
            return

        debug_vis_num = int(getattr(cfg, "debug_vis_num_envs", 10)) if cfg is not None else 10

        # targets
        if hasattr(self, "target_pos_visualizer") and hasattr(state, "navigation"):
            tp = getattr(state.navigation, "target_positions", None)
            if tp is not None and tp.numel() > 0:
                vis_target_pos = tp.squeeze(1)
                if vis_target_pos.shape[0] > debug_vis_num:
                    vis_target_pos = vis_target_pos[:debug_vis_num]
                self.target_pos_visualizer.visualize(vis_target_pos)

        # drone positions
        if hasattr(self, "drone_pos_visualizer") and drone is not None and hasattr(drone, "pos"):
            dp = drone.pos.squeeze(1)
            if dp.shape[0] > debug_vis_num:
                dp = dp[:debug_vis_num]
            scale_shape = torch.tensor([1.0, 1.0, 0.5], device=dp.device, dtype=dp.dtype)
            scales = scale_shape.expand(dp.shape[0], -1)*self.env.cfg.safety_radius
            self.drone_pos_visualizer.visualize(dp, scales=scales)
            # self.drone_pos_visualizer.visualize(dp)

        # local goals & projection points
        use_global_path = bool(getattr(cfg, "use_global_path", False))
        if use_global_path and hasattr(state, "navigation"):
            lg = getattr(state.navigation, "local_goals", None)
            if hasattr(self, "local_goal_visualizer") and lg is not None and lg.numel() > 0:
                lg_vis = lg.squeeze(1)
                if lg_vis.shape[0] > debug_vis_num:
                    lg_vis = lg_vis[:debug_vis_num]
                self.local_goal_visualizer.visualize(lg_vis)

            pj = getattr(state.navigation, "projection_points", None)
            if hasattr(self, "projection_point_visualizer") and pj is not None and pj.numel() > 0:
                pj_vis = pj.squeeze(1)
                if pj_vis.shape[0] > debug_vis_num:
                    pj_vis = pj_vis[:debug_vis_num]
                self.projection_point_visualizer.visualize(pj_vis)

    # ---------- Traffic Visuals ----------
    def _setup_traffic_visuals(self):
        env = self.env
        state = getattr(env, "state", None)
        if state is None or env.traffic_sim is None:
            return

        # traffic positions (red spheres)
        if not hasattr(self, "traffic_visualizer"):
            traffic_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/traffic_pos",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),
                            opacity=0.5,
                        ),
                    )
                },
            )
            self.traffic_visualizer = VisualizationMarkers(traffic_marker_cfg)
            self.traffic_visualizer.set_visibility(True)

        # also show main drone positions in traffic env if absent
        if not hasattr(self, "drone_pos_visualizer") and hasattr(env, "drone") and hasattr(env.drone, "pos"):
            drone_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/drone_position",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                            opacity=0.5,
                        ),
                    )
                },
            )
            self.drone_pos_visualizer = VisualizationMarkers(drone_marker_cfg)
            self.drone_pos_visualizer.set_visibility(True)

    def _update_traffic_visuals(self):
        env = self.env
        state = getattr(env, "state", None)
        if state is None or env.traffic_sim is None:
            return

        positions = getattr(state.traffic, "traffic_positions", None)
        safety_radius = getattr(state.traffic, "traffic_safety_radius", None)
        if positions is not None and positions.numel() > 0 and hasattr(self, "traffic_visualizer"):
            num_agents = positions.shape[0]
            scale_shape = torch.tensor([1.0, 1.0, 0.5], device=positions.device, dtype=positions.dtype)
            scales = scale_shape.expand(num_agents, -1)
            if safety_radius is not None and safety_radius.numel() == num_agents:
                scales = scales * safety_radius.unsqueeze(1)
            self.traffic_visualizer.visualize(translations=positions, scales=scales)


# ---------- 【新】Planned Path Visuals (Lines) ----------
    def _update_planned_path_visuals(self):
        """
        使用 self.debug_draw.plot() 绘制 state.navigation.waypoints 中的路径。
        """
        env = self.env
        cfg = getattr(env, "cfg", None)
        state = getattr(env, "state", None)
        use_global_path = bool(getattr(cfg, "use_global_path", False))
        if not use_global_path:
            return
        if self.debug_draw is None:
            return

        # 安全检查
        if (state is None 
            or not hasattr(state, "navigation")
            or not hasattr(state.navigation, "waypoints")
            or not hasattr(state.navigation, "waypoint_lengths")
            or state.navigation.waypoints is None
            or state.navigation.waypoint_lengths is None):
            return

        debug_vis_num = int(getattr(cfg, "debug_vis_num_envs", 10)) if cfg is not None else 10
        
        all_waypoints = state.navigation.waypoints # [N, MaxW, 3 or 4]
        all_lengths = state.navigation.waypoint_lengths # [N]
        
        num_to_draw = min(env.num_envs, debug_vis_num)
        
        color = (0.5, 0.5, 1.0, 0.8) # 
        size = 2.0
        for i in range(num_to_draw):
            length = int(all_lengths[i].item())
            if length < 2: 
                continue
                
            # 提取 (N, 3) 的点集
            points_tensor = all_waypoints[i, :length, :3]
            
            # 【核心】调用您的 plot 函数
            self.debug_draw.plot(points_tensor, color=color, size=size)


    # ---------- 【新】LiDAR Visuals (Lines) ----------
    def _update_lidar_visuals(self):
        """
        使用 self.debug_draw.vector() 绘制 LiDAR 射线。
        (逻辑来自您的示例代码)
        """
        env = self.env
        
        # 检查 LiDAR 是否存在
        if not hasattr(env, "_lidar") or env._lidar is None:
            return
            
        lidar = env._lidar
        
        try:

            lidar_range = env.cfg.lidar_range 
            
            # 2. 获取 env 0 的 LiDAR 原点
            x = lidar.data.pos_w[0] # Shape (3,)
            
            # 3. 计算所有射线的向量 (从原点到命中点)
            # ray_hits_w[0] shape is (num_rays, 3), e.g., (144, 3)
            v_all_rays = lidar.data.ray_hits_w[0] - x # Shape (144, 3)
            
            # 4. 计算所有射线的实际距离 (模长)
            distances = torch.norm(v_all_rays, dim=-1) # Shape (144,)
            
            # 5. 【核心】创建有效性掩码 (Mask)
            #    条件 1: 必须是有限的数字 (排除 inf)
            #    条件 2: 距离必须小于等于逻辑 lidar_range
            valid_mask = (torch.isfinite(distances)) & (distances <= lidar_range)
            
            # 7. 尝试获取 LiDAR 射线的分辨率 (h, w)
            h, w = (None, None)
            if hasattr(env.cfg, "lidar_resolution"):
                h, w = env.cfg.lidar_resolution
            color = (1.0, 1.0, 0.0, 0.5) # 黄色
            size = 1.0

            # 8. 【核心】根据 resolution 是否存在，执行不同的绘图逻辑
            
            if h is not None and w is not None:
                # --- 方案 A: Resolution 可用，绘制子集 (分环) ---
                
                # Reshape 以恢复空间结构
                v = v_all_rays.reshape(h, w, 3)
                valid_mask_hw = valid_mask.reshape(h, w)

                # 子集 1: v[:, 0] (底环)
                vectors_s1 = v[:, 0]
                mask_s1 = valid_mask_hw[:, 0]
                if mask_s1.any(): # 检查是否至少有一条有效射线
                    valid_vectors_s1 = vectors_s1[mask_s1]
                    valid_origins_s1 = x.expand_as(valid_vectors_s1)
                    self.debug_draw.vector(valid_origins_s1, valid_vectors_s1, color=color, size=size)

                # 子集 2: v[:, -1] (顶环)
                vectors_s2 = v[:, -1]
                mask_s2 = valid_mask_hw[:, -1]
                if mask_s2.any():
                    valid_vectors_s2 = vectors_s2[mask_s2]
                    valid_origins_s2 = x.expand_as(valid_vectors_s2)
                    self.debug_draw.vector(valid_origins_s2, valid_vectors_s2, color=color, size=size)

                # 子集 3: v[:, w//2] (中环)
                vectors_s3 = v[:, w//2]
                mask_s3 = valid_mask_hw[:, w//2]
                if mask_s3.any():
                    valid_vectors_s3 = vectors_s3[mask_s3]
                    valid_origins_s3 = x.expand_as(valid_vectors_s3)
                    self.debug_draw.vector(valid_origins_s3, valid_vectors_s3, color=color, size=size)
            
            else:
                # --- 方案 B: Resolution 不可用，绘制所有有效射线 ---
                
                if valid_mask.any(): # 检查是否至少有一条有效射线
                    # 直接在扁平的张量上应用掩码
                    valid_vectors = v_all_rays[valid_mask]
                    valid_origins = x.expand_as(valid_vectors)
                    
                    # 在一次调用中绘制所有有效的射线
                    self.debug_draw.vector(valid_origins, valid_vectors, color=color, size=size)
        except Exception as e:
            # print(f"Warning: Could not draw LiDAR: {e}")
            pass # 避免在仿真中刷屏

    # ---------- Helpers ----------
    def _hide_all_visuals(self):
        for name in [
            "target_pos_visualizer",
            "drone_pos_visualizer",
            "local_goal_visualizer",
            "projection_point_visualizer",
            "traffic_visualizer",
        ]:
            vis = getattr(self, name, None)
            if vis is not None:
                vis.set_visibility(False)