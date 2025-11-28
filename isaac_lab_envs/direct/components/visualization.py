import torch
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.markers import FRAME_MARKER_CFG
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

        # drone frame marker (current pose frames)
        if not hasattr(self, "drone_frame_visualizer"):
            frame_cfg = FRAME_MARKER_CFG.copy()
            frame_cfg.prim_path = "/Visuals/Command/drone_frame"
            self.drone_frame_visualizer = VisualizationMarkers(frame_cfg)
            self.drone_frame_visualizer.set_visibility(True)

        # drone velocity arrows (green arrows along +x scaled by speed, match ego drone color scheme)
        if not hasattr(self, "drone_vel_visualizer"):
            drone_vel_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            drone_vel_cfg.prim_path = "/Visuals/Command/drone_velocity"
            self.drone_vel_visualizer = VisualizationMarkers(drone_vel_cfg)
            self.drone_vel_visualizer.set_visibility(True)

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
        if hasattr(self, "drone_pos_visualizer") and state is not None and hasattr(state.ego_drone, "positions"):
            dp = state.ego_drone.positions.squeeze(1)
            if dp.shape[0] > debug_vis_num:
                dp = dp[:debug_vis_num]
            scale_shape = torch.tensor([1.0, 1.0, 0.5], device=dp.device, dtype=dp.dtype)
            scales = scale_shape.expand(dp.shape[0], -1)*self.env.cfg.safety_radius
            self.drone_pos_visualizer.visualize(dp, scales=scales)
            # self.drone_pos_visualizer.visualize(dp)

        # drone velocities (arrows)
        if hasattr(self, "drone_vel_visualizer") and hasattr(state.ego_drone, "velocities") and state.ego_drone.velocities is not None:
            dp_full = state.ego_drone.positions.squeeze(1)
            vv_full = state.ego_drone.velocities.squeeze(1)
            if dp_full.shape[0] > debug_vis_num:
                dp = dp_full[:debug_vis_num]
                vv = vv_full[:debug_vis_num]
            else:
                dp = dp_full
                vv = vv_full
            if dp.numel() > 0 and vv.numel() > 0:
                # optional: visualize only XY components to avoid vertical "spikes"
                use_xy_only = bool(getattr(cfg, "debug_vis_velocity_use_xy_only", True)) if cfg is not None else True
                if use_xy_only:
                    vv = vv.clone()
                    vv[:, 2] = 0.0
                vel_scale = float(getattr(cfg, "debug_vis_velocity_scale", 1.0)) if cfg is not None else 1.0
                speeds = torch.norm(vv, dim=-1, keepdim=True)  # [M,1]
                # orientations from +x to velocity direction
                orientations = self._quat_align_x_to_vectors(vv)
                # thickness on Y/Z to avoid too thin "blade" look
                # thickness = float(getattr(cfg, "debug_vis_arrow_thickness", 0.2)) if cfg is not None else 0.2
                # thickness_col = torch.full_like(speeds, thickness)
                scales = torch.cat([torch.clamp(speeds * vel_scale, min=0.0), 
                                    env.cfg.env_scale*torch.ones_like(speeds)*5, 
                                    env.cfg.env_scale*torch.ones_like(speeds)], dim=-1)
                # scales = scales * self.env.cfg.env_scale
                self.drone_vel_visualizer.visualize(translations=dp, orientations=orientations, scales=scales)

        # drone current pose frames
        if hasattr(self, "drone_frame_visualizer") and hasattr(state.ego_drone, "positions") and hasattr(state.ego_drone, "rotations"):
            pos_full = state.ego_drone.positions.squeeze(1)  # [N,3]
            quat_full = state.ego_drone.rotations.squeeze(1)  # [N,4] (w,x,y,z)
            count = pos_full.shape[0]
            if count > 0 and quat_full is not None and quat_full.shape[0] == count:
                if count > debug_vis_num:
                    pos = pos_full[:debug_vis_num]
                    quat = quat_full[:debug_vis_num]
                else:
                    pos = pos_full
                    quat = quat_full
                # optional scale for frame gizmo size
                s = self.env.cfg.safety_radius*3
                scales = torch.tensor([s, s, s], device=pos.device, dtype=pos.dtype).expand(pos.shape[0], 3)
                self.drone_frame_visualizer.visualize(translations=pos, orientations=quat, scales=scales)

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

        # traffic velocities (red arrows)
        if not hasattr(self, "traffic_vel_visualizer"):
            traffic_vel_cfg = RED_ARROW_X_MARKER_CFG.copy()
            traffic_vel_cfg.prim_path = "/Visuals/Command/traffic_velocity"
            self.traffic_vel_visualizer = VisualizationMarkers(traffic_vel_cfg)
            self.traffic_vel_visualizer.set_visibility(True)

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

        # traffic velocities (arrows, keep traffic color scheme: red)
        velocities = getattr(state.traffic, "traffic_velocities", None) if getattr(state, "traffic", None) is not None else None
        if velocities is not None and positions is not None and velocities.numel() > 0 and positions.numel() > 0 and hasattr(self, "traffic_vel_visualizer"):
            # Ensure same count
            count = min(positions.shape[0], velocities.shape[0])
            pos = positions[:count]
            vel = velocities[:count]
            cfg = getattr(env, "cfg", None)
            # optional: XY only to keep arrows in ground plane
            use_xy_only = bool(getattr(cfg, "debug_vis_velocity_use_xy_only", True)) if cfg is not None else True
            if use_xy_only:
                vel = vel.clone()
                vel[:, 2] = 0.0
            vel_scale = float(getattr(cfg, "debug_vis_velocity_scale", 1.0)) if (cfg is not None) else 1.0
            speeds = torch.norm(vel, dim=-1, keepdim=True)
            orientations = self._quat_align_x_to_vectors(vel)
            # thickness = float(getattr(cfg, "debug_vis_arrow_thickness", 0.2)) if cfg is not None else 0.2
            # thickness_col = torch.full_like(speeds, thickness)
            # scales = torch.cat([torch.clamp(speeds * vel_scale, min=0.0), thickness_col, thickness_col], dim=-1)
            scales = torch.cat([torch.clamp(speeds * vel_scale, min=0.0), 
                                env.cfg.env_scale*torch.ones_like(speeds)*5,  # 使这个箭头不要太扁
                                env.cfg.env_scale*torch.ones_like(speeds)], dim=-1)
            self.traffic_vel_visualizer.visualize(translations=pos, orientations=orientations, scales=scales)


# ---------- 【新】Planned Path Visuals (Lines) ----------
    def _update_planned_path_visuals(self):
        """
        使用 self.debug_draw.plot() 绘制 state.navigation.waypoints 中的路径。
        """
        if self.debug_draw is None:
            return
        env = self.env
        cfg = getattr(env, "cfg", None)
        state = getattr(env, "state", None)

        use_global_path = bool(getattr(cfg, "use_global_path", False))

        can_draw_ego_global_path = (
            use_global_path and
            (state is not None) and
            (hasattr(state, "navigation") and hasattr(state.navigation, "waypoints") and hasattr(state.navigation, "waypoint_lengths")) and
            (state.navigation.waypoints is not None) and
            (state.navigation.waypoint_lengths is not None)
        )

        debug_vis_num = int(getattr(cfg, "debug_vis_num_envs", 10)) if cfg is not None else 10
        if can_draw_ego_global_path:
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

        # ---------- 追加：绘制 traffic EVTOL 的全局路径 ----------
        env = self.env
        traffic_sim = getattr(env, "traffic_sim", None)
        evtol_manager = getattr(traffic_sim, "evtol_manager", None) if traffic_sim is not None else None
        state_evtol = getattr(evtol_manager, "state", None) if evtol_manager is not None else None
        can_draw_evtol_paths = (
            (self.debug_draw is not None) and
            (traffic_sim is not None) and
            (evtol_manager is not None) and
            (getattr(evtol_manager, "num_evtols", 0) > 0) and
            (state_evtol is not None) and
            hasattr(state_evtol, "waypoints") and
            hasattr(state_evtol, "waypoint_lengths") and
            (getattr(state_evtol, "waypoints", None) is not None) and
            (getattr(state_evtol, "waypoint_lengths", None) is not None)
        )
        if can_draw_evtol_paths:
            # 使用红色绘制 EVTOL 的路径
            color_evtol = (1.0, 0.0, 0.0, 0.8)
            size_evtol = 2.0
            waypoints_evtol = state_evtol.waypoints  # [Ne, M, 4]
            lengths_evtol = state_evtol.waypoint_lengths  # [Ne]
            num_evtols = int(getattr(evtol_manager, "num_evtols", waypoints_evtol.shape[0] if waypoints_evtol is not None else 0))
            for i in range(num_evtols):
                if i >= waypoints_evtol.shape[0]:
                    break
                length = int(lengths_evtol[i].item()) if lengths_evtol is not None and lengths_evtol.numel() > i else 0
                if length < 2:
                    continue
                pts = waypoints_evtol[i, :length, :3]
                self.debug_draw.plot(pts, color=color_evtol, size=size_evtol)


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

    @staticmethod
    def _quat_align_x_to_vectors(vectors: torch.Tensor) -> torch.Tensor:
        """
        Build quaternions (w, x, y, z) that rotate +X axis to each given vector direction.
        For zero-length vectors, returns identity. For opposite direction, uses 180 deg about +Z.
        Args:
            vectors: [M, 3] tensor
        Returns:
            quats: [M, 4] tensor (w, x, y, z)
        """
        if vectors is None or vectors.numel() == 0:
            return torch.empty((0, 4), device=vectors.device if vectors is not None else "cpu")
        b = vectors
        eps = 1e-8
        speeds = torch.norm(b, dim=-1, keepdim=True)
        zero_mask = speeds.squeeze(-1) < eps
        # Normalize non-zero vectors
        bn = b / torch.clamp(speeds, min=eps)
        # a = (1,0,0); cross(a, b) = (0, -bz, by); dot = bx
        cross = torch.stack([torch.zeros_like(bn[..., 0]), -bn[..., 2], bn[..., 1]], dim=-1)
        dot = bn[..., 0]
        w = 1.0 + dot
        q = torch.zeros(bn.shape[0], 4, device=b.device, dtype=b.dtype)
        q[:, 0] = w
        q[:, 1:] = cross
        # Handle opposite direction (w ~ 0)
        opp_mask = w < 1e-6
        if torch.any(opp_mask):
            q[opp_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=b.device, dtype=b.dtype)
        # Normalize
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q = q / torch.clamp(q_norm, min=eps)
        # For zero vectors, set identity
        if torch.any(zero_mask):
            q[zero_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=b.device, dtype=b.dtype)
        return q