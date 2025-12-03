
import omni.isaac.lab.utils.math as math_utils
import torch

class TaskGenerator:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = "cuda"

    def generate_task(self, num_env: int = 1, flight_height: float = 20.0):
        raise NotImplementedError("Subclasses must implement this method")


class CrossTaskGenerator(TaskGenerator):
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = "cuda"
        self.circle_radius = min(self.cfg.area_bounds.xmax - self.cfg.area_bounds.xmin, 
                                self.cfg.area_bounds.ymax - self.cfg.area_bounds.ymin)/2.0
        # Cache of safe theta indices computed from the occupancy grid
        self._safe_theta_indices = None
        self._safe_theta_radius = None
        self._safe_theta_divisions = None
        self._paths_viz_done = False
        # Cache of free world XY points converted from occupancy grid
        self._free_xy_cpu = None

    def generate_task(self, num_env: int = 1, flight_height: float = 20.0):
        self.device = self.env.device
        if self.cfg.use_global_path:
            if self.env.map_manager is None or self.env.global_path_planner is None:
                return self._generate_crossing_task_with_waypoints(num_env, flight_height)
            else:
                return self._generate_crossing_task_with_planner(num_env, flight_height)
        else:
            return self._generate_crossing_task(num_env, flight_height)

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
        waypoints = torch.cat([start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1)], dim=1)
        waypoints_length = torch.full((num_env,), 2, device=self.device)

        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), waypoints, waypoints_length
    
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
        waypoints_length = torch.full((num_env,), 3, device=self.device)
        max_wps = self.env.state.navigation.waypoints.shape[1]
        fill_waypoints = torch.zeros(num_env, max_wps, 3, device=self.device)
        fill_waypoints[:, :3, :] = waypoints
        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), fill_waypoints, waypoints_length


    def _generate_crossing_task_with_planner(self, num_env: int = 1, flight_height: float = 20.0):
        if num_env <= 0:
            raise ValueError("num_aircraft must be greater than 0")
        area_center = torch.tensor([(self.cfg.area_bounds.xmin + self.cfg.area_bounds.xmax) / 2, 
                                    (self.cfg.area_bounds.ymin + self.cfg.area_bounds.ymax) / 2, 
                                    flight_height], device=self.device)
        area_center = area_center.unsqueeze(0)

        # Prepare outputs
        max_wps = self.env.state.navigation.waypoints.shape[1]  # reasonable upper bound for A* path points
        waypoints = torch.zeros(num_env, max_wps, 3, device=self.device)
        waypoints_length = torch.zeros(num_env, dtype=torch.long, device=self.device)

        # Ensure planner ready and grid available
        planner = self.env.global_path_planner if self.cfg.use_global_path else None
        grid = getattr(self.env.state.map, "extended_occupancy_grid", None)
        bounds = getattr(self.env.state.map, "grid_bounds", None)
        grid_size = getattr(self.env.state.map, "grid_size", None)
        if planner is not None and planner.grid_map_np is None:
            # update grid in case re-generated
            planner.update_grid_map(grid, grid_size, bounds)

        # Pre-compute and cache safe theta set on first use if grid info is available
        # We evaluate safety by requiring both start and goal (opposite points) on a fixed radius to fall in free cells.
        # This gives equal-length tasks and avoids obviously unsafe endpoints.
        if (grid is not None and bounds is not None and grid_size is not None):
            theta_divisions = int(getattr(self.cfg, "theta_divisions", 360))
            # Recompute cache if divisions changed or cache not present
            need_recompute = (
                self._safe_theta_indices is None or
                self._safe_theta_divisions != theta_divisions
            )
            if need_recompute:
                # Use a slightly smaller radius than the theoretical bound to keep margin from edges
                safe_radius = float(self.circle_radius) * 0.95
                xmin, xmax, ymin, ymax = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
                # Grid expected shape [H, W] with order [y, x]
                gm = grid.detach().to("cpu")
                H, W = int(gm.shape[0]), int(gm.shape[1])

                # Build all candidate thetas
                angles = torch.arange(theta_divisions, dtype=torch.float32) * (2.0 * torch.pi / float(theta_divisions))
                cos_vals = torch.cos(angles)
                sin_vals = torch.sin(angles)
                # Build candidate world points on the ring (start) and the opposite (goal)
                center_xy = area_center[0, :2].detach().to("cpu")
                sx = center_xy[0] + safe_radius * cos_vals
                sy = center_xy[1] + safe_radius * sin_vals
                gx = center_xy[0] - safe_radius * cos_vals
                gy = center_xy[1] - safe_radius * sin_vals
                # Map world -> grid indices
                ix_s = torch.clamp(((sx - xmin) / float(grid_size)).floor().long(), 0, W - 1)
                iy_s = torch.clamp(((sy - ymin) / float(grid_size)).floor().long(), 0, H - 1)
                ix_g = torch.clamp(((gx - xmin) / float(grid_size)).floor().long(), 0, W - 1)
                iy_g = torch.clamp(((gy - ymin) / float(grid_size)).floor().long(), 0, H - 1)
                # Occupancy check: True means occupied; we need both free
                occ_s = gm[iy_s, ix_s] > 0
                occ_g = gm[iy_g, ix_g] > 0
                safe_mask = (~occ_s) & (~occ_g)
                safe_indices = torch.where(safe_mask)[0]
                # Cache (store CPU list)
                self._safe_theta_indices = safe_indices.tolist()
                self._safe_theta_radius = safe_radius
                self._safe_theta_divisions = theta_divisions
                # Also cache free cells (world XY) for simple fallback sampling
                if self._free_xy_cpu is None:
                    free_mask = (gm == 0)
                    free_ys, free_xs = torch.where(free_mask)
                    if free_xs.numel() > 0:
                        wx = xmin + (free_xs.to(torch.float32) + 0.5) * float(grid_size)
                        wy = ymin + (free_ys.to(torch.float32) + 0.5) * float(grid_size)
                        self._free_xy_cpu = torch.stack([wx, wy], dim=1)  # CPU tensor [N, 2]

        # Build start and goal tensors
        if getattr(self, "_safe_theta_indices", None) and len(self._safe_theta_indices) > 0:
            # Sample safe thetas (equal-length tasks on a ring)
            sel = torch.randint(low=0, high=len(self._safe_theta_indices), size=(num_env,), device=self.device)
            inds = torch.tensor([self._safe_theta_indices[i.item()] for i in sel], device=self.device, dtype=torch.long)
            angles_sel = inds.float() * (2.0 * torch.pi / float(self._safe_theta_divisions))
            cos_vals = torch.cos(angles_sel)
            sin_vals = torch.sin(angles_sel)
            R = float(self._safe_theta_radius)
            center_xy = area_center[0, :2]
            s_xy = torch.stack([center_xy[0] + R * cos_vals, center_xy[1] + R * sin_vals], dim=1)
            g_xy = torch.stack([center_xy[0] - R * cos_vals, center_xy[1] - R * sin_vals], dim=1)
            z = torch.full((num_env, 1), float(flight_height), device=self.device)
            start_tensor = torch.cat([s_xy, z], dim=1).unsqueeze(1)
            goal_tensor = torch.cat([g_xy, z], dim=1).unsqueeze(1)
        else:
            # Fallback to original random sampling if no safe thetas or grid unavailable
            start_tensor = math_utils.sample_cylinder(self.circle_radius, (0, 0), num_env, self.device)
            goal_tensor = -start_tensor.clone()
            start_tensor = start_tensor + area_center
            goal_tensor = goal_tensor + area_center
            start_tensor = start_tensor.unsqueeze(1)
            goal_tensor = goal_tensor.unsqueeze(1)

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
                # resample start/goal: prefer safe theta set if available; otherwise fallback
                if getattr(self, "_safe_theta_indices", None) and len(self._safe_theta_indices) > 0:
                    idx = int(torch.randint(low=0, high=len(self._safe_theta_indices), size=(1,), device=self.device).item())
                    theta_idx = self._safe_theta_indices[idx]
                    angle = float(theta_idx) * (2.0 * 3.141592653589793 / float(self._safe_theta_divisions))
                    R = float(self._safe_theta_radius)
                    cx, cy = float(area_center[0, 0].item()), float(area_center[0, 1].item())
                    s = torch.tensor([cx + R * torch.cos(torch.tensor(angle, device=self.device)),
                                      cy + R * torch.sin(torch.tensor(angle, device=self.device))],
                                     device=self.device, dtype=start_tensor.dtype)
                    g = torch.tensor([cx - R * torch.cos(torch.tensor(angle, device=self.device)),
                                      cy - R * torch.sin(torch.tensor(angle, device=self.device))],
                                     device=self.device, dtype=start_tensor.dtype)
                    # keep z
                    start_tensor[i, 0, 0] = s[0]
                    start_tensor[i, 0, 1] = s[1]
                    goal_tensor[i, 0, 0] = g[0]
                    goal_tensor[i, 0, 1] = g[1]
                else:
                    s_new = math_utils.sample_cylinder(self.circle_radius, (0, 0), 1, self.device)[0]
                    g_new = -s_new
                    s = s_new + area_center[0]
                    g = g_new + area_center[0]
                    # update tensors to keep consistent with actual path endpoints
                    start_tensor[i, 0, :2] = s[:2]
                    goal_tensor[i, 0, :2] = g[:2]

            if len(path_xy) <= 1:
                # Fallback: sample two random free cells from grid as start/goal,
                # then try A* once more; if still fails, use simple 2-point path.
                if (getattr(self, "_free_xy_cpu", None) is not None) and (self._free_xy_cpu.shape[0] >= 2):
                    # sample two distinct indices
                    idxs = torch.randint(low=0, high=self._free_xy_cpu.shape[0], size=(2,))
                    while idxs[1].item() == idxs[0].item():
                        idxs[1] = torch.randint(low=0, high=self._free_xy_cpu.shape[0], size=(1,))
                    s_xy_world = self._free_xy_cpu[idxs[0]].to(self.device)
                    g_xy_world = self._free_xy_cpu[idxs[1]].to(self.device)
                    # update s,g and tensors
                    s = torch.stack([s_xy_world[0], s_xy_world[1]], dim=0)
                    g = torch.stack([g_xy_world[0], g_xy_world[1]], dim=0)
                    start_tensor[i, 0, 0] = s[0]
                    start_tensor[i, 0, 1] = s[1]
                    goal_tensor[i, 0, 0] = g[0]
                    goal_tensor[i, 0, 1] = g[1]
                    # attempt A* once with these new endpoints
                    path_xy_try = []
                    if planner is not None:
                        path_xy_try = planner.plan_path((float(s[0].item()), float(s[1].item())),
                                                        (float(g[0].item()), float(g[1].item())))
                    if len(path_xy_try) > 1:
                        path_xy = path_xy_try
                    else:
                        # final fallback: simple 2-point path
                        path_xy = [
                            (float(s[0].item()), float(s[1].item())),
                            (float(g[0].item()), float(g[1].item()))
                        ]
                else:
                    # final fallback to current s,g straight line
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
                grid_map=self.env.state.map.occupancy_grid,
                waypoints=waypoints,
                waypoint_lengths=waypoints_length,
                bounds=bounds,
                grid_size=grid_size,
                output_path="ego_planned_paths.png",
                max_trajs=min(10, num_env),
            )
            self._paths_viz_done = True


        return start_tensor, goal_tensor, waypoints, waypoints_length