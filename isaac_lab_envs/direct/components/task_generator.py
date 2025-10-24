
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
        waypoints_length = torch.full((num_env,), 3, device=self.device)
        return start_tensor.unsqueeze(1), goal_tensor.unsqueeze(1), waypoints, waypoints_length


    def _generate_crossing_task_with_planner(self, num_env: int = 1, flight_height: float = 20.0):
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
                grid_map=self.env.state.map.occupancy_grid,
                waypoints=waypoints,
                waypoint_lengths=waypoints_length,
                bounds=bounds,
                grid_size=grid_size,
                output_path="planned_paths.png",
                max_trajs=min(10, num_env),
            )
            self._paths_viz_done = True


        return start_tensor, goal_tensor, waypoints, waypoints_length