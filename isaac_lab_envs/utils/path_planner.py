import torch
import numpy as np
from omni.isaac.lab.utils.configclass import configclass
from typing import Tuple, List
@configclass
class GlobalPathPlannerCfg:
    algorithm: str = "astar"
    smooth_method: str = "shortcut"


class GlobalPathPlanner:
    def __init__(self, cfg: GlobalPathPlannerCfg):
        self.cfg = cfg
        self.grid_map_np = None
        self.grid_size = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.weight_map = None
        self._pyastar = None

    def update_grid_map(self, grid_map: torch.Tensor, grid_size: float, bounds: tuple[float, float, float, float]):
        """
        Update the grid map and the bounds of the global path planner.
        Args:
            grid_map: The grid map of the environment. [H, W]
            grid_size: The size of the grid.
            bounds: The bounds of the environment. (xmin, xmax, ymin, ymax)
        """
        self.grid_size = float(grid_size)
        self.x_min = float(bounds[0])
        self.x_max = float(bounds[1])
        self.y_min = float(bounds[2])
        self.y_max = float(bounds[3])
        # Convert torch tensor to numpy array (boolean -> 0/1)
        if isinstance(grid_map, torch.Tensor):
            grid_np = grid_map.detach().to('cpu').numpy()
        else:
            grid_np = np.asarray(grid_map)
        # Ensure binary map {0,1}
        grid_np = (grid_np > 0).astype(np.uint8)
        self.grid_map_np = grid_np
        # Precompute weight map for pyastar2d: inf for obstacles, 1.0 for free
        self.weight_map = np.where(self.grid_map_np == 1, np.inf, 1.0).astype(np.float32)
        # Lazy import pyastar2d
        try:
            import pyastar2d as _pyastar2d
            self._pyastar = _pyastar2d
        except Exception:
            self._pyastar = None

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Plan a 2D path on the grid map using A*.

        Args:
            start: (x, y) world coordinates
            goal:  (x, y) world coordinates

        Returns:
            List of (x, y) world coordinates along the path; empty if no path.
        """
        if self.grid_map_np is None or self.weight_map is None or self._pyastar is None:
            return []

        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])

        start_rc = self._world_to_grid_rc(sx, sy)
        goal_rc = self._world_to_grid_rc(gx, gy)

        try:
            # Try smoothing if configured (best-effort; fall back if unsupported)
            if getattr(self.cfg, 'smooth_method', None):
                try:
                    path_rc = self._pyastar.astar_path(
                        self.weight_map, start_rc, goal_rc, allow_diagonal=True,
                        smooth_path=True, smooth_method=self.cfg.smooth_method
                    )
                except TypeError:
                    path_rc = self._pyastar.astar_path(self.weight_map, start_rc, goal_rc, allow_diagonal=True)
            else:
                path_rc = self._pyastar.astar_path(self.weight_map, start_rc, goal_rc, allow_diagonal=True)
        except Exception:
            return []

        if path_rc is None or len(path_rc) == 0:
            return []

        path_xy: List[Tuple[float, float]] = []
        for r, c in path_rc:
            x, y = self._grid_to_world_xy(c, r)
            path_xy.append((x, y))

        # snap endpoints to the exact input start/goal if close enough
        tol = max(1e-6, 1.42 * self.grid_size) if self.grid_size is not None else 0.5
        # start
        if len(path_xy) >= 1:
            dx0 = path_xy[0][0] - sx
            dy0 = path_xy[0][1] - sy
            if (dx0 * dx0 + dy0 * dy0) ** 0.5 <= tol:
                path_xy[0] = (sx, sy)
        # goal
        if len(path_xy) >= 1:
            dx1 = path_xy[-1][0] - gx
            dy1 = path_xy[-1][1] - gy
            if (dx1 * dx1 + dy1 * dy1) ** 0.5 <= tol:
                path_xy[-1] = (gx, gy)

        return path_xy

    def _world_to_grid_rc(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x, y) to grid (row, col)."""
        col = int((x - self.x_min) / self.grid_size)
        row = int((y - self.y_min) / self.grid_size)
        # clamp to map bounds
        h, w = self.grid_map_np.shape
        col = max(0, min(col, w - 1))
        row = max(0, min(row, h - 1))
        return (row, col)

    def _grid_to_world_xy(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid (col, row) to world (x, y)."""
        x = float(col * self.grid_size + self.x_min)
        y = float(row * self.grid_size + self.y_min)
        return x, y