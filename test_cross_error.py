#!/usr/bin/env python3
"""
Visual test harness for cross-track projection and local goal computation
using the actual `EnvState.update_navigation_state_vectorized` implementation.

Features
- Generate two random polyline paths with different waypoint counts
- Instantiate `EnvState(num_envs=2)` and fill the two paths
- For each given position, compute projection and local goal on both paths
- Interactive mode: click to analyze the same point on both paths
- Visualization: dashed lines from position -> projection -> local goal

Usage example
  python test_cross_error.py --num1 8 --num2 12 --lookahead 15

Notes
- Relies on torch and the project's `EnvState` class (no reimplementation here).
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from isaac_lab_envs.direct.mdp.state import EnvState


@dataclasses.dataclass
class EnvAnalysisResult:
    position: np.ndarray  # (2,)
    projection_point: np.ndarray  # (2,)
    local_goal: np.ndarray  # (2,)
    cross_track_error: float
    current_waypoint_index: int
    dist_projection_to_goal: float


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def build_env_state_for_two_paths(
    path1: np.ndarray,
    path2: np.ndarray,
    device: str = "cpu",
) -> EnvState:
    """Create an `EnvState` with two environments and load two 2D paths (z=0)."""
    assert path1.ndim == 2 and path1.shape[1] == 2
    assert path2.ndim == 2 and path2.shape[1] == 2

    num_envs = 2
    max_len = int(max(path1.shape[0], path2.shape[0]))

    state = EnvState(device=device, num_envs=num_envs)
    state.initialize_basic_tensors()
    state.initialize_navigation_waypoints(max_wps=max_len)

    # Fill waypoints and lengths
    wps = torch.zeros((num_envs, max_len, 3), dtype=torch.float32, device=device)
    lens = torch.zeros((num_envs,), dtype=torch.long, device=device)

    def np_to_xyz(path: np.ndarray) -> np.ndarray:
        xyz = np.zeros((path.shape[0], 3), dtype=np.float32)
        xyz[:, :2] = path
        xyz[:, 2] = 0.0
        return xyz

    p1_xyz = np_to_xyz(path1)
    p2_xyz = np_to_xyz(path2)
    wps[0, : path1.shape[0]] = torch.from_numpy(p1_xyz)
    wps[1, : path2.shape[0]] = torch.from_numpy(p2_xyz)
    lens[0] = path1.shape[0]
    lens[1] = path2.shape[0]

    state.navigation.waypoints = wps
    state.navigation.waypoint_lengths = lens
    # Targets (used only for very short paths). Set to last waypoint
    tgt = torch.stack((wps[0, lens[0] - 1], wps[1, lens[1] - 1]), dim=0).unsqueeze(1)
    state.navigation.target_positions = tgt

    return state


def run_env_compute(
    state: EnvState,
    pos_xy: np.ndarray,
    lookahead: float,
) -> Tuple[EnvAnalysisResult, EnvAnalysisResult]:
    """Set both env positions to the same (x, y, z=0), run update, and return results."""
    assert pos_xy.shape == (2,)
    pos = torch.tensor([[pos_xy[0], pos_xy[1], 0.0]], dtype=torch.float32, device=state.device)
    # Broadcast to [num_envs, 1, 3]
    positions = pos.repeat(state.num_envs, 1).view(state.num_envs, 1, 3)
    state.ego_drone.positions = positions

    state.update_navigation_state_vectorized(lookahead)

    results = []
    for env_id in range(state.num_envs):
        proj = state.navigation.projection_points[env_id, 0, :2].detach().cpu().numpy()
        goal = state.navigation.local_goals[env_id, 0, :2].detach().cpu().numpy()
        cte = float(state.navigation.cross_track_errors[env_id].item())
        cwi = int(state.navigation.current_waypoint_indices[env_id].item())
        dpg = float(state.navigation.current_dist_along_path[env_id].item())
        results.append(
            EnvAnalysisResult(
                position=pos_xy.astype(np.float64),
                projection_point=proj,
                local_goal=goal,
                cross_track_error=cte,
                current_waypoint_index=cwi,
                dist_projection_to_goal=dpg,
            )
        )
    return results[0], results[1]


def generate_random_path(
    num_points: int,
    bounds: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
    jitter_scale: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a path spanning the area: start/end near opposite bounds, smooth lateral wiggles.

    The path is constructed by linearly interpolating from a start point to an end point
    that are placed near opposite borders, then adding a smoothed lateral noise profile
    orthogonal to the main direction. This avoids clustering in a small region.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    span_x = float(xmax - xmin)
    span_y = float(ymax - ymin)
    margin_x = 0.12 * span_x
    margin_y = 0.12 * span_y

    # Randomly pick horizontal or vertical orientation for start/end
    horizontal = bool(rng.integers(0, 2))
    if horizontal:
        start = np.array([xmin + margin_x, rng.uniform(ymin + margin_y, ymax - margin_y)], dtype=np.float64)
        end = np.array([xmax - margin_x, rng.uniform(ymin + margin_y, ymax - margin_y)], dtype=np.float64)
    else:
        start = np.array([rng.uniform(xmin + margin_x, xmax - margin_x), ymin + margin_y], dtype=np.float64)
        end = np.array([rng.uniform(xmin + margin_x, xmax - margin_x), ymax - margin_y], dtype=np.float64)

    # Main direction and its unit normal
    direction = end - start
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-6:
        direction = np.array([max(1e-3, span_x * 0.01), 0.0], dtype=np.float64)
        dir_norm = float(np.linalg.norm(direction))
    unit_dir = direction / dir_norm
    unit_n = np.array([-unit_dir[1], unit_dir[0]], dtype=np.float64)

    # Interpolation parameters
    t = np.linspace(0.0, 1.0, num_points)
    base = start[None, :] + t[:, None] * direction[None, :]

    # Lateral noise with smoothing
    lateral_sigma = 0.06 * max(span_x, span_y)
    noise = rng.normal(0.0, lateral_sigma, size=num_points)
    if num_points >= 4:
        kernel = np.array([0.25, 0.5, 0.25])
        # simple convolution smoothing (two passes)
        for _ in range(2):
            noise_pad = np.pad(noise, (1, 1), mode="edge")
            noise = (
                kernel[0] * noise_pad[:-2]
                + kernel[1] * noise_pad[1:-1]
                + kernel[2] * noise_pad[2:]
            )

    # Reduce lateral noise at endpoints to keep start/end near borders
    fade = (1.0 - np.cos(np.pi * t)) * 0.5  # 0 at ends, 1 at middle
    noise *= fade

    points = base + noise[:, None] * unit_n[None, :]
    # Ensure first/last exactly at start/end
    points[0] = start
    points[-1] = end

    # Clamp to bounds
    for i in range(num_points):
        points[i, 0] = clamp(points[i, 0], xmin, xmax)
        points[i, 1] = clamp(points[i, 1], ymin, ymax)

    return points


def parse_xy(text: str) -> np.ndarray:
    xs, ys = text.split(",")
    return np.array([float(xs), float(ys)], dtype=np.float64)


def plot_path(ax: plt.Axes, waypoints: np.ndarray, color: str, label: str) -> None:
    ax.plot(waypoints[:, 0], waypoints[:, 1], "-o", color=color, lw=2, ms=4, label=label)


def plot_analysis(
    ax: plt.Axes,
    result: EnvAnalysisResult,
    color: str,
    prefix: str,
) -> None:
    px, py = result.position
    projx, projy = result.projection_point
    gx, gy = result.local_goal
    ax.scatter([px], [py], color=color, marker="x", s=60, zorder=5, label=f"{prefix} pos")
    ax.scatter([projx], [projy], color=color, marker="s", s=40, zorder=6, label=f"{prefix} proj")
    ax.scatter([gx], [gy], color=color, marker="^", s=50, zorder=6, label=f"{prefix} local_goal")
    ax.plot([px, projx], [py, projy], linestyle="--", color=color, alpha=0.7)
    ax.plot([projx, gx], [projy, gy], linestyle=":", color=color, alpha=0.7)


def print_result(prefix: str, res: EnvAnalysisResult) -> None:
    print(
        f"{prefix}: pos={res.position}, proj={res.projection_point}, local={res.local_goal}, "
        f"cte={res.cross_track_error:.3f}, wp_idx={res.current_waypoint_index}, "
        f"dist_proj_to_goal={res.dist_projection_to_goal:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-track projection and local goal visual test (non-interactive)")
    parser.add_argument("--num1", type=int, default=4, help="Number of waypoints for path 1 (>=2)")
    parser.add_argument("--num2", type=int, default=5, help="Number of waypoints for path 2 (>=2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lookahead", type=float, default=10.0, help="Lookahead distance along the path")
    parser.add_argument("--bounds", type=str, default="-50,50,-50,50", help="xmin,xmax,ymin,ymax")
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = [float(v) for v in args.bounds.split(",")]
    bounds = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(args.seed)

    num1 = max(2, args.num1)
    num2 = max(2, args.num2)
    path1 = generate_random_path(num1, bounds=bounds, seed=args.seed)
    path2 = generate_random_path(num2, bounds=bounds, seed=args.seed + 1)
    # Build env state with two envs and load the paths
    device = "cpu"
    state = build_env_state_for_two_paths(path1, path2, device=device)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title("Projection and Local Goal (env0: pos1 on path1, env1: pos2 on path2)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plot_path(ax, path1, color="#1f77b4", label="path 1")
    plot_path(ax, path2, color="#2ca02c", label="path 2")

    # Generate two random positions and assign them to env0 and env1 respectively
    # pos1 = rng.uniform([xmin, ymin], [xmax, ymax])
    # pos2 = rng.uniform([xmin, ymin], [xmax, ymax])
    pos1 = np.array([35, -30], dtype=np.float64)
    pos2 = np.array([35, -30], dtype=np.float64)

    # Run once using both positions
    def run_env_compute_two_positions(state: EnvState, pos1_xy: np.ndarray, pos2_xy: np.ndarray, lookahead: float):
        pos = torch.tensor(
            [
                [pos1_xy[0], pos1_xy[1], 0.0],
                [pos2_xy[0], pos2_xy[1], 0.0],
            ],
            dtype=torch.float32,
            device=state.device,
        ).view(state.num_envs, 1, 3)
        state.ego_drone.positions = pos
        state.update_navigation_state_vectorized(lookahead)

        results = []
        for env_id in range(state.num_envs):
            proj = state.navigation.projection_points[env_id, 0, :2].detach().cpu().numpy()
            goal = state.navigation.local_goals[env_id, 0, :2].detach().cpu().numpy()
            cte = float(state.navigation.cross_track_errors[env_id].item())
            cwi = int(state.navigation.current_waypoint_indices[env_id].item())
            dpg = float(state.navigation.current_dist_along_path[env_id].item())
            pos_xy = pos1 if env_id == 0 else pos2
            results.append(
                EnvAnalysisResult(
                    position=np.asarray(pos_xy, dtype=np.float64),
                    projection_point=proj,
                    local_goal=goal,
                    cross_track_error=cte,
                    current_waypoint_index=cwi,
                    dist_projection_to_goal=dpg,
                )
            )
        return results[0], results[1]

    res_env0, res_env1 = run_env_compute_two_positions(state, pos1, pos2, args.lookahead)

    print_result("pos1 on path1 (env0)", res_env0)
    print_result("pos2 on path2 (env1)", res_env1)

    plot_analysis(ax, res_env0, color="#1f77b4", prefix="env0:p1")
    plot_analysis(ax, res_env1, color="#2ca02c", prefix="env1:p2")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()


