import os
import csv
import argparse
from datetime import datetime
from typing import Tuple

import torch
from omni.isaac.lab.app import AppLauncher
from dataclasses import replace




@torch.no_grad()
def run_test(
    env_scale: float,
    duration_s: float,
    num_envs: int,
    headless: bool,
    out_dir: str,
) -> Tuple[str, dict]:
    # Launch Isaac Sim
    app_launcher = AppLauncher(headless=headless, enable_cameras=False)
    simulation_app = app_launcher.app
    # Use the only working environment: UamEnv
    from isaac_lab_envs.direct.uam_env import UamEnv
    from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg

    # Configure UAM environment
    cfg = UamEnvCfg()
    cfg.scene = replace(cfg.scene, num_envs=num_envs)
    cfg.env_scale = env_scale
    cfg.drone_model = "neo11"
    cfg.num_envs = num_envs
    cfg.debug_vis = not headless
    cfg.controller = "PlanarSpeedController"
    # Use world-frame for clarity
    cfg.action_manager.rl_action_frame = "world"

    # Continuous Box action space
    cfg.action_manager.action_space_type = "gaussian"

    # Instantiate env
    env = UamEnv(cfg=cfg, render_mode=None if headless else "rgb_array")
    obs, _ = env.reset()
    del obs

    device = env.device
    dt = float(env.step_dt)
    steps = int(duration_s / dt)
    target_height = float(env.cfg.flight_height)

    # Command: unit action in +X -> vx = max_speed, vy = 0
    actions = torch.zeros(env.num_envs, env.cfg.num_actions, device=device)
    actions[:, 0] = 1.0
    actions[:, 1] = 1.0
    print("first drone state:",env.state.ego_drone.drone_state)
    # obs, reward, terminated, truncated, info = env.step(actions)
    # obs, _ = env.reset()
    print("second drone state:",env.state.ego_drone.drone_state)

    # Logging
    log_rows = []
    sim_t = 0.0
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(actions)

        pos = env.state.ego_drone.positions[:, 0, :]   # [N,3]
        vel = env.state.ego_drone.velocities[:, 0, :]  # [N,3]
        z = pos[:, 2]
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]
        speed_xy_now = torch.sqrt(vx**2 + vy**2)
        z_err = target_height - z

        for e in range(env.num_envs):
            log_rows.append(
                [
                    sim_t,
                    float(z[e].item()),
                    float(z_err[e].item()),
                    float(vz[e].item()),
                    float(vx[e].item()),
                    float(vy[e].item()),
                    float(speed_xy_now[e].item()),
                ]
            )

        sim_t += dt
        if (terminated | truncated).any():
            break #说明失败了

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"uam_planar_speed_test_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "z", "z_error", "vz", "vx", "vy", "speed_xy"])
        writer.writerows(log_rows)

    z_vals = [r[1] for r in log_rows]
    z_err_abs = [abs(r[2]) for r in log_rows]
    speed_vals = [r[6] for r in log_rows]
    stats = {
        "z_mean": float(sum(z_vals) / max(len(z_vals), 1)),
        "z_err_abs_max": float(max(z_err_abs) if z_err_abs else 0.0),
        "speed_xy_mean": float(sum(speed_vals) / max(len(speed_vals), 1)),
        "speed_xy_max": float(max(speed_vals) if speed_vals else 0.0),
    }
    print(
        f"[Summary] z_mean={stats['z_mean']:.3f}, z_err_abs_max={stats['z_err_abs_max']:.3f}, "
        f"speed_xy_mean={stats['speed_xy_mean']:.3f}, speed_xy_max={stats['speed_xy_max']:.3f}"
    )
    print(f"[Saved] {csv_path}")

    env.close()
    simulation_app.close()
    return csv_path, stats



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=5000.0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default=os.path.join("runs", "controller_tests"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(
        env_scale=args.speed,
        duration_s=args.duration,
        num_envs=args.num_envs,
        headless=args.headless,
        out_dir=args.out_dir,
    )


