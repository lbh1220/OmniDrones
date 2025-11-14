import os
import csv
import json
import argparse
from dataclasses import replace
from datetime import datetime
from typing import List, Tuple, Dict

import torch
from omni.isaac.lab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search PD gains for PlanarSpeedController on UamEnv")
    parser.add_argument("--models", type=str, default="firefly", help="Comma-separated drone models")
    parser.add_argument("--speed", type=float, default=10.0, help="Target high speed (m/s)")
    parser.add_argument("--duration", type=float, default=40.0, help="Trial duration (s)")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per parameter set to reduce randomness")
    parser.add_argument("--num_envs", type=int, default=1, help="Num envs")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    parser.add_argument("--pmin", type=float, default=0.5, help="Min P gain")
    parser.add_argument("--pmax", type=float, default=10.0, help="Max P gain")
    parser.add_argument("--psteps", type=int, default=7, help="Grid steps for P")
    parser.add_argument("--dmin", type=float, default=0.0, help="Min D gain")
    parser.add_argument("--dmax", type=float, default=10.0, help="Max D gain")
    parser.add_argument("--dsteps", type=int, default=7, help="Grid steps for D")
    parser.add_argument("--out_dir", type=str, default=os.path.join("runs", "controller_tuning"), help="Output directory")
    return parser.parse_args()


@torch.no_grad()
def run_trial_on_env(
    env,
    duration_s: float,
    p_gain: float,
    d_gain: float,
) -> Tuple[Dict, List[List[float]]]:
    # Reset and set gains
    env.reset()
    device = env.device
    env.controller.height_p_gain.data = torch.tensor(p_gain, device=device, dtype=env.controller.height_p_gain.dtype)
    if hasattr(env.controller, "height_d_gain"):
        env.controller.height_d_gain.data = torch.tensor(d_gain, device=device, dtype=env.controller.height_p_gain.dtype)

    # Constant action in +X
    actions = torch.zeros(env.num_envs, env.cfg.num_actions, device=device)
    actions[:, 0] = 1.0
    actions[:, 1] = 0.0

    dt = float(env.step_dt)
    steps = max(1, int(duration_s / dt))
    target_height = float(env.cfg.flight_height)

    log_rows: List[List[float]] = []
    sim_t = 0.0
    failed = False

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        pos = env.state.ego_drone.positions[:, 0, :]
        vel = env.state.ego_drone.velocities[:, 0, :]
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
            failed = True
            break

    z_vals = [r[1] for r in log_rows]
    z_err_abs = [abs(r[2]) for r in log_rows]
    speed_vals = [r[6] for r in log_rows]
    z_rms = float((torch.tensor(z_err_abs).pow(2).mean().sqrt().item()) if len(z_err_abs) else 0.0)
    summary = {
        "model": env.cfg.drone_model,
        "p_gain": float(p_gain),
        "d_gain": float(d_gain),
        "failed": bool(failed),
        "z_mean": float(sum(z_vals) / max(len(z_vals), 1)) if z_vals else 0.0,
        "z_err_abs_max": float(max(z_err_abs) if z_err_abs else 0.0),
        "z_rms": z_rms,
        "speed_xy_mean": float(sum(speed_vals) / max(len(speed_vals), 1)) if speed_vals else 0.0,
        "steps_run": len(log_rows),
        "duration_s": float(len(log_rows) * dt),
    }
    return summary, log_rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(args.out_dir, f"tune_results_{ts}.csv")
    models: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    p_values = torch.linspace(args.pmin, args.pmax, steps=args.psteps).tolist()
    d_values = torch.linspace(args.dmin, args.dmax, steps=args.dsteps).tolist()

    # Launch simulator once
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    simulation_app = app_launcher.app

    try:
        # Prepare CSV
        with open(results_csv, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "model", "p_gain", "d_gain", "repeats", "failure_rate",
                "z_mean", "z_err_abs_max", "z_rms",
                "speed_xy_mean", "steps_run", "duration_s"
            ])

            best_per_model: Dict[str, Dict] = {}

            # Import env classes after app launch
            from isaac_lab_envs.direct.uam_env import UamEnv
            from isaac_lab_envs.direct.uam_env_cfg import UamEnvCfg

            for model in models:
                # Build one env per model
                cfg = UamEnvCfg()
                cfg.scene = replace(cfg.scene, num_envs=args.num_envs)
                cfg.num_envs = args.num_envs
                cfg.debug_vis = not args.headless
                cfg.drone_model = model
                cfg.controller = "PlanarSpeedController"
                cfg.action_manager.rl_action_frame = "world"
                cfg.action_manager.action_space_type = "gaussian"
                # cfg.env_scale = args.speed
                cfg.env_scale = 1.0

                env = UamEnv(cfg=cfg, render_mode=None if args.headless else "rgb_array")
                env.reset()

                best = None
                for p in p_values:
                    for d in d_values:
                        # Repeat trials for this parameter set
                        summaries = []
                        for _rep in range(max(1, args.repeats)):
                            summary, _ = run_trial_on_env(
                                env=env,
                                duration_s=args.duration,
                                p_gain=p,
                                d_gain=d,
                            )
                            summaries.append(summary)
                        # Aggregate across repeats
                        failures = sum(1 for s in summaries if s["failed"])
                        failure_rate = failures / float(len(summaries))
                        z_mean = sum(s["z_mean"] for s in summaries) / len(summaries)
                        z_err_abs_max = sum(s["z_err_abs_max"] for s in summaries) / len(summaries)
                        z_rms = sum(s["z_rms"] for s in summaries) / len(summaries)
                        speed_xy_mean = sum(s["speed_xy_mean"] for s in summaries) / len(summaries)
                        steps_run = sum(s["steps_run"] for s in summaries) / len(summaries)
                        duration_s_avg = sum(s["duration_s"] for s in summaries) / len(summaries)
                        agg = {
                            "model": model,
                            "p_gain": float(p),
                            "d_gain": float(d),
                            "repeats": int(len(summaries)),
                            "failure_rate": float(failure_rate),
                            "z_mean": float(z_mean),
                            "z_err_abs_max": float(z_err_abs_max),
                            "z_rms": float(z_rms),
                            "speed_xy_mean": float(speed_xy_mean),
                            "steps_run": float(steps_run),
                            "duration_s": float(duration_s_avg),
                        }
                        writer.writerow([
                            agg["model"], agg["p_gain"], agg["d_gain"], agg["repeats"], agg["failure_rate"],
                            agg["z_mean"], agg["z_err_abs_max"], agg["z_rms"],
                            agg["speed_xy_mean"], agg["steps_run"], agg["duration_s"]
                        ])
                        fcsv.flush()
                        print("finished parameter set:", p, d,"with success rate:", 1-failure_rate)

                        # Rank: prefer non-failed, then lower z_err_abs_max, then lower z_rms
                        if best is None:
                            best = agg
                        else:
                            better = False
                            if best["failure_rate"] > agg["failure_rate"]:
                                better = True
                            elif abs(best["failure_rate"] - agg["failure_rate"]) < 1e-9:
                                if agg["z_err_abs_max"] < best["z_err_abs_max"]:
                                    better = True
                                elif abs(agg["z_err_abs_max"] - best["z_err_abs_max"]) < 1e-6:
                                    if agg["z_rms"] < best["z_rms"]:
                                        better = True
                            if better:
                                best = agg

                best_per_model[model] = best
                env.close()

        # Save best JSON
        best_json = os.path.join(args.out_dir, f"tune_best_{ts}.json")
        with open(best_json, "w") as fjson:
            json.dump(best_per_model, fjson, indent=2)

        # Print concise suggestions
        for model, s in best_per_model.items():
            print(f"[Best] {model}: P={s['p_gain']:.3f}, D={s['d_gain']:.3f}, failed={s['failed']}, "
                  f"z_err_abs_max={s['z_err_abs_max']:.3f}, z_rms={s['z_rms']:.3f}")
        print(f"[Saved] results: {results_csv}")
        print(f"[Saved] best:    {best_json}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()


