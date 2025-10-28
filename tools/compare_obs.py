from __future__ import annotations
import os
import sys
import numpy as np
from types import SimpleNamespace

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import two implementations
from isaac_lab_envs.direct.mdp.observation_airsim import IsaacLikeObservationProcessor as IsaacObsProc  # type: ignore
from compare_airsim import IsaacLikeObservationProcessor as AirsimObsProc  # type: ignore


def build_mock_state(num_aircraft: int = 5):
    # Vehicle (ego)
    vehicle = SimpleNamespace(
        radius=0.6,
        max_velocity=8.0,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=10.0),
            orientation=SimpleNamespace(
                to_array=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            ),
        ),
        get_position_array=lambda: np.array([0.0, 0.0, 10.0], dtype=np.float32),
        get_velocity_array=lambda: np.array([5.0, 0.0, 0.0], dtype=np.float32),
        get_yaw=lambda: 0.0,
    )

    # Navigation
    navigation = SimpleNamespace(
        goal_position=SimpleNamespace(
            to_array=lambda: np.array([100.0, 0.0, 10.0], dtype=np.float32)
        ),
        has_local_goal=False,
        local_goal_position=SimpleNamespace(
            to_array=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ),
        closest_point_on_path=SimpleNamespace(
            to_array=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ),
    )

    # Traffic list
    def make_aircraft(i: int):
        angle = (i + 1) * (np.pi / (num_aircraft + 1))
        r = 50 + 10 * i
        px = r * np.cos(angle)
        py = r * np.sin(angle)
        pz = 10.0
        vx = -np.sin(angle) * 3.0
        vy = np.cos(angle) * 3.0
        vz = 0.0
        return SimpleNamespace(
            name=f"aircraft_{i}",
            aircraft_type="evtol" if (i % 3 == 0) else "uav",
            position=SimpleNamespace(x=px, y=py, z=pz),
            linear_velocity=SimpleNamespace(x=vx, y=vy, z=vz),
            radius=1.2 if (i % 3 == 0) else 0.6,
        )

    raw_aircraft_states = [make_aircraft(i) for i in range(num_aircraft)]

    # EnvironmentState stub
    state = SimpleNamespace(
        vehicle=vehicle,
        navigation=navigation,
        raw_aircraft_states=raw_aircraft_states,
    )

    return state


def to_numpy_dict(policy):
    out = {}
    for k, v in policy.items():
        arr = np.asarray(v)
        out[k] = arr
    return out


def compare_arrays(a: np.ndarray, b: np.ndarray, name: str, atol=1e-6, rtol=1e-6):
    same_shape = a.shape == b.shape
    if not same_shape:
        print(f"[shape-mismatch] {name}: {a.shape} vs {b.shape}")
        return False
    if np.allclose(a, b, atol=atol, rtol=rtol):
        print(f"[ok] {name} equal (max_abs_err={np.max(np.abs(a-b)):.3e})")
        return True
    else:
        diff = np.abs(a - b)
        max_err = np.max(diff)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"[diff] {name} not equal, max_abs_err={max_err:.6g} at {idx}: {a[idx]} vs {b[idx]}")
        return False


def main():
    # Unified configs
    cfg = SimpleNamespace(
        observation_radius=1000.0,
        vehicle_name="ego",
        max_aircraft_in_obs=20,
        obs_noise_level=0.0,
        use_global_planner=False,
        predict_steps=4,
        pred_timestep=0.1,
    )

    sb3_cfg = SimpleNamespace(
        obs=SimpleNamespace(
            use_angle_distance_obs=False,
            norm_scale=1.0,
            circle_radius=113.137085,
        )
    )

    # Instantiate both processors
    isaac_proc = IsaacObsProc(config=cfg, sb3_config=sb3_cfg)
    airsim_proc = AirsimObsProc(config=cfg, sb3_config=sb3_cfg)

    # Build same mock state
    state = build_mock_state(num_aircraft=6)

    # Monkeypatch AirSim-side extractor to use the same raw_aircraft_states
    def _extract_aircraft_states_same(self, s):
        aircraft_states = []
        current_pos = s.vehicle.get_position_array()
        distances = []
        for aircraft in s.raw_aircraft_states:
            if aircraft.name == self.config.vehicle_name:
                continue
            aircraft_pos = np.array([aircraft.position.x, aircraft.position.y, aircraft.position.z])
            dist = np.linalg.norm(aircraft_pos - current_pos)
            if dist <= self.config.observation_radius:
                aircraft_dict = {
                    'aircraft_type': aircraft.aircraft_type,
                    'position': aircraft_pos.tolist(),
                    'velocity': [aircraft.linear_velocity.x, aircraft.linear_velocity.y, aircraft.linear_velocity.z],
                    'radius': aircraft.radius,
                    'name': aircraft.name,
                    'distance': float(dist),
                }
                distances.append((aircraft_dict['distance'], aircraft_dict))
        distances.sort(key=lambda x: x[0])
        num_aircraft = min(len(distances), self.config.max_aircraft_in_obs)
        aircraft_states = [distances[i][1] for i in range(num_aircraft)]
        return aircraft_states

    # Apply monkeypatch to both to ensure identical upstream ordering/filtering
    IsaacObsProc._extract_aircraft_states = _extract_aircraft_states_same  # type: ignore
    AirsimObsProc._extract_aircraft_states = _extract_aircraft_states_same  # type: ignore

    # Run
    isaac_out = isaac_proc.process_observation(state)
    airsim_out = airsim_proc.process_observation(state)

    isaac_np = to_numpy_dict(isaac_out)
    airsim_np = to_numpy_dict(airsim_out)

    keys = [
        'robot_node',
        'temporal_edges',
        'spatial_edges',
        'visible_masks',
        'spatial_types',
    ]

    all_ok = True
    for k in keys:
        ok = compare_arrays(isaac_np[k], airsim_np[k], name=k, atol=1e-6, rtol=1e-6)
        all_ok = all_ok and ok

    print("== Result ==", "MATCH" if all_ok else "DIFFERS")


if __name__ == "__main__":
    main()


