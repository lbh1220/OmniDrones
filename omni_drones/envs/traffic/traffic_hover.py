import hydra
import torch
import torch.distributions as D
from omni_drones.envs import IsaacEnv
from omni_drones.traffic import TrafficSimulator, TrafficConfig
from omni_drones.robots.drone import MultirotorBase
from omni_drones.controllers.lee_position_controller import LeePositionController
from omni_drones.views import ArticulationView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from tensordict.tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni_drones.utils.torchrl import AgentSpec


class TrafficAwareHover(IsaacEnv):
    """
    Example environment that integrates traffic simulation.
    
    This environment extends the basic Hover task by adding background traffic
    aircraft and collision detection capabilities.
    """
    
    def __init__(self, cfg, headless):
        
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        # Initialize traffic simulator
        self.traffic_config = self._create_traffic_config(cfg)
        
        self.traffic_simulator = TrafficSimulator(self.traffic_config, device=cfg.sim.device)
        super().__init__(cfg, headless)
        


        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -2.5, 1.], device=self.device),
            torch.tensor([2.5, 2.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )

        self.target_pos = torch.tensor([[0.0, 0.0, 2.]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8
        # Set up specs
        self.traffic_simulator.initialize()
        print(f"TrafficAwareHover initialized with {self.traffic_config.num_drones} traffic aircraft")

    def _design_scene(self):
        """Design the scene including main drones and traffic setup."""
        # Create drone instances
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 2.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(),
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        # Initialize traffic simulator (it will create its own prims under /World/Traffic)
        self.traffic_simulator.create_traffic_prim()
        

        
        return ["/World/defaultGroundPlane"]

    def _create_traffic_config(self, cfg) -> TrafficConfig:
        """Create traffic configuration from main config."""
        traffic_cfg = cfg.get("traffic", {})
        
        return TrafficConfig(
            num_drones=traffic_cfg.get("num_drones", 10),
            drone_model=traffic_cfg.get("drone_model", "crazyflie"),
            flight_height=traffic_cfg.get("flight_height", 5.0),
            max_speed=traffic_cfg.get("max_speed", 3.0),
            arrival_threshold=traffic_cfg.get("arrival_threshold", 1.0),
            area_bounds=traffic_cfg.get("area_bounds", {
                "xmin": -20.0, "xmax": 20.0, "ymin": -20.0, "ymax": 20.0
            })
        )
    
    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
    
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset both main environment and traffic simulation."""
        # Reset drone
        self.drone._reset_idx(env_ids, self.training)
        
        # Set initial drone positions
        init_pos = torch.zeros(len(env_ids), 3, device=self.device)
        init_pos[:, 2] = 2.0  # Start at height 2
        init_rot = euler_to_quaternion(torch.zeros(len(env_ids), 3, device=self.device))
        init_rot = init_rot.unsqueeze(1)    
        init_pos = init_pos.unsqueeze(1)
        self.drone.set_world_poses(init_pos, init_rot, env_ids)
        # self.drone.set_velocities(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        # Reset traffic simulation
        self.traffic_simulator.reset()
        
        # Reset stats
        self.stats[env_ids] = 0.
    
    def _pre_sim_step(self, tensordict):
        """Pre-simulation step including traffic updates."""
        # Apply actions to drone
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        
        # Update traffic simulation
        self.traffic_simulator.step()
    
    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        obs = [self.rpos, self.drone_state[..., 3:], self.rheading,]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pose reward
        self.reward_effort_weight = 0.0
        self.reward_action_smoothness_weight = 0.0
        self.reward_distance_scale = 0.0
        pos_error = torch.norm(self.rpos, dim=-1)
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)

        distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)

        reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)
        # uprightness
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # spin reward
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
        )

        misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        traffic_collision = self.check_traffic_collision(self.drone.pos)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
    
    def _get_traffic_observations(self):
        """Get traffic-related observations for the agents."""
        # Get traffic aircraft positions
        traffic_positions = self.traffic_simulator.get_aircraft_positions()
        traffic_velocities = self.traffic_simulator.get_aircraft_velocities()
        
        # For this demo, we'll provide basic statistics about nearby traffic
        if len(traffic_positions) > 0:
            # Find nearest traffic aircraft to each agent
            agent_positions = self.drone.pos
            
            # Calculate distances to all traffic aircraft
            distances = torch.cdist(agent_positions, traffic_positions)
            nearest_distance = distances.min(dim=1)[0]
            traffic_count = torch.full((self.num_envs,), len(traffic_positions), device=self.device)
        else:
            nearest_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
            traffic_count = torch.zeros(self.num_envs, device=self.device)
        
        return torch.stack([nearest_distance, traffic_count.float()], dim=1)
    
    def check_traffic_collision(self, agent_positions: torch.Tensor, safety_radius: float = 2.0):
        """Check for collisions between agents and traffic aircraft."""
        return self.traffic_simulator.check_collision(agent_positions, safety_radius)
    
    def get_traffic_info(self):
        """Get comprehensive traffic information."""
        return self.traffic_simulator.get_aircraft_info()