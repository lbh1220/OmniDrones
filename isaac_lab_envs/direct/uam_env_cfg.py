from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.utils import configclass


from isaac_lab_envs.direct.mdp.action import ActionManagerCfg
from isaac_lab_envs.direct.mdp.rewards import RewardCalculatorCfg, RewardCalculator
from isaac_lab_envs.direct.mdp.observations import ObservationProcessorCfg, ObservationProcessor
from isaac_lab_envs.direct.components.map import MapManagerCfg
from isaac_lab_envs.utils.path_planner import GlobalPathPlannerCfg
from isaac_lab_envs.traffic.cfg.config import AreaBoundsCfg, TrafficCfg, OrcaCfg

@configclass
class UamEnvCfg(DirectRLEnvCfg):
    """Configuration for the UAM navigation environment."""

    seed = 42
    
    # environment settings
    episode_length_s = 300.0
    decimation = 10  # env step every 1 sim steps
    num_actions = 2  # 只输出vx,vy
    num_observations = 7  # robot_node(5) + temporal_edges(2) = 7
    num_states = 0
    debug_vis = False
    debug_vis_num_envs = 10  # only visualize the first 5 environments
    is_training = True

    total_timesteps = 10000000

    viewer: ViewerCfg = field(default_factory=lambda: ViewerCfg(
        resolution=(960, 720),
        eye=(8, 0., 6.),  #  <-- 使用非默认值
        lookat=(0., 0., 1.)
    ))
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 20,  # 60Hz simulation
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=8.0, replicate_physics=False)


    
    # drone配置 - 使用原始OmniDrones的配置系统
    drone_model: str = "hummingbird"  # 可以选择: firefly, crazyflie, hummingbird, iris等
    controller: Optional[str] = "PlanarSpeedController"  # 可以设置控制器，如 "lee_controller"
    # task config
    flight_height: float = 20.0
    safety_radius: float = 1.0
    arrival_threshold: float = 2.0
    max_speed: float = 1.0
    min_speed: float = 0.0
    v_pref: float = 1.0 # preferred speed
    area_bounds: AreaBoundsCfg = field(default_factory=lambda: AreaBoundsCfg(
        xmin=-50.0,
        xmax=50.0,
        ymin=-50.0,
        ymax=50.0
    ))
    
    # mdp components
    action_manager: ActionManagerCfg = ActionManagerCfg(
        action_space_type="discrete",
        action_space_num_per_dim=7,
        action_mode="velocity_components"
    )
    # action与reward的初始化方式略有不同，需要分别指定类和配置
    reward_cfg: RewardCalculatorCfg = RewardCalculatorCfg()
    reward_calculator_cls: Type[RewardCalculator] = RewardCalculator
    observation_cfg: ObservationProcessorCfg = ObservationProcessorCfg()
    observation_processor_cls: Type[ObservationProcessor] = ObservationProcessor


        # 原始参数配置
    lidar_range: float = 15.0
    lidar_vfov: tuple[float, float] = (-10.0, 20.0)  # degrees
    lidar_resolution: tuple[int, int] = (36, 4)  # horizontal x vertical
    lidar_attach_yaw_only: bool = True

    
    # 随机化配置
    randomization: Dict = field(default_factory=dict)
    time_encoding: bool = True

    # global planner
    use_global_path: bool = True
    global_path_planner_cfg: GlobalPathPlannerCfg = GlobalPathPlannerCfg()
    # 如果用了global, 则必须有map manager, 否则没有grid map
    # 但是如果有map manager, 可以没有global path planner，
    # map manager
    map_cfg: Optional[MapManagerCfg] = None

    # traffic config
    traffic_sim: Optional[TrafficCfg] = None
    orca: OrcaCfg = field(default_factory=lambda: OrcaCfg(enable=False))
    predict_steps: int = 5
    pred_timestep: float = 2.0

@configclass
class CityUamEnvCfg(UamEnvCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50.0, 50.0),
            border_width=0.0,
            num_rows=2,
            num_cols=2,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(50.0, 50.0),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=10.0,
                    num_obstacles=3,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 15),
                    obstacle_height_range=(25.0, 40.0),
                    platform_width=0.0,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )

    map_cfg: MapManagerCfg = MapManagerCfg()
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_cross_track_coeff = 0.0

@configclass
class OpenAirEnvCfg(UamEnvCfg):
    traffic_sim: TrafficCfg = field(default_factory=lambda: TrafficCfg(
        num_drones=10,
        num_evtols=1,
        flight_height=20.0,
        area_bounds=AreaBoundsCfg(
            xmin=-80.0,
            xmax=80.0,
            ymin=-80.0,
            ymax=80.0
        )
    ))

@configclass
class DyanmicUamEnvCfg(UamEnvCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50.0, 50.0),
            border_width=0.0,
            num_rows=2,
            num_cols=2,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(50.0, 50.0),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=10.0,
                    num_obstacles=3,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 15),
                    obstacle_height_range=(25.0, 40.0),
                    platform_width=0.0,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )

    map_cfg: MapManagerCfg = MapManagerCfg()

    traffic_sim: TrafficCfg = field(default_factory=lambda: TrafficCfg(
        num_drones=10,
        num_evtols=1,
        flight_height=20.0,
        area_bounds=AreaBoundsCfg(
            xmin=-80.0,
            xmax=80.0,
            ymin=-80.0,
            ymax=80.0
        )
    ))