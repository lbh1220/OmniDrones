from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import TiledCameraCfg, TiledCamera

from isaac_lab_envs.direct.mdp.action import ActionManagerCfg
from isaac_lab_envs.direct.mdp.rewards import RewardManagerCfg, RewardManager
from isaac_lab_envs.direct.mdp.observations import ObservationManagerCfg, ObservationManager
from isaac_lab_envs.direct.components.map import MapManagerCfg
from isaac_lab_envs.direct.components.map import UrbanTerrainCfg
from isaac_lab_envs.utils.path_planner import GlobalPathPlannerCfg
from isaac_lab_envs.traffic.cfg.config import AreaBoundsCfg, TrafficCfg, OrcaCfg, TrafficEvtolCfg, TrafficDroneCfg

@configclass
class UamEnvCfg(DirectRLEnvCfg):
    """Configuration for the UAM navigation environment."""
    name: str = "uam_env"
    seed = 42
    num_envs = 128
    # environment settings
    episode_length_s = 300.0
    decimation = 10  # env step every 1 sim steps
    num_actions = 2  # 只输出vx,vy
    num_observations = 7  # robot_node(5) + temporal_edges(2) = 7
    num_states = 0
    debug_vis = False
    debug_vis_velocity_scale = 50.0
    debug_vis_num_envs = 10  # only visualize the first 5 environments
    is_training = True

    use_skrl_metrics = True # if use skrl, metrics will be recorded in different way

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

    urban_terrain: UrbanTerrainCfg = UrbanTerrainCfg()
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
    dv_limit: float = 0.2
    dtheta_limit_deg: float = 5.0
    area_bounds: AreaBoundsCfg = field(default_factory=lambda: AreaBoundsCfg(
        xmin=-50.0,
        xmax=50.0,
        ymin=-50.0,
        ymax=50.0,
        grid_size=1.0
    ))
    
    # mdp components
    action_manager: ActionManagerCfg = ActionManagerCfg(
        action_space_type="discrete",
        action_space_num_per_dim=7,
        action_mode="velocity_components"
    )
    # action与reward的初始化方式略有不同，需要分别指定类和配置
    reward_cfg: RewardManagerCfg = RewardManagerCfg(modules=["nav"])
    reward_calculator_cls: Type[RewardManager] = RewardManager
    observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node"])
    observation_processor_cls: Type[ObservationManager] = ObservationManager


    # 原始参数配置
    # lidar配置
    lidar_range: float = 15.0
    lidar_vfov: tuple[float, float] = (-90.0, 60.0)  # degrees
    lidar_resolution: tuple[int, int] = (36, 15)  # horizontal x vertical
    lidar_attach_yaw_only: bool = True
    # camera conf
    cameras: Dict[str, TiledCameraCfg] = field(default_factory=lambda: {
            # 1. 前视相机 (Front Camera)
            "front_cam": TiledCameraCfg(
                # 这里的 prim_path 需要使用 regex 匹配所有环境的无人机
                # 注意：{ENV_REGEX} 是占位符，需要在 Env 类中处理，或者直接写死正则
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link/front_cam",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.1, 0.0, 0.0),  # 安装在机头
                    rot=(1.0, 0.0, 0.0, 0.0), # Quaternion (w, x, y, z) - 假设是前向
                    convention="ros", # 或者 "opengl"，决定坐标系方向
                ),
                data_types=["rgb", "depth"], # 输出 RGB 和 深度图
                width=84,
                height=84,
                spawn=None, # 我们假设这里是虚拟挂载，不生成实际 USD Prim，或者依靠 Robot 定义
            ),
            
            # # 2. 下视相机 (Down Camera)
            # "down_cam": TiledCameraCfg(
            #     prim_path="/World/envs/env_.*/Hummingbird_0/base_link/down_cam",
            #     offset=TiledCameraCfg.OffsetCfg(
            #         pos=(0.0, 0.0, -0.05),
            #         rot=(0.707, 0.0, 0.707, 0.0), # 旋转90度向下
            #         convention="ros",
            #     ),
            #     data_types=["rgb"], # 只需要 RGB
            #     width=64,
            #     height=64,
            #     spawn=None, # 我们假设这里是虚拟挂载，不生成实际 USD Prim，或者依靠 Robot 定义
            # ),
        })
    
    # 随机化配置
    randomization: Dict = field(default_factory=dict)
    time_encoding: bool = True

    # global planner
    use_global_path: bool = True
    global_path_planner_cfg: GlobalPathPlannerCfg = GlobalPathPlannerCfg(max_waypoints=20)
    # 如果用了global, 则必须有map manager, 否则没有grid map
    # 但是如果有map manager, 可以没有global path planner，
    # map manager
    map_cfg: Optional[MapManagerCfg] = None

    # traffic config
    traffic_sim: Optional[TrafficCfg] = None
    orca: OrcaCfg = field(default_factory=lambda: OrcaCfg(enable=False))
    predict_steps: int = 5
    pred_timestep: float = 2.0


    env_scale = 1.0

    dynamic_obstacle_num = 5 # for navrl
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_approaching_reward_coeff = 0.0
    rew_smoothness_coeff = -0.0
    rew_evtol_future_penalty = -0.0
    rew_drone_future_penalty = -0.0
    rew_time_penalty = 0.0
    rew_drones_threshold_factor = 2.0
    rew_drones_decay_factor = 0.667
    rew_evtols_threshold_factor = 1.5
    rew_evtols_decay_factor = 1.0
    ## drones in previous, 2.0, 0.667; evtols in previous, 1.5, 0.9
    rew_cross_track_coeff = 0.0

    rew_ttc_threshold = 10.0
    rew_ttc_alpha = 0.0
    rew_ttc_beta = 5.0
    rew_ttc_idle_penalty = 0.0
    rew_patience_coeff = 0.0


@configclass
class CityUamEnvCfg(UamEnvCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(30, 30),
            border_width=0.0,
            num_rows=3,
            num_cols=3,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(30, 30),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=5.0,
                    num_obstacles=4,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 8),
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

    reward_cfg: RewardManagerCfg = RewardManagerCfg(modules=["nav", "cross_track"])
    observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node", "lidar"])


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
            ymax=80.0,
            grid_size=1.0
        )
    ))

    reward_cfg: RewardManagerCfg = RewardManagerCfg(modules=["nav"])
    observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node", "traffic_spatial_edges"])
@configclass
class DyanmicUamEnvCfg(UamEnvCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            border_width=0.0,
            num_rows=2,
            num_cols=2,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(50, 50),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=4.0,
                    num_obstacles=4,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 12),
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
            ymax=80.0,
            grid_size=1.0
        ),
        evtol=TrafficEvtolCfg(safety_radius=5.0),
        drone=TrafficDroneCfg(safety_radius=1.0),
        ))

    reward_cfg: RewardManagerCfg = RewardManagerCfg(modules=["nav", "traffic_future", "cross_track"])
    observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node", "lidar", "traffic_spatial_edges"])
    
    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_smoothness_coeff = -0.0
    rew_evtol_future_penalty = -0.0
    rew_drone_future_penalty = -0.0
    rew_time_penalty = 0.0
    rew_drones_threshold_factor = 2.0
    rew_drones_decay_factor = 0.667
    rew_evtols_threshold_factor = 2.0
    rew_evtols_decay_factor = 0.9
    ## drones in previous, 2.0, 0.667; evtols in previous, 1.5, 0.9
    rew_cross_track_coeff = 0.0

    rew_ttc_threshold = 10.0
    rew_ttc_alpha = 0.0
    rew_ttc_beta = 5.0
    rew_ttc_idle_penalty = 0.0
    rew_patience_coeff = 0.0



@configclass
class NavrlEnvCfg(UamEnvCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            border_width=0.0,
            num_rows=2,
            num_cols=2,
            horizontal_scale=0.5,
            vertical_scale=1.0,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(50, 50),
                    horizontal_scale=0.5,
                    vertical_scale=1.0,
                    border_width=3.0,
                    num_obstacles=4,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(5, 12),
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
        num_evtols=0,
        flight_height=20.0,
        area_bounds=AreaBoundsCfg(
            xmin=-80.0,
            xmax=80.0,
            ymin=-80.0,
            ymax=80.0,
            grid_size=1.0
        ),
        evtol=TrafficEvtolCfg(safety_radius=5.0),
        drone=TrafficDroneCfg(safety_radius=1.0),
        ))

    reward_cfg: RewardManagerCfg = RewardManagerCfg(modules=["navrl"])
    observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node", "lidar", "dynamic_obstacle"])
    # observation_cfg: ObservationManagerCfg = ObservationManagerCfg(modules=["robot_node", "lidar", "traffic_spatial_state"])

    arrival_threshold = -1.0 # navrl不会用到reach goal的termination, 所以这里设置为-1.0

    dynamic_obstacle_num = 5
    use_global_path = False


    rew_success = 15.0
    rew_collision = -16.0
    rew_potential = 0.5
    rew_smoothness_coeff = -0.0
    rew_evtol_future_penalty = -0.0
    rew_drone_future_penalty = -0.0
    rew_time_penalty = 0.0
    rew_drones_threshold_factor = 2.0
    rew_drones_decay_factor = 0.667
    rew_evtols_threshold_factor = 1.5
    rew_evtols_decay_factor = 1.0
    ## drones in previous, 2.0, 0.667; evtols in previous, 1.5, 0.9
    rew_cross_track_coeff = 0.0

    rew_ttc_threshold = 10.0
    rew_ttc_alpha = 0.0
    rew_ttc_beta = 5.0
    rew_ttc_idle_penalty = 0.0
    rew_patience_coeff = 0.0

