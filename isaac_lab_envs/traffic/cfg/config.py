from __future__ import annotations
from dataclasses import field
from omni.isaac.lab.utils.configclass import configclass
from isaac_lab_envs.utils.path_planner import GlobalPathPlannerCfg
# 1. 为每一个子配置定义一个独立的 configclass
@configclass
class AreaBoundsCfg:
    xmin: float = -80.0
    xmax: float = 80.0
    ymin: float = -80.0
    ymax: float = 80.0
    grid_size: float = 1.0

@configclass
class TrafficDroneCfg:
    model: str = "firefly"
    safety_radius: float = 1.0
    max_speed: float = 1.0
    min_speed: float = 0.0
    v_pref: float = 1.0 # preferred speed
    arrival_threshold: float = 1.0
    target_num: int = 8
    max_offset_radius: float = 4.0 # maximum offset radius for target generation
    random_safety_radius: bool = False
    random_speed: bool = True

    global_path_planner: GlobalPathPlannerCfg = GlobalPathPlannerCfg(
        algorithm="astar",
        smooth_method="shortcut"
    )
    # local guidance
    lookahead_distance: float = 10.0


@configclass
class TrafficEvtolCfg:
    model: str = "type1"
    safety_radius: float = 10.0
    max_speed: float = 2.0
    min_speed: float = 1.0
    v_pref: float = 2.0 # preferred speed
    arrival_threshold: float = 2.0
    turn_radius: float = 2.0
    course_num: int = 3
    random_speed: bool = True
    random_safety_radius: bool = False


    global_path_planner: GlobalPathPlannerCfg = GlobalPathPlannerCfg(
        algorithm="astar",
        smooth_method="shortcut"
    )

@configclass
class OrcaCfg:
    enable: bool = True
    neighbor_dist: float = 100.0
    time_horizon: float = 11.0
    time_horizon_obst: float = 11.0
    safety_space: float = 1.5
    max_neighbors: int = 10

# 2. 创建顶层的 TrafficCfg，将上面的子配置组合起来
@configclass
class TrafficCfg:
    num_drones: int = 20
    num_evtols: int = 3
    flight_height: float = 20.0
    # 将子配置类作为属性的类型
    area_bounds: AreaBoundsCfg = field(default_factory=AreaBoundsCfg)
    drone: TrafficDroneCfg = field(default_factory=TrafficDroneCfg)
    evtol: TrafficEvtolCfg = field(default_factory=TrafficEvtolCfg)
    orca: OrcaCfg = field(default_factory=OrcaCfg)

    reset_interval: int = 1024


# --- 如何在您的主环境配置中使用 ---
# @configclass
# class MyEnvCfg:
#     traffic: TrafficCfg = field(default_factory=TrafficCfg)