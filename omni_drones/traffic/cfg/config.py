from __future__ import annotations
from dataclasses import field
from omni.isaac.lab.utils.configclass import configclass

# 1. 为每一个子配置定义一个独立的 configclass
@configclass
class AreaBoundsCfg:
    xmin: float = -50.0
    xmax: float = 50.0
    ymin: float = -50.0
    ymax: float = 50.0

@configclass
class TrafficDroneCfg:
    model: str = "firefly"
    safety_radius: float = 1.0
    max_speed: float = 1.0
    arrival_threshold: float = 1.0
    target_num: int = 4

@configclass
class TrafficEvtolCfg:
    model: str = "iris"
    safety_radius: float = 5.0
    max_speed: float = 2.0
    arrival_threshold: float = 2.0
    turn_radius: float = 10.0
    course_num: int = 4

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
    num_drones: int = 50
    num_evtols: int = 3
    flight_height: float = 20.0
    # 将子配置类作为属性的类型
    area_bounds: AreaBoundsCfg = field(default_factory=AreaBoundsCfg)
    drone: TrafficDroneCfg = field(default_factory=TrafficDroneCfg)
    evtol: TrafficEvtolCfg = field(default_factory=TrafficEvtolCfg)
    orca: OrcaCfg = field(default_factory=OrcaCfg)

# --- 如何在您的主环境配置中使用 ---
# @configclass
# class MyEnvCfg:
#     traffic: TrafficCfg = field(default_factory=TrafficCfg)