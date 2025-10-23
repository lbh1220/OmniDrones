from this import d
import numpy as np
import rvo2
import time
from isaac_lab_envs.traffic.utils.state import TrafficState
from typing import Dict


class ORCA:
    def __init__(self, config):
        """
        NHORCA (Non-Holonomic ORCA) 策略
        基于 ORCA 算法，但考虑了非完整约束
        """
        self.config = config
        self.name = 'ORCA'
        self.time_step = 0.16
        
        # ORCA 参数
        self.neighbor_dist = config.orca.neighbor_dist
        self.max_neighbors = config.orca.max_neighbors
        self.time_horizon = config.orca.time_horizon
        self.time_horizon_obst = config.orca.time_horizon_obst
        self.safety_space = config.orca.safety_space
        
        # RVO2 模拟器
        self.sim = None

        self.static_obstacles = None
    
    def set_static_obstacles(self, obstacles):
        """
        设置静态障碍物
        """
        self.static_obstacles = obstacles


    def predict(self, self_state: TrafficState, other_aircraft_state: TrafficState=None, dt: float=None):
        """
        集中式的计算所有self_state的目标速度
        """
        # 设置参数
        if dt is not None:
            self.time_step = dt
        self.sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            1.0,# default radius
            1.0 # default max speed
        )
        agent_idx = 0
        for i in range(self_state.num_aircraft):
            self.sim.addAgent(
                (self_state.positions[i][0], self_state.positions[i][1]),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self_state.safety_radius[i] + self.safety_space,
                self_state.max_speed[i],
                (self_state.velocities[i][0], self_state.velocities[i][1])
            )
            self.sim.setAgentPrefVelocity(agent_idx, (self_state.velocity_commands[i][0], self_state.velocity_commands[i][1]))
            agent_idx += 1

        if other_aircraft_state is not None:
            for i in range(other_aircraft_state.num_aircraft):
                self.sim.addAgent(
                    (other_aircraft_state.positions[i][0], other_aircraft_state.positions[i][1]),
                    0.1,
                    0,
                    self.time_horizon,
                    self.time_horizon_obst,
                    other_aircraft_state.safety_radius[i] + self.safety_space,
                    other_aircraft_state.max_speed[i], # set small speed
                    (other_aircraft_state.velocities[i][0], other_aircraft_state.velocities[i][1])
                )
            self.sim.setAgentPrefVelocity(agent_idx, (other_aircraft_state.velocity_commands[i][0], other_aircraft_state.velocity_commands[i][1]))
            # self.sim.setAgentPrefVelocity(agent_idx, (0, 0))
            agent_idx += 1


        try:
            if self.static_obstacles is not None:
                for hull in self.static_obstacles:
                    if len(hull) >= 3:
                        obstacle_vertices = [(float(point[0]), float(point[1])) for point in hull]
                        self.sim.addObstacle(obstacle_vertices)
        except Exception as e:
            print(f"[ERROR][traffic]Error adding static obstacles: {e}")
        try:
            self.sim.doStep()
        
            for i in range(self_state.num_aircraft):
                vx,vy = self.sim.getAgentVelocity(i)
                self_state.velocity_commands[i][0] = vx
                self_state.velocity_commands[i][1] = vy
        except Exception as e:
            print(f"[ERROR][traffic]Error doing step in ORCA: {e}")
        # do not cal speed for evtol states
        
