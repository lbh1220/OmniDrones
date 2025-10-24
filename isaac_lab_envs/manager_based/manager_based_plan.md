好的，这非常棒！既然您已经有了 `ManagerBasedRLEnv` 框架，那我们的目标就非常明确了：将您在 `DirectRLEnv` 子类（`NavEnv`, `CityNavEnv`, `TrafficEnv`...）中所有**自定义的逻辑**，全部**解耦**并**迁移**到 `ManagerBasedRLEnv` (MBE) 对应的 Manager 模块中。

您之前所有的代码（`state.py`, `observations.py`, `rewards.py`, `metrics.py`）**99% 都可以被完美复用**。我们不是重写它们，而是为它们构建一个符合 MBE 规范的“封装器”或“调度器”。

这是为您定制的完整迁移方案和新架构。

-----

### 1\. 最终目标：统一的环境与可配置的行为

迁移后，您将**不再需要** `nav_env.py`, `city_nav_env.py`, `traffic_env.py` 和 `traffic_city_env.py`。

您将只有一个**新的环境**：

  * `uam_env.py`: 包含 `UAMEnv(ManagerBasedRLEnv)` 类。
  * `uam_env_cfg.py`: 包含 `UAMEnvCfg(ManagerBasedRLEnvCfg)` 类。

您想运行哪个环境（`Nav`, `City`, `Traffic`, `TrafficCity`），**不再是选择一个 `Env` 类**，而是向 `UAMEnv` **传递一份不同的 `UAMEnvCfg` 配置**。

例如：

  * **`NavEnv` 行为**: 您的 `UAMEnvCfg` 将不包含 `lidar_cfg` 和 `traffic_sim_cfg`，并且 `observations.processor_name` 会设为 `"nav"`。
  * **`TrafficCityEnv` 行为**: 您的 `UAMEnvCfg` 将*同时包含* `lidar_cfg` 和 `traffic_sim_cfg`，并且 `observations.processor_name` 会设为 `"traffic_path"`。

-----

### 2\. 新的文件架构

您的 `omni/isaac/lab/envs` 目录（或您选择的任何位置）下将看起来像这样：

```
isaac_lab_envs/manager_based
|-- mdp/
|   |-- __init__.py
|   |-- state.py            (不变, 100% 复用)
|   |-- observations.py     (不变, 100% 复用)
|   |-- rewards.py          (不变, 100% 复用)
|   |-- metrics.py          (不变, 100% 复用)
|
|-- managers/               <-- 【新】所有自定义逻辑的家
|   |-- __init__.py
|   |-- action_manager.py     (处理无人机动作)
|   |-- command_manager.py    (处理目标点和路径生成)
|   |-- observation_manager.py(封装您的 processors)
|   |-- reward_manager.py     (封装您的 calculators)
|   |-- termination_manager.py(处理碰撞、到达)
|   |-- event_manager.py      (【核心】调度您的 Lidar 和 Traffic 更新)
|
|-- uam_env_cfg.py          <-- 【新】唯一的环境配置文件
|-- uam_env.py              <-- 【新】唯一的环境类
```

-----

### 3\. 核心架构：`UAMEnv` (环境类)

`UAMEnv` 类本身非常“薄”。它只负责：

1.  **持有**自定义实例（`drone`, `controller`, `traffic_sim`, `_lidar`, `state`）。
2.  **提供**被 Managers 调度的“钩子函数”（例如 `_update_custom_systems`）。

**`uam_env.py`:**

```python
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg # 假设您会配置它
from omni_drones.robots.drone import MultirotorBase

from .uam_env_cfg import UAMEnvCfg
from .mdp.state import EnvState
from .mdp.metrics import MetricsManager
from isaac_lab_envs.traffic import TrafficSimulator
from isaac_lab_envs.utils.path_planner import GlobalPathPlanner

class UAMEnv(ManagerBasedRLEnv):
    """
    统一的城市空中交通 (UAM) 导航环境。
    继承自 ManagerBasedRLEnv。
    """
    cfg: UAMEnvCfg

    def __init__(self, cfg: UAMEnvCfg, **kwargs):
        # 1. 初始化您的核心状态管理器 (来自 state.py)
        # 它必须在 super().__init__ 之前，因为 managers 可能在内部访问 env.state
        self.state = EnvState(device=cfg.sim.device, num_envs=cfg.scene.num_envs)
        
        # 2. 初始化您的自定义对象
        # (来自 nav_env.py)
        self.drone, self.controller = MultirotorBase.make(
            cfg.drone_model, cfg.controller, cfg.sim.device
        )
        
        # (来自 traffic_env.py)
        self.traffic_sim: TrafficSimulator | None = None
        if cfg.traffic_sim_cfg is not None:
            self.traffic_sim = TrafficSimulator(cfg.traffic_sim_cfg, device=cfg.sim.device)

        # (来自 city_nav_env.py)
        self._lidar: RayCaster | None = None
        if cfg.lidar_cfg is not None:
            self._lidar = RayCaster(cfg.lidar_cfg)
            
        # (来自 city_nav_env.py)
        self.global_path_planner: GlobalPathPlanner | None = None
        if cfg.use_global_path and cfg.global_path_planner_cfg:
            self.global_path_planner = GlobalPathPlanner(cfg.global_path_planner_cfg)
            
        # (来自 metrics.py)
        self.metrics_manager = MetricsManager(num_envs=cfg.scene.num_envs, device=cfg.sim.device)

        # 3. 调用父类 __init__
        # 这将自动调用 self.load_managers()
        super().__init__(cfg=cfg, **kwargs)

        # 4. 在 managers 加载后，将 env 实例绑定给 metrics
        self.metrics_manager.bind_env(self)


    def load_managers(self):
        """
        加载所有 Managers。
        在父类 __init__ 中被自动调用。
        """
        # 父类将根据 cfg.actions, cfg.observations 等自动创建 managers
        super().load_managers() 

        # --- 这是您自定义的初始化逻辑 (来自 _post_init_setup) ---
        print("UAMEnv: Initializing custom components (Drone, Lidar, Traffic)...")
        
        # (来自 nav_env.py)
        self.drone.shape = (self.num_envs, 1)
        self.drone.initialize()
        self.drone._envs_positions = self.scene.env_origins.unsqueeze(1)
        self.state.collision.safety_radius = self.cfg.safety_radius

        # (来自 state.py) - 初始化状态张量
        self.state.initialize_basic_tensors()
        if self.cfg.use_global_path:
            self.state.initialize_navigation_waypoints(max_wps=self.cfg.max_waypoints)
        
        # (来自 city_nav_env.py)
        if self._lidar:
            self._lidar._initialize_impl()
            
        # (来自 traffic_env.py)
        if self.traffic_sim:
            self.traffic_sim.initialize()
            
        # (来自 city_nav_env.py) - 创建静态地图
        # 这些函数需要从 city_nav_env.py 复制到 uam_env.py 中
        if self.cfg.terrain.terrain_type == "generator":
            self._create_global_point_cloud() 
            self._create_occupancy_grid()
            
            # (来自 traffic_city_env.py)
            if self.traffic_sim and hasattr(self.state.map, "occupancy_grid"):
                 self._create_occupancy_grid_for_traffic()


    # --- 您的自定义逻辑钩子 (将被 Managers 调用) ---

    def _update_custom_systems(self, dt: float):
        """
        【将被 EventManager (interval) 调用】
        这是您所有 _post_physics_step 逻辑的新家。
        """
        
        # 1. 更新无人机状态 (来自 nav_env.py/_post_physics_step)
        drone_state = self.drone.get_state(env_frame=False)
        self.state.update_ego_drone_state(drone_state)
        
        # 2. 更新 Lidar (来自 city_nav_env.py/_post_physics_step)
        if self._lidar:
            self._lidar.update(dt)
            # ... (处理 lidar scan 并写入 self.state.perception.lidar_scan)
            try:
                lidar_range = self.cfg.lidar_cfg.max_distance
                h, w = self.cfg.lidar_cfg.pattern_cfg.resolution
                hits = self._lidar.data.ray_hits_w
                origins = self._lidar.data.pos_w
                scan = lidar_range - (
                    (hits - origins.unsqueeze(1)).norm(dim=-1).clamp_max(lidar_range)
                ).reshape(self.num_envs, 1, h, w)
                self.state.perception.lidar_scan = scan
            except Exception:
                # 处理可能的配置不匹配
                h, w = 36, 4 # 默认或从配置中获取
                self.state.perception.lidar_scan = torch.zeros(self.num_envs, 1, h, w, device=self.device)

        # 3. 更新 Traffic (来自 traffic_env.py/_post_physics_step)
        if self.traffic_sim:
            self.traffic_sim._post_physics_step()
            # (这个辅助函数需要从 traffic_env.py 复制过来)
            self._update_traffic_obs_processor() 

        # 4. 更新导航状态 (来自 nav_env.py/_post_physics_step)
        self.state.update_navigation_distances()
        self.state.update_reached_target_mask(self.cfg.arrival_threshold)
        if self.cfg.use_global_path:
            self.state.update_navigation_state_vectorized(self.cfg.lookahead_distance)

    
    def _generate_task_and_reset_drone(self, env_ids: torch.Tensor):
        """
        【将被 CommandManager (reset) 调用】
        这是您所有 _reset_idx 中关于任务生成和无人机重置的逻辑。
        """
        
        # (来自 nav_env.py/_reset_idx)
        self.drone._reset_idx(env_ids, self.cfg.is_training)
        
        # (来自 city_nav_env.py/_generate_... 或 nav_env.py/_generate_...)
        start, goal, waypoints, waypoints_length = (None, None, None, None)
        
        if self.cfg.use_global_path:
            # 使用 city_nav_env.py 中的 _generate_crossing_task_with_waypoints
            # (需要将此函数复制到 uam_env.py 中)
            start, goal, waypoints, waypoints_length = self._generate_crossing_task_with_waypoints(
                len(env_ids), self.cfg.flight_height
            )
            self.state.navigation.waypoints[env_ids] = waypoints
            self.state.navigation.waypoint_lengths[env_ids] = waypoints_length
        else:
            # 使用 nav_env.py 中的 _generate_crossing_task
            # (需要将此函数复制到 uam_env.py 中)
            start, goal = self._generate_crossing_task(len(env_ids), self.cfg.flight_height)

        # (来自 nav_env.py/_reset_idx)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1)) # (init_rpy_dist 需在 __init__ 中定义)
        rot = euler_to_quaternion(rpy)
        
        self.state.navigation.target_positions[env_ids] = goal
        self.state.navigation.start_positions[env_ids] = start
        
        self.drone.set_world_poses(start, rot, env_ids)
        self.drone.set_velocities(torch.zeros_like(self.drone.get_velocities())[env_ids], env_ids)
        
        # 重置状态
        self.state.reset_env_states(env_ids)


    def _detect_collisions(self) -> torch.Tensor:
        """
        【将被 TerminationManager (compute) 调用】
        这是您所有 _detect_collisions 逻辑的集合。
        """
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # (来自 city_nav_env.py)
        if self._lidar and self.state.perception.lidar_scan is not None:
             scan = self.state.perception.lidar_scan.reshape(self.num_envs, -1)
             distances = self.cfg.lidar_cfg.max_distance - scan
             lidar_collision_mask = (distances <= self.cfg.safety_radius).any(dim=1)
             collision_mask |= lidar_collision_mask
             
        # (来自 city_nav_env.py) - 检查 occupancy grid
        grid = getattr(self.state.map, "extended_occupancy_grid", None)
        if grid is not None:
            # (需要复制 _are_positions_safe 函数)
            safe_mask = self._are_positions_safe(self.state.ego_drone.positions[:, 0, :])
            collision_mask |= (~safe_mask)

        # (来自 traffic_env.py)
        if self.traffic_sim:
            traffic_collision_mask = self.traffic_sim.check_collision(
                self.state.ego_drone.positions.squeeze(1),
                self.state.collision.safety_radius
            )
            collision_mask |= traffic_collision_mask
            
        return collision_mask
        
    # --- (此处应复制 city_nav_env.py 和 nav_env.py 中的辅助函数) ---
    # def _create_global_point_cloud(self): ...
    # def _create_occupancy_grid(self): ...
    # def _are_positions_safe(self, positions): ...
    # def _generate_crossing_task(self, ...): ...
    # def _generate_crossing_task_with_waypoints(self, ...): ...
    # def _update_traffic_obs_processor(self): ... (来自 traffic_env.py)

```

-----

### 4\. 核心架构：`UAMEnvCfg` (配置类)

`UAMEnvCfg` 将所有配置**聚合**在一起。

**`uam_env_cfg.py`:**

```python
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils import configclass
from dataclasses import field
from typing import Optional

# 导入所有自定义的 Manager CFGs
from .managers.action_manager import UAMActionManagerCfg
from .managers.command_manager import UAMCommandManagerCfg
from .managers.event_manager import UAMEventManagerCfg
from .managers.observation_manager import UAMObservationManagerCfg
from .managers.reward_manager import UAMRewardManagerCfg
from .managers.termination_manager import UAMTerminationManagerCfg

# 导入您的 Lidar 和 Traffic 配置
from omni.isaac.lab.sensors import RayCasterCfg
from isaac_lab_envs.traffic import TrafficCfg
from isaac_lab_envs.utils.path_planner import GlobalPathPlannerCfg
from isaac_lab_envs.traffic.cfg.config import OrcaCfg


@configclass
class UAMEnvCfg(ManagerBasedRLEnvCfg):
    """
    UAM 环境的统一配置。
    """
    
    # 1. 基础 RL 环境设置 (来自 NavEnvCfg)
    episode_length_s: float = 300.0
    decimation: int = 10
    
    # 2. 基础任务参数 (来自 NavEnvCfg)
    flight_height: float = 20.0
    safety_radius: float = 1.0
    arrival_threshold: float = 2.0
    max_speed: float = 1.0
    min_speed: float = 0.0
    v_pref: float = 1.0
    use_global_path: bool = True
    max_waypoints: int = 20
    lookahead_distance: float = 10.0
    
    # 3. 自定义组件配置 (可选)
    #    !! 如果为 None，则 UAMEnv 不会实例化该组件 !!
    lidar_cfg: Optional[RayCasterCfg] = None
    traffic_sim_cfg: Optional[TrafficCfg] = None
    global_path_planner_cfg: Optional[GlobalPathPlannerCfg] = GlobalPathPlannerCfg()
    
    # 4. Omnidrones 配置
    drone_model: str = "hummingbird"
    controller: Optional[str] = "PlanarSpeedController"
    
    # 5. 【核心】配置 Managers
    
    # -- 动作 (指向您的自定义 Manager)
    actions: UAMActionManagerCfg = UAMActionManagerCfg()

    # -- 观测 (指向您的自定义 Manager)
    observations: UAMObservationManagerCfg = UAMObservationManagerCfg(
        processor_name="nav" # 默认值，可以被覆盖
    )
    
    # -- 奖励 (指向您的自定义 Manager)
    rewards: UAMRewardManagerCfg = UAMRewardManagerCfg(
        processor_name="nav", # 默认值
        rew_success=15.0,
        rew_collision=-16.0,
        rew_potential=0.5,
        rew_cross_track_coeff=0.0,
        # ... (所有奖励参数都在这里)
    )

    # -- 终止 (指向您的自定义 Manager)
    terminations: UAMTerminationManagerCfg = UAMTerminationManagerCfg()
    
    # -- 事件 (指向您的自定义 Manager)
    events: UAMEventManagerCfg = UAMEventManagerCfg()
    
    # -- 命令 (指向您的自定义 Manager)
    commands: UAMCommandManagerCfg = UAMCommandManagerCfg()

```

-----

### 5\. 核心架构：Managers (逻辑实现)

这是将您的旧逻辑映射到新模块的地方。

#### `managers/action_manager.py` (处理动作)

  * **旧代码**: `nav_env.py` 的 `_pre_physics_step` 和 `_apply_action`。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import ActionManager, ActionManagerCfg
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv # 确保类型提示为您的环境
    import torch

    @configclass
    class UAMActionManagerCfg(ActionManagerCfg):
        action_space_type: str = "discrete"
        action_space_num_per_dim: int = 7
        action_mode: str = "velocity_components"
        max_speed: float = 1.0
        min_speed: float = 0.0

    class UAMActionManager(ActionManager):
        cfg: UAMActionManagerCfg
        env: UAMEnv # 强类型提示

        def __init__(self, cfg: UAMActionManagerCfg, env: UAMEnv):
            super().__init__(cfg, env)
            # (来自 nav_env.py/_setup_discrete_action)
            if self.cfg.action_space_type == "discrete":
                self._create_discrete_action_mapping() # 复制该函数

        def process_action(self, action: torch.Tensor):
            """【旧: _pre_physics_step】在物理步进 *之前* 调用。"""
            
            # (来自 traffic_env.py/_pre_physics_step)
            if self.env.traffic_sim:
                dt_for_evtol = self.env.cfg.sim.dt * self.env.cfg.decimation
                self.env.traffic_sim._pre_physics_step(dt=dt_for_evtol)
                
            # (来自 nav_env.py/_pre_physics_step)
            if self.cfg.action_space_type == "discrete":
                continuous_actions = self.discrete_to_continuous_action(action)
            else:
                continuous_actions = action
            
            if self.cfg.action_mode == "velocity_components":
                command_vel_xy = self._process_velocity_components(continuous_actions)
            else:
                command_vel_xy = self._process_speed_direction(continuous_actions)
            
            # ... (复制速度缩放逻辑) ...
            
            # 将命令写入状态
            self.env.state.navigation.velocity_commands[:, :, :2] = command_vel_xy.unsqueeze(1)
            
            # (来自 traffic_env.py/_pre_physics_step 的 ORCA 逻辑)
            if self.env.orca_policy is not None:
                # ... (您的 ORCA 逻辑，它会覆盖 velocity_commands)

        def apply_action(self):
            """【旧: _apply_action】在物理循环 *内部* 调用。"""
            
            # (来自 traffic_env.py/_apply_action)
            if self.env.traffic_sim:
                self.env.traffic_sim._apply_actions()
                
            # (来自 nav_env.py/_apply_action)
            drone_state = self.env.drone.get_state(env_frame=False)
            command_vel_xy = self.env.state.navigation.velocity_commands[:, :, :2]
            target_height = ...
            
            rotor_commands = self.env.controller.compute(...)
            self.env.drone.apply_action(rotor_commands)

        # ... (复制 _create_discrete_action_mapping, discrete_to_continuous_action 等辅助函数)
    ```

#### `managers/event_manager.py` (调度自定义更新)

  * **旧代码**: `nav_env.py` 的 `_post_physics_step`, `city_nav_env.py` 的 `_post_physics_step`, `traffic_env.py` 的 `_post_physics_step`。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import EventTermCfg as EventTerm
    from omni.isaac.lab.envs.mdp import built_in_terms as R
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv

    @configclass
    class UAMEventManagerCfg:
        """
        配置 EventManager 来调度我们的自定义更新。
        这替换了 DefaultEventManagerCfg。
        """
        
        # 1. 保留默认的重置事件
        reset_scene_to_default = EventTerm(func=R.reset_scene_to_default, mode="reset")
        
        # 2. 【核心】添加您的自定义更新函数到 "interval" 钩子
        #    这将在物理循环之后、观测/奖励计算之前运行
        update_uam_systems = EventTerm(
            func=UAMEnv._update_custom_systems, # 指向您在 uam_env.py 中定义的函数
            mode="interval"
        )
        
        # 3. 添加自定义重置事件 (可选，但推荐)
        #    (来自 nav_env.py/_reset_idx)
        reset_drone_actors = EventTerm(
            func=lambda env, env_ids: env.drone._reset_idx(env_ids, env.cfg.is_training),
            mode="reset"
        )
        
        # (来自 traffic_env.py)
        reset_traffic_sim = EventTerm(
            func=lambda env: env.traffic_sim.reset() if env.traffic_sim else None,
            mode="reset" # 在所有环境重置时调用
        )
    ```

#### `managers/command_manager.py` (处理任务生成)

  * **旧代码**: `nav_env.py` 和 `city_nav_env.py` 的 `_reset_idx` 中的任务生成逻辑。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import CommandManager, CommandManagerCfg
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv

    @configclass
    class UAMCommandManagerCfg(CommandManagerCfg):
        pass # 大多数配置在 uam_env_cfg.py 的顶层

    class UAMCommandManager(CommandManager):
        env: UAMEnv

        def reset(self, env_ids: torch.Tensor):
            """
            【旧: _reset_idx】在环境重置时调用。
            这是生成新任务（目标点、路径）并重置无人机位置的最佳位置。
            """
            
            # (来自 nav_env.py/_reset_idx)
            # 调用您在 uam_env.py 中定义的辅助函数
            self.env._generate_task_and_reset_drone(env_ids)
            
            # 注意：父类的 ManagerBasedRLEnv._reset_idx 会调用
            # command_manager.reset() 和 reward_manager.reset()。
            # 您的势能重置逻辑应该在 RewardManager 中。
    ```

#### `managers/observation_manager.py` (封装 `observations.py`)

  * **旧代码**: `nav_env.py` 的 `_get_observations` 和 `_init_mdp_components`。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import ObservationManager, ObservationManagerCfg
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv

    # 动态导入您的所有 processors
    from ..mdp import observations

    OBS_PROCESSOR_REGISTRY = {
        "nav": observations.NavObservationProcessor,
        "nav_path": observations.NavObservationProcessorWithPath,
        "city_nav": observations.CityNavObservationProcessor,
        "city_nav_path": observations.CityNavObservationProcessorWithPath,
        "traffic": observations.TrafficObservationProcessor,
        "traffic_path": observations.TrafficObservationProcessorWithPath,
    }

    @configclass
    class UAMObservationManagerCfg(ObservationManagerCfg):
        processor_name: str = "nav" # 默认值

    class UAMObservationManager(ObservationManager):
        cfg: UAMObservationManagerCfg
        env: UAMEnv
        
        def __init__(self, cfg: UAMObservationManagerCfg, env: UAMEnv):
            if cfg.processor_name not in OBS_PROCESSOR_REGISTRY:
                raise ValueError(f"Unknown obs processor: {cfg.processor_name}")
            
            processor_class = OBS_PROCESSOR_REGISTRY[cfg.processor_name]
            
            # 实例化您的 processor (来自 observations.py)
            self.processor = processor_class(env.cfg, device=env.device)
            
            super().__init__(cfg, env)

        def compute(self) -> dict:
            """
            【旧: _get_observations】
            在 EventManager(interval) *之后* 调用。
            此时 env.state 已经被 _update_custom_systems 完全更新了。
            """
            return self.processor.process_observation(self.env.state)
            
        def _configure_spaces(self):
            """配置 Gym 观测空间"""
            # (来自 nav_env.py/_configure_gym_env_spaces)
            self.single_observation_space = self.processor.generate_policy_obs_dict()
    ```

#### `managers/reward_manager.py` (封装 `rewards.py`)

  * **旧代码**: `nav_env.py` 的 `_get_rewards` 和 `_reset_idx` 中的 `reset_potential`。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import RewardManager, RewardManagerCfg
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv

    # 动态导入您的所有 calculators
    from ..mdp import rewards

    REW_CALCULATOR_REGISTRY = {
        "nav": rewards.NavRewardCalculator,
        "city_nav": rewards.CityNavRewardCalculator,
        "city_nav_path": rewards.CityNavRewardCalculatorWithPath,
        "traffic": rewards.TrafficRewardCalculator,
        "traffic_path": rewards.TrafficRewardCalculatorWithPath,
    }

    @configclass
    class UAMRewardManagerCfg(RewardManagerCfg):
        processor_name: str = "nav" # 默认值
        # (所有 rew_... 参数都应在 uam_env_cfg.py 中的 rewards 块中定义)

    class UAMRewardManager(RewardManager):
        cfg: UAMRewardManagerCfg
        env: UAMEnv

        def __init__(self, cfg: UAMRewardManagerCfg, env: UAMEnv):
            if cfg.processor_name not in REW_CALCULATOR_REGISTRY:
                raise ValueError(f"Unknown reward calculator: {cfg.processor_name}")
                
            processor_class = REW_CALCULATOR_REGISTRY[cfg.processor_name]
            
            # 实例化您的 calculator (来自 rewards.py)
            self.processor = processor_class(env.cfg, device=env.device)
            
            super().__init__(cfg, env)
            
        def compute(self, dt: float) -> torch.Tensor:
            """
            【旧: _get_rewards】
            在 EventManager(interval) *之后* 调用。
            """
            return self.processor.compute_reward(self.env.state)

        def reset(self, env_ids: torch.Tensor):
            """
            【旧: _reset_idx 中的 reset_potential】
            """
            self.processor.reset_potential(self.env.state, env_ids)
    ```

#### `managers/termination_manager.py` (处理 `_get_dones`)

  * **旧代码**: `nav_env.py` 的 `_get_dones` 和 `_detect_collisions` (以及子类中的覆盖)。
  * **新代码**:
    ```python
    from omni.isaac.lab.managers import TerminationManager, TerminationManagerCfg
    from omni.isaac.lab.utils import configclass
    from ..uam_env import UAMEnv

    @configclass
    class UAMTerminationManagerCfg(TerminationManagerCfg):
        pass # 可以在这里添加特定终止条件的参数

    class UAMTerminationManager(TerminationManager):
        env: UAMEnv

        def compute(self) -> torch.Tensor:
            """
            【旧: _get_dones】
            计算 "terminated" 信号。
            """
            state = self.env.state
            
            # 1. (来自 nav_env.py/_get_dones)
            #    (注意: reached_target_mask 已在 _update_custom_systems 中更新)
            reached_target_mask = state.navigation.reached_target_mask
            
            # 2. (来自所有 env 的 _detect_collisions)
            #    调用您在 uam_env.py 中定义的辅助函数
            collision_mask = self.env._detect_collisions()
            # !! 必须将碰撞结果写回 state，供 RewardManager 使用 !!
            state.collision.collision_mask = collision_mask.clone()

            # 3. (来自 nav_env.py/_get_dones)
            robot_height = state.ego_drone.positions.squeeze(1)[:, 2]
            height_abnormal = (
                (robot_height < (self.env.cfg.flight_height - 3 * self.env.cfg.safety_radius)) |
                (robot_height > (self.env.cfg.flight_height + 3 * self.env.cfg.safety_radius))
            )
            
            # 4. (来自 nav_env.py/_get_dones)
            hasnan = torch.isnan(state.ego_drone.drone_state).any(dim=(1, 2))
            
            # 5. 合并
            terminated = reached_target_mask | collision_mask | height_abnormal | hasnan
            self.terminated[:] = terminated
            
            # 6. 【重要】触发您的 MetricsManager
            #    (check_time_out 会自动被父类调用)
            time_outs = self.check_time_out()
            self.env.metrics_manager.on_done(terminated, time_outs)
            
            return self.terminated

        def reset(self, env_ids: torch.Tensor):
            """
            在重置时，触发 metrics_manager.on_reset
            """
            self.env.metrics_manager.on_reset(env_ids)
            super().reset(env_ids)
    ```

-----

### 6\. 总结：MBE 的工作流程（迁移后）

当您的训练脚本调用 `env.step(action)` 时，`ManagerBasedRLEnv.step` 内部的执行顺序将是：

1.  `UAMActionManager.process_action(action)` 被调用。

      * `traffic_sim._pre_physics_step()` 运行。
      * 您的（离散）动作被转换为 `command_vel_xy` 并存入 `env.state`。
      * ORCA 逻辑（如果启用）会修正 `env.state` 中的 `command_vel_xy`。

2.  **进入物理循环 (`for _ in range(decimation)`)**：

      * `UAMActionManager.apply_action()` 被调用。
          * `traffic_sim._apply_actions()` 运行。
          * `omnidrones` 的 `controller.compute()` 和 `drone.apply_action()` 运行。
      * `self.sim.step()` 运行。
      * `self.scene.update()` 运行（更新标准资产）。

3.  **物理循环结束**。

4.  `UAMTerminationManager.compute()` 被调用。

      * 调用 `env._detect_collisions()`（检查 Lidar, Grid, Traffic）。
      * 检查到达、高度、NaNs。
      * `terminated` 张量被计算出来。
      * **`metrics_manager.on_done()` 被触发**。

5.  `UAMRewardManager.compute()` 被调用。

      * 调用 `self.processor.compute_reward(env.state)`（此时 `env.state.collision.collision_mask` 已经被 `TerminationManager` 更新了）。
      * `reward_buf` 被计算出来。

6.  `reset_env_ids` 被确定。

7.  `env._reset_idx(reset_env_ids)` 被调用。

      * `UAMCommandManager.reset(reset_env_ids)` 被调用。
          * 调用 `env._generate_task_and_reset_drone(reset_env_ids)`。
          * 新的 `start`/`goal`/`waypoints` 被生成并存入 `env.state`。
          * 无人机被重置到 `start` 位置。
      * `UAMRewardManager.reset(reset_env_ids)` 被调用。
          * `self.processor.reset_potential(env.state, reset_env_ids)` 被调用。
      * `UAMTerminationManager.reset(reset_env_ids)` 被调用。
          * **`metrics_manager.on_reset()` 被触发**。
      * `EventManager.apply(mode="reset")` 被调用。
          * `traffic_sim.reset()` 被调用。

8.  `UAMEventManager.apply(mode="interval")` 被调用。

      * **`env._update_custom_systems(dt)` 被调用**。
          * `drone.get_state()` -\> 更新 `env.state`。
          * `_lidar.update()` -\> 更新 `env.state`。
          * `traffic_sim._post_physics_step()` -\> 更新 `env.state`。
          * 所有导航距离、`reached_target_mask` 被更新。

9.  `UAMObservationManager.compute()` 被调用。

      * 调用 `self.processor.process_observation(env.state)`。
      * 此时 `env.state` 包含了**所有**最新信息（来自无人机、Lidar、Traffic、新目标点）。

10. `env.step()` 返回最终的 `obs_buf`, `reward_buf`, `terminated`, `truncated`, `extras`。

这个架构完美地将您的所有自定义逻辑集成到了 MBE 框架中，并实现了您通过 CFG 切换所有行为的最终目标。