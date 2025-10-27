from __future__ import annotations

from collections import deque
from typing import Dict, List

import torch


class MetricModule:
    """Base metric module with a single on_done entry.

    Contract:
    - Modules maintain their own internal state (deques/tensors) as needed.
    - The only hook called by manager is on_done(terminated, truncated).
    - on_done returns a dict[str, float] of aggregated scalar metrics (rolling means).
    """

    def __init__(self):
        self.num_envs: int = 0
        self.device: torch.device | None = None

    def on_env_init(self, manager: "MetricsManager"):
        self.num_envs = manager.num_envs
        self.device = manager.device
        self.mean_values_per_env = torch.zeros(manager.num_envs, device=manager.device)

    def on_done(self, manager: "MetricsManager", terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
        return {}
    def on_reset(self, manager: "MetricsManager", env_ids: torch.Tensor):
        self.mean_values_per_env[env_ids] = 0.0

    @staticmethod
    def mean(q: deque) -> float:
        if len(q) == 0:
            return 0.0
        return float(sum(q) / len(q))
    def mean_to_tensor(self, q: deque) -> torch.Tensor:
        mean_value = self.mean(q)
        # return a tensor of shape [num_envs], fill with mean_value
        return torch.full((self.num_envs,), mean_value, device=self.device)
class MetricsManager:
    """Central manager calling modules' on_done and publishing their scalars.

    - Modules compute and maintain values; manager only triggers and publishes.
    - Published destination: extras['metrics'] as a flat dict of scalar metrics.
    """

    def __init__(self, num_envs: int, device: torch.device, queue_size: int = 100, use_skrl: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.modules: List[MetricModule] = []
        self.use_skrl = use_skrl

        # Pointers to env/state will be set at initialization
        self.env = None

    # -------------------- registration --------------------
    def register(self, module: MetricModule):
        self.modules.append(module)
        if self.env is not None:
            module.on_env_init(self)
        return self

    def bind_env(self, env):
        self.env = env
        for m in self.modules:
            m.on_env_init(self)

    # -------------------- lifecycle hooks --------------------
    def on_done(self, terminated: torch.Tensor, truncated: torch.Tensor):
        # Trigger all modules and merge their scalar outputs
        merged: Dict[str, float] = {}
        for m in self.modules:
            out = m.on_done(self, terminated, truncated)
            if out:
                merged.update(out)
        # Publish as flat keys to satisfy SB3 wrapper (no nested dicts except 'log')
        # Keys will be written as: "metrics/<name>"
        for name, value in merged.items():
            if self.use_skrl:
                # self.env.extras需要有eposide这个key，并且这个key下必须是字典，每个subkey是一个tensor标量
                # 只记录rolling的指标, episode的记录抖动会比较严重
                if not name.startswith('rolling'):   
                    continue
                if 'episode' not in self.env.extras:
                    self.env.extras['episode'] = {} 
                # 将指标值转换为 [1] 形状的标量tensor（多元素取mean）
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        scalar = value.reshape(1)
                    else:
                        scalar = value.float().mean().reshape(1)
                else:
                    # 非tensor，转为tensor并放到manager设备
                    scalar = torch.tensor([float(value)], device=self.device)
                self.env.extras['episode'][name] = scalar
            else:
                self.env.extras[f"metrics/{name}"] = value

    def on_reset(self, env_ids: torch.Tensor):
        for m in self.modules:
            m.on_reset(self, env_ids)

class CrossTrackModule(MetricModule):
    """Cross-track error: episode mean window aggregator."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.rb_values: deque = deque(maxlen=self.window)

    def on_done(self, manager: "MetricsManager", terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
        env = manager.env
        finished = (terminated | truncated)
        # 应该先更新avg，然后取结束的值更新rb_values, 同时将结束的mean_values_per_env更新为0
        
        if (env.state.navigation.cross_track_errors is not None and 
            env.cfg.use_global_path):
            current_errors = env.state.navigation.cross_track_errors  # [num_envs]
            old_avg = self.mean_values_per_env  # [num_envs]
            current_step = env.episode_length_buf + 1  # [num_envs]
            new_avg = (old_avg * (current_step - 1) + current_errors) / current_step
            self.mean_values_per_env = new_avg
            for idx in finished.nonzero().squeeze(-1).tolist():
                self.rb_values.append(float(self.mean_values_per_env[idx].item()))

        return {"rolling/mean_recent_cross_track_error": self.mean_to_tensor(self.rb_values),
                "episode/mean_cross_track_error": self.mean_values_per_env.clone()}

class FlagsModule(MetricModule):
    """Tracks goal/collision/timeout flags per episode and rolling rates."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.rb_success: deque = deque(maxlen=self.window)
        self.rb_collision: deque = deque(maxlen=self.window)
        self.rb_timeout: deque = deque(maxlen=self.window)

    def on_done(self, manager: "MetricsManager", terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
        env = manager.env
        finished = (terminated | truncated)
        if finished.any():
            reached = env.state.navigation.reached_target_mask
            collision = getattr(env.state.collision, "collision_mask", None)
            if collision is None:
                collision = torch.zeros(manager.num_envs, dtype=torch.bool, device=manager.device)
            timeout_mask = finished & (~reached) & (~collision)
            env_ids = torch.nonzero(finished, as_tuple=False).squeeze(-1)
            for idx in env_ids.tolist():
                self.rb_success.append(float(reached[idx].item()))
                self.rb_collision.append(float(collision[idx].item()))
                self.rb_timeout.append(float(timeout_mask[idx].item()))

        return {
            "rolling/success_rate": self.mean_to_tensor(self.rb_success),
            "rolling/collision_rate": self.mean_to_tensor(self.rb_collision),
            "rolling/timeout_rate": self.mean_to_tensor(self.rb_timeout),
        }


class AccelerationModule(MetricModule):
    """Acceleration episode mean window aggregator."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.rb_values: deque = deque(maxlen=self.window)
        

    def on_done(self, manager: "MetricsManager", terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
        env = manager.env
        finished = (terminated | truncated)
        # 应该先更新avg，然后取结束的值更新rb_values, 同时将结束的mean_values_per_env更新为0
        # 计算当前加速度（速度变化的模）
        if (env.state.ego_drone.velocities is not None and 
            env.state.ego_drone.previous_velocities is not None):
            
            current_velocity = env.state.ego_drone.velocities   # [num_envs, 1, 3]
            previous_velocity = env.state.ego_drone.previous_velocities   # [num_envs, 1, 3]
            
            # 计算速度变化（加速度）
            velocity_change = current_velocity - previous_velocity  # [num_envs, 1, 3]
            
            # 计算速度变化的模（L2范数）
            current_acceleration = torch.norm(velocity_change.squeeze(1), dim=1)  # [num_envs]
            
            # 递增平均值公式: new_avg = (old_avg * (n-1) + new_value) / n
            current_step = env.episode_length_buf + 1  # [num_envs]
            old_avg = self.mean_values_per_env  # [num_envs]
            new_avg = (old_avg * (current_step - 1) + current_acceleration) / current_step
            
            self.mean_values_per_env = new_avg
            # # 更新rb_values
            for idx in finished.nonzero().squeeze(-1).tolist():
                self.rb_values.append(float(self.mean_values_per_env[idx].item()))

        return {"rolling/mean_recent_acceleration": self.mean_to_tensor(self.rb_values),
                "episode/mean_acceleration": self.mean_values_per_env.clone()}

class NearCollisionModule(MetricModule):
    """Near-collision episode mean window aggregator."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.rb_values: deque = deque(maxlen=self.window)

    def on_done(self, manager: "MetricsManager", terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
        env = manager.env
        finished = (terminated | truncated)
        # 应该先更新avg，然后取结束的值更新rb_values, 同时将结束的mean_values_per_env更新为0


        # 计算方法
        # 自车位置: [num_envs, 1, 3]
        ego_positions = env.state.ego_drone.positions  # [E, 1, 3]
        # traffic 数据: positions [T, 3], types [T], safety_radius [T]
        traffic_positions = env.state.traffic.traffic_positions  # [T, 3]
        traffic_types = env.state.traffic.traffic_types  # [T]
        traffic_safety_radius = env.state.traffic.traffic_safety_radius  # [T]

        if traffic_positions is None or traffic_positions.numel() == 0:
            return {"rolling/mean_recent_near_collision_ratio": self.mean_to_tensor(self.rb_values),
                    "episode/mean_near_collision_ratio": self.mean_values_per_env.clone()}

        # 计算 pairwise 距离: [E, T]
        # 广播: (E, 1, 3) - (1, T, 3) -> (E, T, 3)
        deltas = ego_positions.squeeze(1).unsqueeze(1) - traffic_positions.unsqueeze(0)
        distances = torch.norm(deltas, dim=-1)  # [E, T]

        # 获取每个 traffic 的 near-collision 阈值: [T]
        # thresholds = self._get_near_collision_ratio_thresholds(traffic_types, traffic_safety_radius)  # [T]
        
        thresholds = traffic_safety_radius + env.cfg.safety_radius
        thresholds = thresholds*2.0
        # 比较: [E, T]
        near_matrix = distances < thresholds.unsqueeze(0)
        near_any = near_matrix.any(dim=1)  # [E]

        # 更新均值
        old_avg = self.mean_values_per_env  # [E]
        current_step = env.episode_length_buf + 1  # [E]
        new_avg = (old_avg * (current_step - 1) + near_any.float()) / current_step
        self.mean_values_per_env = new_avg
        for idx in finished.nonzero().squeeze(-1).tolist():
            self.rb_values.append(float(self.mean_values_per_env[idx].item()))

        return {"rolling/mean_recent_near_collision_ratio": self.mean_to_tensor(self.rb_values),
                "episode/mean_near_collision_ratio": self.mean_values_per_env.clone()}

    # def _get_near_collision_ratio_thresholds(self, traffic_types: torch.Tensor, traffic_safety_radius: torch.Tensor) -> torch.Tensor:
    #     """
    #     返回每个 traffic 的 near-collision 判定阈值（形状 [T]）。
        
    #     占位实现：当前直接使用各自的安全半径作为阈值。
    #     后续可根据 traffic_types（例如 drone / eVTOL）定制不同的倍数或固定值，
    #     例如：
    #         - drone: threshold = k1 * safety_radius
    #         - eVTOL: threshold = k2 * safety_radius
    #     也可以引入绝对距离阈值或二维/三维不同的度量方式。
    #     """
    #     # TODO(liang): 按类型定制阈值逻辑。当前占位为 safety_radius 本身。

    #     thresholds = traffic_safety_radius + env.cfg.safety_radius
    #     thresholds = thresholds*2.0
    #     return thresholds


# (no env attribute injection; manager holds all accumulators)


