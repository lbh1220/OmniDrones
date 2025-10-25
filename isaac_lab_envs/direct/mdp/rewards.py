import torch
from isaac_lab_envs.direct.mdp.state import EnvState
from abc import ABC, abstractmethod
from omni.isaac.lab.utils import configclass




class RewardModule(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda"

    def compute_reward(self, state: EnvState) -> torch.Tensor:
        reward = torch.zeros(state.num_envs, device=self.device)
        return reward
    
    def reset_potential(self, state: EnvState, env_ids: torch.Tensor):
        pass

class NavRewardModule(RewardModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.collision_penalty = cfg.rew_collision
        self.success_reward = cfg.rew_success
        self.pot_factor = cfg.rew_potential
        self.previous_potential = None

    def compute_reward(self, state: EnvState) -> torch.Tensor:

        num_envs = state.num_envs

        reached_target_mask = state.navigation.reached_target_mask
        collision_mask = state.collision.collision_mask

        reward = torch.zeros(num_envs, device=self.device)
        
        # 1. 碰撞惩罚
        reward = torch.where(collision_mask, 
                           torch.full_like(reward, self.collision_penalty), 
                           reward)
        
        # 2. 成功奖励
        reward = torch.where(reached_target_mask,
                           torch.full_like(reward, self.success_reward),
                           reward)

        potential_reward = self._compute_potential_reward(state)
        reward += potential_reward
        return reward
    
    def reset_potential(self, state: EnvState, env_ids: torch.Tensor):
        """重置势能缓存"""
        # 计算当前势能（负距离）
        if env_ids is None or env_ids.shape[0] == 0:
            return
            
        
        # 计算2D距离（只考虑x,y）
        current_distance = state.navigation.current_dist_to_target[env_ids] # [N]
        current_potential = -current_distance  # [N]
        if self.previous_potential is None:
            # 第一次调用，初始化previous_potential
            self.previous_potential = current_potential.clone()
        else:
            self.previous_potential[env_ids] = current_potential.clone()

    def _compute_potential_reward(self, state: EnvState) -> torch.Tensor:
        """计算势能奖励（基于距离变化）
        
        Args:
            state: 环境状态对象
            
        Returns:
            potential_reward: [N] 势能奖励
        """
        # 计算当前势能（负距离）

        
        # 计算2D距离（只考虑x,y）

        current_distance = state.navigation.current_dist_to_target # [N]
        if self.cfg.use_global_path:
            current_distance = state.navigation.current_dist_along_path # [N]
        current_potential = -current_distance  # [N]
        
        if self.previous_potential is None:
            # 第一次调用，初始化previous_potential
            self.previous_potential = current_potential.clone()
            return torch.zeros_like(current_potential)
        
        # 计算势能变化
        potential_reward = self.pot_factor * (current_potential - self.previous_potential)
        
        # 更新previous_potential
        self.previous_potential = current_potential.clone()
        
        return potential_reward

class CrossTrackRewardModule(RewardModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cross_track_reward_coeff = cfg.rew_cross_track_coeff
        self.alpha = getattr(cfg, 'rew_cross_track_alpha', 1.0)
        self.previous_cross_track = None

    def compute_reward(self, state: EnvState) -> torch.Tensor:
        reward = torch.zeros(state.num_envs, device=self.device)
        # 添加横向误差奖励项
        if self.cross_track_reward_coeff != 0.0 and state.navigation.cross_track_errors is not None:
            cross_track_reward = self._compute_cross_track_reward(state)
            reward += cross_track_reward
        return reward
    def _compute_cross_track_reward(self, state: EnvState) -> torch.Tensor:
        """计算横向误差奖励
        
        Args:
            state: 环境状态对象
            
        Returns:
            cross_track_reward: [num_envs] 横向误差奖励
        """
        cross_track_errors = state.navigation.cross_track_errors  # [num_envs]
        safety_radius = state.collision.safety_radius
        
        # 减去安全半径，如果小于安全半径则影响不大
        effective_errors = torch.clamp(cross_track_errors - safety_radius, min=0.0)
        
        if self.cross_track_reward_coeff > 0:
            # 正系数：奖励模式 - 距离越小奖励越大
            # this value is in range [0, 1]
            cross_track_reward = self.cross_track_reward_coeff * torch.exp(-self.alpha * effective_errors)
        else:
            # 负系数：惩罚模式 - 距离越大惩罚越大
            # clamp this value to [0, 1]
            cross_track_reward = self.cross_track_reward_coeff * torch.clamp(effective_errors**2, max=1.0)

            
        return cross_track_reward

class TrafficFutureRewardModule(RewardModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.future_evtol_penalty = cfg.rew_evtol_future_penalty  # EVTOL未来碰撞惩罚
        self.future_drone_penalty = cfg.rew_drone_future_penalty  # Drone未来碰撞惩罚
        self.previous_future_penalty = None
        # future reward param
        self.drones_threshold_factor = cfg.rew_drones_threshold_factor
        self.drones_decay_factor = cfg.rew_drones_decay_factor
        self.evtols_threshold_factor = cfg.rew_evtols_threshold_factor
        self.evtols_decay_factor = cfg.rew_evtols_decay_factor
        
    def compute_reward(self, state: EnvState) -> torch.Tensor:


        reward = torch.zeros(state.num_envs, device=self.device)

        future_penalty = self._compute_future_collision_penalty_refactored(state)
        reward += future_penalty
        return reward

    def _compute_future_collision_penalty_refactored(
        self, 
        state: EnvState
    ) -> torch.Tensor:
        """
        计算未来碰撞风险惩罚（向量化版本）。
        逻辑对标 AirSim 版本的 "取最大风险" 模式。

        Args:
            drone_state: [num_envs, 1, 13] drone状态
            obs_processor: 观测处理器，包含预计算的轨迹和属性

        Returns:
            future_penalty: [num_envs] 未来碰撞惩罚
        """
        # 从状态对象提取数据
        drone_state = state.ego_drone.drone_state
        target_pos = state.navigation.target_positions
        collision_mask = state.collision.collision_mask
        reached_target_mask = state.navigation.reached_target_mask
        
        # 从交通命名空间提取数据
        traffic_positions = state.traffic.traffic_positions if state.traffic else None
        traffic_velocities = state.traffic.traffic_velocities if state.traffic else None
        traffic_types = state.traffic.traffic_types if state.traffic else None
        traffic_safety_radius = state.traffic.traffic_safety_radius if state.traffic else None
        traffic_future_traj = state.traffic.traffic_future_traj if state.traffic else None
        if (traffic_future_traj is None or 
            traffic_future_traj.numel() == 0):
            return torch.zeros(drone_state.shape[0], device=self.device)

        # --- 准备数据 ---
        num_envs = drone_state.shape[0]
        robot_pos = drone_state[:, :, :2]  # 形状: [num_envs, 1, 2]
        robot_radius = self.cfg.safety_radius

        # 从 obs_processor 获取交通数据
        traffic_traj = traffic_future_traj[:, :, :2] # 形状: [total_traffic, predict_steps+1, 2]
        traffic_types = traffic_types              # 形状: [total_traffic]
        traffic_radii = traffic_safety_radius    # 形状: [total_traffic]
        
        total_traffic, num_steps, _ = traffic_traj.shape

        # --- 1. 计算所有无人机到所有交通轨迹点的距离 ---
        # 广播: [num_envs, 1, 1, 2] - [1, total_traffic, num_steps, 2]
        # 结果 relative_pos 形状: [num_envs, total_traffic, num_steps, 2]
        relative_pos = robot_pos.unsqueeze(2) - traffic_traj.unsqueeze(0)
        relative_dist = torch.norm(relative_pos, dim=-1) # 形状: [num_envs, total_traffic, num_steps]

        # --- 2. 根据交通类型，构建参数张量 ---
        # 创建一个 [total_traffic] 的张量，其中 evtol 为 True, drone 为 False
        is_evtol_mask = (traffic_types == 2)

        # 使用 torch.where 根据类型选择不同的参数值
        penalty_factors = torch.where(is_evtol_mask, self.future_evtol_penalty, self.future_drone_penalty) # [total_traffic]
        # threshold_factors = torch.where(is_evtol_mask, 1.5, 2.0) # [total_traffic]
        threshold_factors = torch.where(is_evtol_mask, self.evtols_threshold_factor, self.drones_threshold_factor) # [total_traffic]
        decay_factors = torch.where(is_evtol_mask, self.evtols_decay_factor, self.drones_decay_factor) # [total_traffic]
        
        # --- 3. 计算随时间衰减的碰撞阈值 ---
        # 创建时间步指数 [0, 1, 2, ..., predict_steps]
        time_exponents = torch.arange(num_steps, device=self.device).view(1, 1, -1) # [1, 1, num_steps]
        
        # 广播计算每个交通飞机在每个未来时间点的衰减因子
        # decay_factors 形状 [total_traffic] -> [1, total_traffic, 1]
        time_decay = decay_factors.view(1, -1, 1) ** time_exponents # 形状: [1, total_traffic, num_steps]

        # 计算每个交通飞机的总半径
        total_radii = robot_radius + traffic_radii # 形状: [total_traffic]
        
        # 广播计算每个交通飞机在每个未来时间点的碰撞阈值
        # threshold_factors 形状 [total_traffic] -> [1, total_traffic, 1]
        # total_radii 形状 [total_traffic] -> [1, total_traffic, 1]
        collision_thresholds = (total_radii * threshold_factors).view(1, -1, 1) * time_decay
        # 确保阈值不小于基础安全距离
        collision_thresholds = torch.max(collision_thresholds, total_radii.view(1, -1, 1)) # [1, total_traffic, num_steps]

        # --- 4. 计算惩罚并找到最小惩罚（最大风险） ---
        # 计算距离与阈值的比率（旧版中的 tooclose_dist）
        # 形状: [num_envs, total_traffic, num_steps]
        tooclose_dist = (relative_dist - collision_thresholds) / (total_radii.view(1, -1, 1))
        
        # 计算所有可能的惩罚值
        # penalty_factors 形状 [total_traffic] -> [1, total_traffic, 1]
        penalties = -penalty_factors.view(1, -1, 1) * time_decay
        reward_future_matrix = tooclose_dist * penalties

        # 关键一步：只考虑发生碰撞风险的情况 (tooclose_dist < 0)
        # 将没有风险的地方设置为0（一个无害的大值），以便 min 操作能正确找到最小负值
        reward_future_matrix[tooclose_dist >= 0] = 0.0

        # 在所有交通飞机和所有未来时间步中，找到那个最小的惩罚值（即最大的风险）
        # min over dim=2 (time), then min over dim=1 (traffic)
        future_penalty, _ = torch.min(reward_future_matrix, dim=2)
        future_penalty, _ = torch.min(future_penalty, dim=1) # 形状: [num_envs]

        # 确保惩罚不会是正的
        return torch.min(future_penalty, torch.zeros_like(future_penalty))

class TTCRewardModule(RewardModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # TTC-based reward parameters (reasonable defaults; can be overridden by cfg)
        # R_risk = -alpha * exp(-TTC / beta), for TTC < threshold
        self.ttc_threshold = getattr(cfg, 'rew_ttc_threshold', 10.0)
        self.ttc_alpha = getattr(cfg, 'rew_ttc_alpha', 0.0)
        self.ttc_beta = getattr(cfg, 'rew_ttc_beta', 5.0)
        # Contextual potential: (1 - risk_factor) * R_potential - risk_factor * delta
        self.ttc_idle_penalty = getattr(cfg, 'rew_ttc_idle_penalty', 0.01)
        # Optional patience reward coefficient (omega). 0 disables this term.
        self.patience_coeff = getattr(cfg, 'rew_patience_coeff', 0.0)
        # Memory for patience reward (per-env). Initialized lazily on first use.
        self.previous_min_d_cpa = None

    def compute_reward(self, state: EnvState) -> torch.Tensor:
        reward = torch.zeros(state.num_envs, device=self.device)
        use_ttc = abs(self.ttc_alpha) > 1e-4
        drone_state = state.ego_drone.drone_state

        if use_ttc:
            # 7. 计算TTC/CPA指标（基于当前 traffic 的位置和速度）
            robot_vel_2d = state.ego_drone.velocities[:, :, :2]
            # Filter eVTOLs on the fly using traffic_types == 2
            use_positions = None
            if (state.traffic is not None and
                state.traffic.traffic_positions is not None and state.traffic.traffic_positions.numel() > 0 and
                state.traffic.traffic_velocities is not None and state.traffic.traffic_velocities.numel() > 0 and
                state.traffic.traffic_safety_radius is not None and state.traffic.traffic_safety_radius.numel() > 0 and
                state.traffic.traffic_types is not None and state.traffic.traffic_types.numel() > 0):
                evtol_mask = (state.traffic.traffic_types == 2)
                if evtol_mask.any():
                    use_positions = state.traffic.traffic_positions[evtol_mask]
                    use_velocities = state.traffic.traffic_velocities[evtol_mask]
                    use_safety_radius = state.traffic.traffic_safety_radius[evtol_mask]

            if use_positions is not None:
                min_ttc, min_d_cpa = self._compute_ttc_metrics(
                    drone_state,
                    robot_vel_2d,
                    use_positions,
                    use_velocities,
                    use_safety_radius
                )
            else:
                # No traffic: TTC=+inf leads to no risk; CPA uses current distance surrogate
                num_envs = drone_state.shape[0]
                min_ttc = torch.full((num_envs,), float('inf'), device=self.device)
                # use zeros for d_cpa so patience reward contributes 0
                min_d_cpa = torch.zeros((num_envs,), device=self.device)

            # 8. TTC风险惩罚 R_risk
            ttc_risk_penalty = self._compute_ttc_risk_penalty(min_ttc)
            reward += ttc_risk_penalty
            # 10. 可选耐心奖励：仅在存在碰撞风险时鼓励增大与威胁的最近距离
            if self.patience_coeff != 0.0:
                patience_reward = self._compute_patience_reward(min_d_cpa, min_ttc)
                reward += patience_reward

        return reward

    def _compute_ttc_metrics(
        self,
        drone_state: torch.Tensor,
        robot_vel_2d: torch.Tensor,
        traffic_positions: torch.Tensor,
        traffic_velocities: torch.Tensor,
        traffic_safety_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vectorized TTC and CPA metrics against all traffic.

        Returns:
            min_ttc: [num_envs] minimum TTC per env (inf if no collision predicted)
            min_d_cpa: [num_envs] minimum CPA distance per env
        """
        EPS = 1e-6

        # Ego state (2D)
        robot_pos = drone_state[:, :, :2]  # [N, 1, 2]
        robot_vel = robot_vel_2d  # [N, 1, 2]

        # Traffic state (2D)
        traffic_pos_2d = traffic_positions[:, :2]  # [T, 2]
        traffic_vel_2d = traffic_velocities[:, :2]  # [T, 2]

        # Relative position/velocity: broadcast to [N, T, 2]
        p_rel = traffic_pos_2d.unsqueeze(0) - robot_pos  # [N, T, 2]
        v_rel = traffic_vel_2d.unsqueeze(0) - robot_vel  # [N, T, 2]

        v_rel_sq = torch.sum(v_rel * v_rel, dim=-1)  # [N, T]
        p_dot_v = torch.sum(p_rel * v_rel, dim=-1)  # [N, T]

        # Time to CPA
        t_cpa = -p_dot_v / (v_rel_sq + EPS)  # [N, T]
        p_cpa = p_rel + v_rel * t_cpa.unsqueeze(-1)  # [N, T, 2]
        d_cpa_sq = torch.sum(p_cpa * p_cpa, dim=-1)  # [N, T]
        current_dist_sq = torch.sum(p_rel * p_rel, dim=-1)  # [N, T]
        is_future = t_cpa > 0
        d_cpa_sq = torch.where(is_future, d_cpa_sq, current_dist_sq)
        d_cpa = torch.sqrt(d_cpa_sq + EPS)  # [N, T]

        # Time-to-collision via quadratic solution
        robot_radius = self.cfg.safety_radius
        total_radii = robot_radius + traffic_safety_radius  # [T]
        a = v_rel_sq  # [N, T]
        b = 2.0 * p_dot_v  # [N, T]
        c = current_dist_sq - (total_radii.view(1, -1)) ** 2  # [N, T]
        delta = b * b - 4.0 * a * c  # [N, T]

        ttc = torch.full_like(delta, float('inf'))  # [N, T]
        mask = (delta >= 0.0) & (b < 0.0)
        if mask.any():
            sqrt_delta = torch.sqrt(torch.clamp(delta[mask], min=0.0))
            t1 = (-b[mask] - sqrt_delta) / (2.0 * a[mask] + EPS)
            # Small positive times only
            t1 = torch.where(t1 > 0.0, t1, torch.full_like(t1, float('inf')))
            ttc[mask] = t1

        # Reduce over traffic dimension
        min_ttc, _ = torch.min(ttc, dim=1)  # [N]
        min_d_cpa, _ = torch.min(d_cpa, dim=1)  # [N]

        return min_ttc, min_d_cpa

    

    def _compute_ttc_risk_penalty(self, min_ttc: torch.Tensor) -> torch.Tensor:
        """Compute R_risk based on min TTC per env.

        R_risk = -alpha * exp(-TTC / beta) when TTC < threshold, else 0.
        """
        if self.ttc_alpha == 0.0:
            return torch.zeros_like(min_ttc)

        risk_zone = min_ttc < self.ttc_threshold
        # 这里的min_ttc中可能有inf, 但是inf是不会小于self.ttc_threshold的, 不会参与计算
        penalty = torch.zeros_like(min_ttc)
        if risk_zone.any():
            penalty[risk_zone] = -self.ttc_alpha * torch.exp(-min_ttc[risk_zone] / (self.ttc_beta + 1e-6))
        return penalty

    def _compute_contextual_potential_from_base(
        self, base_potential_reward: torch.Tensor, min_ttc: torch.Tensor
    ) -> torch.Tensor:
        """Compute contextual potential reward from base potential and TTC.

        risk_factor = clamp(1 - TTC / threshold, 0, 1)
        R_potential_contextual = (1 - risk_factor) * base_potential - risk_factor * delta
        """
        # Broadcast to match base_potential shape [N]
        risk_factor = torch.clamp(1.0 - (min_ttc / (self.ttc_threshold + 1e-6)), 0.0, 1.0)
        contextual = (1.0 - risk_factor) * base_potential_reward - risk_factor * self.ttc_idle_penalty
        return contextual

    def _compute_patience_reward(self, min_d_cpa: torch.Tensor, min_ttc: torch.Tensor) -> torch.Tensor:
        """Optional patience reward encouraging increasing minimum CPA distance.

        R_patience = omega * (d_cpa_min^t - d_cpa_min^{t-1})
        Stores per-env previous values and updates them.
        """
        if self.patience_coeff == 0.0:
            return torch.zeros_like(min_d_cpa)

        num_envs = min_d_cpa.shape[0]
        if self.previous_min_d_cpa is None or self.previous_min_d_cpa.shape[0] != num_envs:
            # Initialize memory on first use or if env count changed
            self.previous_min_d_cpa = min_d_cpa.clone()
            return torch.zeros_like(min_d_cpa)

        # Only apply when there is risk: min_ttc < threshold
        risk_mask = min_ttc < (self.ttc_threshold)

        # Compute delta only for risky envs; otherwise set delta=0 and refresh memory to current to avoid drift
        patience = torch.zeros_like(min_d_cpa)
        if risk_mask.any():
            # Guard against NaN sentinel: if previous is NaN, skip reward this step
            prev_vals = self.previous_min_d_cpa[risk_mask]
            curr_vals = min_d_cpa[risk_mask]
            valid_prev_mask = ~torch.isnan(prev_vals)
            if valid_prev_mask.any():
                delta = curr_vals[valid_prev_mask] - prev_vals[valid_prev_mask]
                patience[risk_mask.nonzero(as_tuple=False).squeeze(1)[valid_prev_mask]] = self.patience_coeff * delta

        # Refresh memory for all envs to current min_d_cpa so that when risk appears later,
        # delta is computed from the latest baseline, avoiding accumulation across safe periods.
        self.previous_min_d_cpa = min_d_cpa.clone()
        return patience
    def reset_potential(self, state: EnvState, env_ids: torch.Tensor):
        # Reset patience memory for the specified envs
        if self.previous_min_d_cpa is not None:
            if self.previous_min_d_cpa.shape[0] < state.ego_drone.drone_state.shape[0]:
                # Expand to full size if needed
                full = torch.zeros(state.ego_drone.drone_state.shape[0], device=self.device)
                full[: self.previous_min_d_cpa.shape[0]] = self.previous_min_d_cpa
                self.previous_min_d_cpa = full
            # Use NaN sentinel to indicate re-initialization is needed on next step,
            # avoiding misleading delta due to using 0 (which implies collision distance).
            self.previous_min_d_cpa[env_ids] = torch.nan


class NavrlRewardModule(RewardModule):
    """
    复现NavRL项目
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev_vel_2d: torch.Tensor | None = None
        self.lidar_range = getattr(cfg, 'lidar_range', 10.0)
        self.dynamic_obstacle_num = getattr(cfg, 'dynamic_obstacle_num', 5)

    def compute_reward(self, state: EnvState) -> torch.Tensor:
        num_envs = state.num_envs
        reward = torch.zeros(num_envs, device=self.device)
        # 1) 生存奖励 +1
        reward = reward + 1.0
        # 2) 速度沿目标方向分量
        reward = reward + self._reward_vel(state)
        # 3) 静态障碍安全 (LiDAR)
        reward = reward + self._reward_safety_static(state)
        # 4) 动态障碍安全 (最近K个)
        reward = reward + self._reward_safety_dynamic(state)
        # 5) 平滑性惩罚（系数 -0.1 已在函数内部体现）
        reward = reward + self._penalty_smoothness(state)
        return reward
    def _reward_vel(self, state: EnvState) -> torch.Tensor:
        # 计算速度在target上的点积分量，作为奖励
        # 类似于reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)
        drone_state = state.ego_drone.drone_state  # [N,1,13]
        robot_vel_2d = state.ego_drone.velocities[:, :, :2]  # [N,1,2]
        robot_pos_2d = drone_state[:, :, :2]  # [N,1,2]
        target_pos_2d = state.navigation.target_positions[:, :, :2]  # [N,1,2]
        if self.cfg.use_global_path:
            target_pos_2d = state.navigation.local_goals[:, :, :2]
        to_goal = target_pos_2d - robot_pos_2d  # [N,1,2]
        to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)  # [N,1,1]
        # 单位方向
        goal_dir = torch.where(to_goal_norm > 1e-6, to_goal / to_goal_norm, torch.zeros_like(to_goal))  # [N,1,2]
        # 点积（2D速度在目标方向上的投影）
        proj_speed = (robot_vel_2d * goal_dir).sum(dim=-1).squeeze(1)  # [N]
        return proj_speed
    def _reward_safety_static(self, state: EnvState) -> torch.Tensor:

        lidar_scan = getattr(state.perception, 'lidar_scan', None)
        if lidar_scan is None:
            return torch.zeros(state.num_envs, device=self.device)
        # 预期形状 [N, 1, H, W]，数值范围 [0, lidar_range] 且已取反（0安全，range危险）
        # 扣除机器人安全半径的影响（简单减法后截断到[0, lidar_range]）
        safety_radius = getattr(self.cfg, 'safety_radius', 0.0)
        effective = (self.lidar_range - lidar_scan - safety_radius).clamp(min=1e-6, max=self.lidar_range)
        # 对 H, W 维度做均值
        rew = torch.log(effective).mean(dim=(2, 3))  # [N, 1]
        return rew.squeeze(1)
    def _reward_safety_dynamic(self, state: EnvState) -> torch.Tensor:
        # reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)
        # 取最近的N个动态障碍物的最近距离，取log计算
        # 不过这个最近距离需要减去robot和traffic的safety radius
        if (state.traffic is None or
            state.traffic.traffic_positions is None or state.traffic.traffic_positions.numel() == 0 or
            state.traffic.traffic_safety_radius is None or state.traffic.traffic_safety_radius.numel() == 0):
            return torch.zeros(state.num_envs, device=self.device)
        drone_state = state.ego_drone.drone_state
        robot_pos_2d = drone_state[:, :, :2]  # [N,1,2]
        robot_radius = getattr(self.cfg, 'safety_radius', 0.0)
        traffic_pos_2d = state.traffic.traffic_positions[:, :2]  # [T,2]
        traffic_radii = state.traffic.traffic_safety_radius      # [T]
        # 距离矩阵 [N,T]
        rel = traffic_pos_2d.unsqueeze(0) - robot_pos_2d  # [N,1,2] broadcasting -> [N,T,2]
        dists = torch.norm(rel, dim=-1)  # [N,T]
        # 净距离（减安全半径）
        net = dists - (robot_radius + traffic_radii.view(1, -1))  # [N,T]
        # 选最近的K个
        K = int(self.dynamic_obstacle_num)
        if net.shape[1] >= K:
            idx = torch.topk(net, k=K, largest=False, dim=1).indices  # [N,K]
            nearest = torch.gather(net, 1, idx)  # [N,K]
        else:
            nearest = net
            # 不足K时右侧补零
            pad = torch.zeros((state.num_envs, K - net.shape[1]), device=self.device)
            nearest = torch.cat([nearest, pad], dim=1)
        # 映射到正域并取log，再对K取均值
        closest_reward = torch.log(nearest.clamp(min=1e-6, max=self.lidar_range))  # [N,K]
        return closest_reward.mean(dim=-1)  # [N]
    def _penalty_smoothness(self, state: EnvState) -> torch.Tensor:
        # 计算前后两帧之间的2d速度差的norm, 系数是-0.1
        vel_2d = state.ego_drone.velocities[:, :, :2].squeeze(1)  # [N,2]
        if self.prev_vel_2d is None or self.prev_vel_2d.shape[0] != vel_2d.shape[0]:
            self.prev_vel_2d = vel_2d.clone()
            return torch.zeros(vel_2d.shape[0], device=self.device)
        delta = torch.norm(vel_2d - self.prev_vel_2d, dim=-1)  # [N]
        self.prev_vel_2d = vel_2d.clone()
        return -0.1 * delta

    def reset_potential(self, state: EnvState, env_ids: torch.Tensor):
        if self.prev_vel_2d is not None:
            if self.prev_vel_2d.shape[0] < state.ego_drone.velocities.shape[0]:
                # Expand to full size if needed
                full = torch.zeros(state.ego_drone.velocities.shape[0], device=self.device)
                full[: self.prev_vel_2d.shape[0]] = self.prev_vel_2d
                self.prev_vel_2d = full
            self.prev_vel_2d[env_ids] = state.ego_drone.velocities[env_ids, :, :2].clone()
        else:
            self.prev_vel_2d = state.ego_drone.velocities[:, :, :2].clone()
# -------------------- Modular Reward Manager --------------------
@configclass
class RewardManagerCfg:
    # names registered in REWARD_MODULES
    modules: list[str] = ("nav", "cross_track")


REWARD_MODULES: dict[str, type[RewardModule]] = {
    "nav": NavRewardModule,
    "cross_track": CrossTrackRewardModule,
    "traffic_future": TrafficFutureRewardModule,
    "ttc": TTCRewardModule,
    "navrl": NavrlRewardModule,
}


class RewardManager:
    def __init__(self, env_cfg, manager_cfg: RewardManagerCfg | None = None, device: str = "cuda"):
        self.env_cfg = env_cfg
        self.device = device
        self.manager_cfg = manager_cfg or RewardManagerCfg()
        self.modules: list[RewardModule] = []
        # build modules by names
        seen = set()
        for name in self.manager_cfg.modules:
            if name in seen:
                continue
            seen.add(name)
            mod_cls = REWARD_MODULES.get(name)
            if mod_cls is None:
                continue

            mod = mod_cls(env_cfg)
            mod.device = device
            self.modules.append(mod)

    def compute_reward(self, state: EnvState) -> torch.Tensor:
        if not self.modules:
            return torch.zeros(state.num_envs, device=self.device)
        total = torch.zeros(state.num_envs, device=self.device)
        for mod in self.modules:
            r = mod.compute_reward(state)
            if r is not None:
                total = total + r
        return total

    def reset_potential(self, state: EnvState, env_ids: torch.Tensor):
        for mod in self.modules:
            if hasattr(mod, "reset_potential"):
                mod.reset_potential(state, env_ids)
