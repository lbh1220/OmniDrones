import torch
from isaac_lab_envs.direct.traffic_env import TrafficEnvCfg
from isaac_lab_envs.direct.mdp.observations import TrafficObservationProcessor

class TrafficRewardCalculator:
    """Traffic环境的奖励计算器，基于Isaac Lab tensor操作优化"""
    
    def __init__(self, cfg: TrafficEnvCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        
        # 奖励配置参数
        self.collision_penalty = cfg.rew_collision
        self.pot_factor = cfg.rew_potential  # 势能奖励因子
        self.time_penalty = cfg.rew_time_penalty  # 时间惩罚
        self.success_reward = cfg.rew_success
        
        self.future_evtol_penalty = cfg.rew_evtol_future_penalty  # EVTOL未来碰撞惩罚
        self.future_drone_penalty = cfg.rew_drone_future_penalty  # Drone未来碰撞惩罚


        # self.future_penalty = cfg.rew_evtol_future_penalty  # 未来碰撞惩罚
        self.discomfort_dist = 0.2  # 不适距离
        self.discomfort_penalty_factor = 0.5
        # 仿真参数
        self.predict_steps = cfg.predict_steps
        self.pred_timestep = cfg.pred_timestep
        
        # 势能缓存
        self.previous_potential = None
        
    def compute_reward(self, 
                      drone_state: torch.Tensor,
                      target_pos: torch.Tensor,
                      traffic_positions: torch.Tensor,
                      traffic_velocities: torch.Tensor,
                      collision_mask: torch.Tensor,
                      reached_target_mask: torch.Tensor,
                      traffic_future_traj: torch.Tensor,
                      traffic_safety_radius: torch.Tensor,
                      traffic_types: torch.Tensor) -> torch.Tensor:
        """计算复杂奖励
        
        Args:
            drone_state: [num_envs, 1, 13] ego drone状态
            target_pos: [num_envs, 1, 3] 目标位置
            traffic_positions: [total_traffic, 3] traffic位置
            traffic_velocities: [total_traffic, 3] traffic速度
            traffic_types: traffic类型列表
            collision_mask: [num_envs] 碰撞mask
            reached_target_mask: [num_envs] 到达目标mask
            obs_processor: 观测处理器（用于获取预测轨迹）
            
        Returns:
            reward: [num_envs] 奖励张量
        """
        num_envs = drone_state.shape[0]
        reward = torch.zeros(num_envs, device=self.device)
        
        # 1. 碰撞惩罚
        reward = torch.where(collision_mask, 
                           torch.full_like(reward, self.collision_penalty), 
                           reward)
        
        # 2. 成功奖励
        reward = torch.where(reached_target_mask,
                           torch.full_like(reward, self.success_reward),
                           reward)
        
        # 对于既没有碰撞也没有到达目标的环境，计算其他奖励
        # continue_mask = ~(collision_mask | reached_target_mask)

        potential_reward = self._compute_potential_reward(
            drone_state, 
            target_pos
        )
        reward += potential_reward

        # 3. 不适距离惩罚（与traffic的距离过近）
        # discomfort_penalty = self._compute_discomfort_penalty(
        #     drone_state[continue_mask],
        #     traffic_positions,
        #     traffic_velocities
        # )
        # reward[continue_mask] += discomfort_penalty
            
            # 5. 未来碰撞风险惩罚
        future_penalty = self._compute_future_collision_penalty_refactored(
            drone_state,
            traffic_future_traj,
            traffic_safety_radius,
            traffic_types
        )
        reward += future_penalty
        
        # 6. 时间惩罚（所有环境都有）
        dt = self.cfg.sim.dt * self.cfg.decimation
        reward += self.time_penalty * dt
        
        return reward
    
    def _compute_potential_reward(self, drone_state: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """计算势能奖励（基于距离变化）
        
        Args:
            drone_state: [N, 1, 13] drone状态
            target_pos: [N, 1, 3] 目标位置
            
        Returns:
            potential_reward: [N] 势能奖励
        """
        # 计算当前势能（负距离）
        robot_pos = drone_state[:, :, :3]  # [N, 1, 3]
        goal_pos = target_pos  # [N, 1, 3]
        
        # 计算2D距离（只考虑x,y）
        current_distance = torch.norm(goal_pos[:, :, :2] - robot_pos[:, :, :2], dim=-1)  # [N, 1]
        current_potential = -current_distance.squeeze(-1)  # [N]
        
        if self.previous_potential is None:
            # 第一次调用，初始化previous_potential
            self.previous_potential = current_potential.clone()
            return torch.zeros_like(current_potential)
        
        # 计算势能变化
        potential_reward = self.pot_factor * (current_potential - self.previous_potential)
        
        # 更新previous_potential
        self.previous_potential = current_potential.clone()
        
        return potential_reward
    
    def _compute_discomfort_penalty(self, drone_state: torch.Tensor, 
                                  traffic_positions: torch.Tensor,
                                  traffic_velocities: torch.Tensor) -> torch.Tensor:
        """计算不适距离惩罚
        
        Args:
            drone_state: [N, 1, 13] drone状态
            traffic_positions: [total_traffic, 3] traffic位置
            traffic_velocities: [total_traffic, 3] traffic速度
            
        Returns:
            discomfort_penalty: [N] 不适惩罚
        """
        if traffic_positions.numel() == 0:
            return torch.zeros(drone_state.shape[0], device=self.device)
        
        robot_pos = drone_state[:, :, :2]  # [N, 1, 2]
        traffic_pos_2d = traffic_positions[:, :2]  # [total_traffic, 2]
        
        # 计算所有环境与所有traffic的距离
        # robot_pos: [N, 1, 2] -> [N, 1, 1, 2]
        # traffic_pos_2d: [total_traffic, 2] -> [1, total_traffic, 2]
        distances = torch.norm(
            robot_pos.unsqueeze(2) - traffic_pos_2d.unsqueeze(0), 
            dim=-1
        )  # [N, 1, total_traffic]
        
        # 计算最小距离
        min_distances = torch.min(distances, dim=-1)[0].squeeze(-1)  # [N]
        
        # 计算不适惩罚
        robot_radius = self.cfg.safety_radius
        traffic_radius = 1.0  # 假设traffic的安全半径
        safe_distance = robot_radius + traffic_radius + self.discomfort_dist
        
        discomfort_mask = min_distances < safe_distance
        penalty_ratio = torch.clamp(
            (safe_distance - min_distances) / safe_distance, 
            min=0.0, max=1.0
        )
        
        discomfort_penalty = -self.discomfort_penalty_factor * penalty_ratio
        discomfort_penalty = torch.where(discomfort_mask, discomfort_penalty, torch.zeros_like(discomfort_penalty))
        
        return discomfort_penalty
    
 
    def _compute_future_collision_penalty_refactored(
        self, 
        drone_state: torch.Tensor,
        traffic_future_traj: torch.Tensor,
        traffic_safety_radius: torch.Tensor,
        traffic_types: torch.Tensor
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
        is_evtol_mask = (traffic_types == 0)

        # 使用 torch.where 根据类型选择不同的参数值
        penalty_factors = torch.where(is_evtol_mask, self.future_evtol_penalty, self.future_drone_penalty) # [total_traffic]
        threshold_factors = torch.where(is_evtol_mask, 1.5, 2.0) # [total_traffic]
        decay_factors = torch.where(is_evtol_mask, 0.9, 0.667) # [total_traffic]
        
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
    
    def reset_potential(self, drone_state: torch.Tensor, target_pos: torch.Tensor, env_ids: torch.Tensor):
        """重置势能缓存"""
        # 计算当前势能（负距离）
        if env_ids is None or env_ids.shape[0] == 0:
            return
        robot_pos = drone_state[env_ids, :, :3]  # [N, 1, 3]
        goal_pos = target_pos[env_ids, :, :3]  # [N, 1, 3]
        
        # 计算2D距离（只考虑x,y）
        current_distance = torch.norm(goal_pos[:, :, :2] - robot_pos[:, :, :2], dim=-1)  # [N, 1]
        current_potential = -current_distance.squeeze(-1)  # [N]
        if self.previous_potential is None:
            # 第一次调用，初始化previous_potential
            self.previous_potential = current_potential.clone()
        else:
            self.previous_potential[env_ids] = current_potential.clone()    