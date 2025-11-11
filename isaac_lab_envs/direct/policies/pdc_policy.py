import torch
from typing import Dict, Optional, Any
from .base_policy import ModelBasedPolicy


class PDCPolicy(ModelBasedPolicy):
    """
    Practical Distributed Control (PDC) policy for cooperative flight in 2D (XY).
    - Works with parallel environments (batch dimension as first dim).
    - Uses absolute coordinates from env.state, similar to ORCAPolicy.
    - Outputs normalized XY velocity (divide by env_cfg.max_speed).
    """

    def __init__(self, policy_cfg, env_cfg, name: str = "PDC"):
        super().__init__(policy_cfg, env_cfg, name)

        # Environment kinematics and speed preferences
        self.max_speed = getattr(env_cfg, "max_speed", 5.0)
        self.min_speed = getattr(env_cfg, "min_speed", 0.0)
        self.v_pref = getattr(env_cfg, "v_pref", 1.0)

        # PDC (Free Flight) parameters from PolicyConfig
        self.k1 = getattr(policy_cfg, "k1", 1.0)
        self.k2 = getattr(policy_cfg, "k2", 2.0)
        self.epsilon = getattr(policy_cfg, "epsilon", 1e-5)
        self.epsilon_s = getattr(policy_cfg, "epsilon_s", 1e-6)
        self.l_i = getattr(policy_cfg, "l_i", 2.0)
        self.d1_scale = getattr(policy_cfg, "d1", 2.0)
        self.d2_scale = getattr(policy_cfg, "d2", 5.0)

        # Finite difference step for dV/d||xi||
        self._delta = 1e-5

    @staticmethod
    def _sat_vec2(v: torch.Tensor, v_max: float) -> torch.Tensor:
        """
        Saturation for 2D vectors (per row).
        v: [N, 2]
        """
        if v.numel() == 0:
            return v
        norms = torch.norm(v, dim=-1, keepdim=True)
        scale = torch.clamp(v_max / (norms + 1e-8), max=1.0)
        return v * scale

    @staticmethod
    def _sigma(x: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """
        Smooth transition function σ(x, d1, d2) (piecewise cubic).
        Vectorized for tensor inputs with broadcasting.
        """
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        # Polynomial coefficients per pair
        denom = (d1 - d2) ** 3 + 1e-12
        A = -2.0 / denom
        B = 3.0 * (d1 + d2) / denom
        C = -6.0 * d1 * d2 / denom
        D = d2**2 * (3.0 * d1 - d2) / denom

        poly = A * x**3 + B * x**2 + C * x + D
        return torch.where(x <= d1, ones, torch.where(x >= d2, zeros, poly))

    def _s_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        s(x, epsilon_s) piecewise smoothing (vectorized).
        """
        eps_s = self.epsilon_s
        # constants
        x2 = 1.0 + eps_s * torch.tan(torch.tensor(67.5 * torch.pi / 180.0, device=x.device, dtype=x.dtype))
        x1 = x2 - torch.sin(torch.tensor(torch.pi / 4.0, device=x.device, dtype=x.dtype)) * eps_s
        ones = torch.ones_like(x)

        mid = (1.0 - eps_s) + torch.sqrt(torch.clamp(eps_s**2 - (x - x2) ** 2, min=0.0))
        return torch.where(x <= x1, x, torch.where(x >= x2, ones, mid))

    def _compute_V(self, xi_norm: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """
        Repulsive potential V_m_ij as a function of ||xi_m_ij||.
        Vectorized over pairs (i, j).
        """
        sigma_val = self._sigma(xi_norm, d1, d2)
        s_val = self._s_fn(torch.clamp(xi_norm / (d1 + 1e-8), min=0.0))
        denom = (1.0 + self.epsilon) * xi_norm - d1 * s_val
        # Avoid division by zero or negative
        denom = torch.clamp(denom, min=1e-8)
        return self.k2 * sigma_val / denom

    def _compute_dV_dnorm(self, xi_norm: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """
        Numerical derivative dV/d||xi|| using forward finite difference.
        """
        V_plus = self._compute_V(xi_norm + self._delta, d1, d2)
        V_now = self._compute_V(xi_norm, d1, d2)
        return (V_plus - V_now) / self._delta

    def _predict(self, state) -> torch.Tensor:
        """
        Compute raw XY velocities for each env using PDC.
        Returns: [num_envs, 2] tensor on same device as state.
        """
        device = self.device
        dtype = torch.float32

        # Ego (robot) states (2D only)
        p_i = state.ego_drone.positions.squeeze(1)[:, :2].to(device=device, dtype=dtype)  # [E, 2]
        v_i = state.ego_drone.velocities.squeeze(1)[:, :2].to(device=device, dtype=dtype)  # [E, 2]
        num_envs = p_i.shape[0]

        # Goals (absolute world frame, choose local goal if available)
        if state.navigation.local_goals is not None:
            target = state.navigation.local_goals.squeeze(1)[:, :2].to(device=device, dtype=dtype)  # [E, 2]
        else:
            target = state.navigation.target_positions.squeeze(1)[:, :2].to(device=device, dtype=dtype)  # [E, 2]

        # Robot radius (single value in env cfg)
        robot_radius = torch.full((num_envs, 1), getattr(self.env_cfg, "safety_radius", 0.5), device=device, dtype=dtype)  # [E, 1]

        # Traffic (global list across all envs)
        traffic_pos = state.traffic.traffic_positions[:, :2].to(device=device, dtype=dtype)  # [T, 2]
        traffic_vel = state.traffic.traffic_velocities[:, :2].to(device=device, dtype=dtype)  # [T, 2]
        traffic_radius = state.traffic.traffic_safety_radius.to(device=device, dtype=dtype).view(1, -1)  # [1, T]
        total_traffic = traffic_pos.shape[0]

        # Filtered positions xi = p + v / l_i
        xi_i = p_i + v_i / self.l_i  # [E, 2]

        # Attractive term: k1 * (xi_i - target), with saturation
        xi_err = xi_i - target  # [E, 2]
        attractive = self._sat_vec2(self.k1 * xi_err, self.v_pref)  # [E, 2]

        # Repulsive term: sum_j b_ij * (xi_i - xi_j), b_ij = max(0, -dV/dnorm / norm)
        repulsive = torch.zeros_like(xi_i)
        if total_traffic > 0:
            xi_j = traffic_pos + traffic_vel / self.l_i  # [T, 2]
            # Pairwise differences xi_m_ij = xi_i - xi_j
            xi_i_exp = xi_i.unsqueeze(1)  # [E, 1, 2]
            xi_j_exp = xi_j.unsqueeze(0)  # [1, T, 2]
            xi_m_ij = xi_i_exp - xi_j_exp  # [E, T, 2]
            xi_m_norm = torch.norm(xi_m_ij, dim=-1)  # [E, T]

            # Distance thresholds per pair using radii
            # d1 = d1_scale * (r_i + r_j), d2 = d2_scale * (r_i + r_j)
            r_i = robot_radius  # [E, 1]
            r_j = traffic_radius  # [1, T]
            radii_sum = r_i + r_j  # [E, T]
            d1 = self.d1_scale * radii_sum
            d2 = self.d2_scale * radii_sum

            # Compute dV/dnorm (vectorized)
            dV_dnorm = self._compute_dV_dnorm(xi_m_norm, d1, d2)  # [E, T]

            # b_ij = -dV/dnorm / ||xi||, clamp to >= 0, handle zero norms
            inv_norm = torch.where(xi_m_norm > 1e-8, 1.0 / xi_m_norm, torch.zeros_like(xi_m_norm))
            b_ij = -dV_dnorm * inv_norm
            b_ij = torch.clamp(b_ij, min=0.0)  # [E, T]

            # Only interact within d2 (avoid far-away agents)
            within_range = (xi_m_norm < d2).to(b_ij.dtype)  # [E, T]
            b_ij = b_ij * within_range

            # Sum_j b_ij * xi_m_ij
            repulsive = torch.einsum("et,etd->ed", b_ij, xi_m_ij)  # [E, 2]

        total_cmd = attractive - repulsive  # [E, 2]
        v_cmd = -self._sat_vec2(total_cmd, self.v_pref)  # [E, 2]
        return v_cmd

    def act(self, observation, timestep, timesteps):
        # For skrl API, we do not need observation, timestep, timesteps
        if self.env is None or self.env.state is None:
            return torch.zeros((1, 2), device=self.device)
        state_env = self.env.state
        raw_xy = self._predict(state_env)  # [E, 2]
        scaled_xy = raw_xy / self.max_speed
        return scaled_xy, None, None

    def predict(self, observation: Dict[str, torch.Tensor], state: Optional[Any] = None, episode_start: Optional[Any] = None,
                deterministic: bool = True) -> tuple[torch.Tensor, Optional[Any]]:
        # Switch to env.state (consistent with ORCAPolicy)
        state_env = self.env.state
        raw_xy = self._predict(state_env)  # [E, 2]
        scaled_xy = raw_xy / self.max_speed
        return scaled_xy, None

