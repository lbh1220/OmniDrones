import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, Mapping, Any
from einops.layers.torch import Rearrange


class LidarFeatureExtractor(nn.Module):
    """
    LiDAR 分支特征提取模块: (B, C, H, V) -> (B, 128)
    """
    def __init__(self, lidar_shape, lidar_fc_in_dim=128):
        super().__init__()
        lidar_c = lidar_shape[0]
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=lidar_c, out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *lidar_shape)
            conv_out = self.convs(dummy)
            flat = Rearrange("n c w h -> n (c w h)")(conv_out)
            fc_in = flat.shape[1]
        self.net = nn.Sequential(
            self.convs,
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(fc_in, lidar_fc_in_dim), nn.LayerNorm(lidar_fc_in_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicObstacleFeatureExtractor(nn.Module):
    """
    动态障碍物分支特征提取模块: (B, C, N, D) -> (B, 64)
    """
    def __init__(self, dyn_shape, dyn_fc_in_dim=64):
        super().__init__()
        dyn_flatten_dim = int(np.prod(dyn_shape))
        internal_features_dim = dyn_fc_in_dim*2
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dyn_flatten_dim, internal_features_dim),
            nn.LeakyReLU(), nn.LayerNorm(internal_features_dim),
            nn.Linear(internal_features_dim, dyn_fc_in_dim),
            nn.LeakyReLU(), nn.LayerNorm(dyn_fc_in_dim),
        )

    def forward(self, x):
        return self.net(x)


class NavRLFeaturesNetwork(nn.Module):
    """
    一个适用于 NavRL 环境的特征提取网络。
    从 'lidar'、'traffic_states' 和 'robot_node' 观测中提取特征。

    Notes:
    - 该实现不依赖 action_space 或 device；如需放到 GPU，请在外部调用 .to(device)。
    - 输入参数仅需 observation_space 与 features_dim。
    """
    def __init__(self, observation_space, features_dim: int = 256, **kwargs):
        super().__init__()
        self.features_dim = features_dim

        # --- 1. LiDAR 分支 ---
        # 从 obs_space 获取 lidar 形状 (C, H, V)
        # 在原始代码中, C (通道数) = 1
        lidar_shape = observation_space["lidar"].shape
        self.lidar_extractor = LidarFeatureExtractor(lidar_shape, lidar_fc_in_dim=128)

        # --- 2. 动态障碍物分支 ---
        # dyn: (C, N, 10), C=1
        dyn_shape = observation_space["traffic_states"].shape
        self.dyn_extractor = DynamicObstacleFeatureExtractor(dyn_shape, dyn_fc_in_dim=64)
        
        # --- 3. 共享 MLP 分支 ---
        # 获取 state 维度 (例如: 8)
        state_dim = observation_space["robot_node"].shape[-1]
        
        # 最终融合: [lidar(128) + state(8) + dyn(64)]
        mlp_input_dim = 128 + 64 + state_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, features_dim), nn.LeakyReLU(), nn.LayerNorm(features_dim),
        )

        # 初始化权重 (你提供的函数很好)
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 注意: 原始 ppo.py 对 actor/critic 使用 0.01 的增益
                # 你使用的 np.sqrt(2) 是 ReLU/LeakyReLU 的标准增益，完全合理
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

    def forward(self, obs):
        """
        前向传播
        Args:
            obs: dict，包含 'lidar' (B,C,H,V)、'traffic_states' (B,C,N,D)、'robot_node' (B, D) 或 (B,1,D)
        Returns:
            features: (B, features_dim)
        """
        # 1. LiDAR 分支: (B, C, H, V) -> (B, 128)
        lidar_feat = self.lidar_extractor(obs["lidar"])

        # 2. 动态障碍分支: (B, C, N, D) -> (B, 64)
        dyn_feat = self.dyn_extractor(obs["traffic_states"])

        # 3. 机器人状态: (B, D)
        state_data = obs["robot_node"]
        if state_data.dim() == 3:
            state_data = state_data.squeeze(1)

        # 4. 融合
        combined_feat = torch.cat([lidar_feat, state_data, dyn_feat], dim=-1)

        # 5. 共享 MLP
        features = self.mlp(combined_feat)
        return features



class DynamicTrafficNavRLFeaturesNetwork(nn.Module):
    """
    一个适用于 NavRL 环境的特征提取网络。
    从 'lidar'、'traffic_states' 和 'robot_node' 观测中提取特征。
    """
    
    def __init__(self, observation_space, features_dim: int = 256, **kwargs):
        super().__init__()
        self.features_dim = features_dim

        # --- 1. LiDAR 分支 ---
        # 从 obs_space 获取 lidar 形状 (C, H, V)
        # 在原始代码中, C (通道数) = 1
        lidar_shape = observation_space["lidar"].shape
        self.lidar_extractor = LidarFeatureExtractor(lidar_shape, lidar_fc_in_dim=128)
        from .traffic_attn import DynamicTrafficFeaturesExtractor

        self.traffic_extractor = DynamicTrafficFeaturesExtractor(observation_space, features_dim=128, use_type_split=True)
        
        # --- 3. 共享 MLP 分支 ---
        # 获取 state 维度 (例如: 8)
        state_dim = observation_space["robot_node"].shape[-1]
        
        # 最终融合: [lidar(128) + state(8) + traffic(128)]
        mlp_input_dim = 128 + state_dim + 128
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, features_dim), nn.LeakyReLU(), nn.LayerNorm(features_dim),
        )

        # 初始化权重 (你提供的函数很好)
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 注意: 原始 ppo.py 对 actor/critic 使用 0.01 的增益
                # 你使用的 np.sqrt(2) 是 ReLU/LeakyReLU 的标准增益，完全合理
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

    def forward(self, obs):
        """
        前向传播
        Args:
            obs: dict，包含 'lidar' (B,C,H,V)、'traffic_states' (B,N,D)、'robot_node' (B, D) 或 (B,1,D)
        Returns:
            features: (B, features_dim)
        """
        # 1. LiDAR 分支: (B, C, H, V) -> (B, 128)
        lidar_feat = self.lidar_extractor(obs["lidar"])

        # 2. 交通分支: (B, C, N, D) -> (B, 128)
        traffic_feat = self.traffic_extractor(obs)

        # 3. 机器人状态: (B, D)
        state_data = obs["robot_node"]
        if state_data.dim() == 3:
            state_data = state_data.squeeze(1)

        # 4. 融合
        combined_feat = torch.cat([lidar_feat, state_data, traffic_feat], dim=-1)

        # 5. 共享 MLP
        features = self.mlp(combined_feat)
        return features