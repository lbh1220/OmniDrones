import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, Mapping, Any
from einops.layers.torch import Rearrange
from skrl.models.torch import Model
class NavRLFeaturesNetwork(Model):
    """
    一个适用于 NavRL 环境的特征提取网络 (SKRL 兼容)。
    它从 'lidar'、'dynamic_obstacle' 和 'state' 观测中提取特征。
    """
    def __init__(self, observation_space, action_space, device=None, features_dim: int = 256, **kwargs):
        super().__init__(observation_space, action_space, device, **kwargs)
        self.features_dim = features_dim

        # --- 1. LiDAR 分支 ---
        # 从 obs_space 获取 lidar 形状 (C, H, V)
        # 在原始代码中, C (通道数) = 1
        lidar_shape = observation_space["lidar"].shape
        lidar_c = lidar_shape[0]
        
        # 1.a. 定义卷积部分
        self.lidar_convs = nn.Sequential(
            # 替换 LazyConv2d: 明确指定 in_channels=lidar_c
            nn.Conv2d(in_channels=lidar_c, out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
        )
        
        # 1.b. 自动计算卷积输出 -> 全连接输入的维度
        # 我们在 __init__ 中只做一次，以避免使用 LazyLinear
        with torch.no_grad():
            # 创建一个匹配 obs_space 的虚拟张量
            dummy_lidar = torch.zeros(1, *lidar_shape, device=self.device)
            # 推送它通过卷积层
            conv_out = self.lidar_convs(dummy_lidar)
            # 推送它通过展平层
            flat_out = Rearrange("n c w h -> n (c w h)")(conv_out)
            # 获取展平后的维度
            lidar_fc_in_dim = flat_out.shape[1]

        # 1.c. 定义完整的 lidar_net (卷积 + 展平 + 全连接)
        self.lidar_net = nn.Sequential(
            self.lidar_convs,
            Rearrange("n c w h -> n (c w h)"),
            # 替换 LazyLinear: 明确指定 in_features
            nn.Linear(lidar_fc_in_dim, 128), nn.LayerNorm(128),
        )

        # --- 2. 动态障碍物分支 ---
        # dyn: (C, N, 10), C=1
        dyn_flatten_dim = self._dyn_flatten_dim(observation_space)
        self.dyn_net = nn.Sequential(
            nn.Flatten(),  # 替代 Rearrange
            nn.Linear(dyn_flatten_dim, 128),
            nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(), nn.LayerNorm(64),
        )
        
        # --- 3. 共享 MLP 分支 ---
        # 获取 state 维度 (例如: 8)
        state_dim = self._robot_state_dim(observation_space)
        
        # 最终融合: [lidar(128) + state(8) + dyn(64)]
        mlp_input_dim = 128 + 64 + state_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, features_dim), nn.LeakyReLU(), nn.LayerNorm(features_dim),
        )

        # 初始化权重 (你提供的函数很好)
        self._initialize_weights()

    def _dyn_flatten_dim(self, obs_space):
        # 计算 (C, N, D) 展平后的大小
        return int(np.prod(obs_space["dynamic_obstacle"].shape))

    def _robot_state_dim(self, obs_space):
        # 我将 "robot_state" 改为 "state" 以匹配 ppo.py
        return obs_space["robot_state"].shape[-1]
        
    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 注意: 原始 ppo.py 对 actor/critic 使用 0.01 的增益
                # 你使用的 np.sqrt(2) 是 ReLU/LeakyReLU 的标准增益，完全合理
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

    def compute(self, obs):
        """
        SKRL 的前向传播方法
        """
        # obs 是一个字典, 包含 'state', 'lidar', 'dynamic_obstacle' 等
        
        # 1. 处理 LiDAR: (B, C, H, V) -> (B, 128)
        lidar_feat = self.lidar_net(obs["lidar"])
        
        # 2. 处理动态障碍物: (B, C, N, 10) -> (B, 64)
        dyn_feat = self.dyn_net(obs["dynamic_obstacle"])
        
        # 3. 获取状态: (B, 8)
        # 同样, 我将 "robot_state" 改为 "state"
        state_data = obs["robot_state"]
        
        # 4. 拼接所有特征
        # (B, 128 + 8 + 64)
        combined_feat = torch.cat([lidar_feat, state_data, dyn_feat], dim=-1)
        
        # 5. 通过共享 MLP
        # (B, features_dim)
        features = self.mlp(combined_feat)
        
        # SKRL 特征提取器应返回提取的特征
        return features