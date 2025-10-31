"""
GRU-based Recurrent Actor-Critic Policy for SB3
基于GRU的循环Actor-Critic策略，兼容sb3_contrib的接口

这个模块实现了一个使用GRU而不是LSTM的循环策略，同时保持与sb3_contrib完全的接口兼容性。
"""

from typing import Any, Optional, Union
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class GRURecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    基于GRU的循环Actor-Critic策略
    
    与原始的RecurrentActorCriticPolicy兼容，但使用GRU而不是LSTM。

    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        gru_hidden_size: int = 256,
        n_gru_layers: int = 1,
        shared_gru: bool = True,
        enable_critic_gru: bool = False,
        gru_kwargs: Optional[dict[str, Any]] = None,
    ):
        # 将GRU参数映射为LSTM参数以便父类初始化
        self.gru_hidden_size = gru_hidden_size
        self.n_gru_layers = n_gru_layers
        self.shared_gru = shared_gru
        self.enable_critic_gru = enable_critic_gru
        self.gru_kwargs = gru_kwargs or {}
        
        # 调用父类初始化，临时使用LSTM参数
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size=gru_hidden_size,
            n_lstm_layers=n_gru_layers,
            shared_lstm=shared_gru,
            enable_critic_lstm=enable_critic_gru,
            lstm_kwargs=gru_kwargs,
        )
        
        # 用GRU替换LSTM
        self._replace_lstm_with_gru()
        
        # 更新hidden state形状以匹配GRU
        self.gru_hidden_state_shape = (n_gru_layers, 1, gru_hidden_size)
        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _replace_lstm_with_gru(self):
        """将LSTM模块替换为GRU模块"""
        # 替换actor的LSTM为GRU
        self.gru_actor = nn.GRU(
            self.features_dim,
            self.gru_hidden_size,
            num_layers=self.n_gru_layers,
            **self.gru_kwargs,
        )
        delattr(self, 'lstm_actor')
        
        # 如果启用了critic的GRU，则替换
        if self.enable_critic_gru:
            self.gru_critic = nn.GRU(
                self.features_dim,
                self.gru_hidden_size,
                num_layers=self.n_gru_layers,
                **self.gru_kwargs,
            )
            if hasattr(self, 'lstm_critic'):
                delattr(self, 'lstm_critic')
        else:
            self.gru_critic = None
    
    @staticmethod
    def _process_sequence_gru(
        features: th.Tensor,
        gru_states: tuple[th.Tensor, th.Tensor],  # (hidden, dummy_cell)
        episode_starts: th.Tensor,
        gru: nn.GRU,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        使用GRU进行序列前向传播
        
        :param features: 输入特征张量
        :param gru_states: GRU状态 (hidden_state, dummy_cell_state)
        :param episode_starts: 指示新episode开始的位置
        :param gru: GRU模块
        :return: GRU输出和更新的状态
        """
        # 提取hidden state，忽略dummy cell state
        hidden_state = gru_states[0]
        dummy_cell_state = gru_states[1]
        
        # GRU逻辑
        n_seq = hidden_state.shape[1]
        
        # Batch to sequence
        features_sequence = features.reshape((n_seq, -1, gru.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        
        # 如果序列中间不需要重置状态，可以避免for循环以加速
        if th.all(episode_starts == 0.0):
            gru_output, new_hidden_state = gru(features_sequence, hidden_state)
            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)
            # 返回格式：(hidden, dummy_cell) 以保持兼容性
            return gru_output, (new_hidden_state, dummy_cell_state)
        
        gru_output = []
        current_hidden = hidden_state
        
        # 逐步处理序列
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, current_hidden = gru(
                features.unsqueeze(dim=0),
                # 在新episode开始时重置状态
                (1.0 - episode_start).view(1, n_seq, 1) * current_hidden,
            )
            gru_output.append(hidden)
        
        # Sequence to batch
        gru_output = th.flatten(th.cat(gru_output).transpose(0, 1), start_dim=0, end_dim=1)
        # 返回格式：(hidden, dummy_cell) 以保持兼容性
        return gru_output, (current_hidden, dummy_cell_state)
    
    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,  # 保持原始名称以兼容接口
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        在所有网络中进行前向传播（actor和critic）
        
        :param obs: 观察
        :param lstm_states: 最后的hidden和memory状态（兼容LSTM接口）
        :param episode_starts: 是否对应新episode开始
        :param deterministic: 是否使用确定性动作
        :return: 动作、价值和动作的log概率
        """
        # 预处理观察
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
            
        # Actor的GRU前向传播
        latent_pi, gru_states_pi = self._process_sequence_gru(
            pi_features, lstm_states.pi, episode_starts, self.gru_actor
        )
        
        # Critic的处理
        if self.gru_critic is not None:
            # 使用单独的critic GRU
            latent_vf, gru_states_vf = self._process_sequence_gru(
                vf_features, lstm_states.vf, episode_starts, self.gru_critic
            )
        elif self.shared_gru:
            # 重用actor的GRU特征但不反向传播
            latent_vf = latent_pi.detach()
            gru_states_vf = (gru_states_pi[0].detach(), gru_states_pi[1].detach())
        else:
            # Critic只有前馈网络
            latent_vf = self.critic(vf_features)
            gru_states_vf = gru_states_pi
        
        # 通过MLP提取器
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        
        # 评估给定观察的价值
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob, RNNStates(gru_states_pi, gru_states_vf)
    
    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple:
        """获取当前策略分布"""
        features = super(RecurrentActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence_gru(features, lstm_states, episode_starts, self.gru_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states
    
    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """根据当前策略预测价值"""
        features = super(RecurrentActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        
        if self.gru_critic is not None:
            latent_vf, _ = self._process_sequence_gru(features, lstm_states, episode_starts, self.gru_critic)
        elif self.shared_gru:
            # 使用actor的GRU
            latent_pi, _ = self._process_sequence_gru(features, lstm_states, episode_starts, self.gru_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)
        
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)
    
    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor, 
        lstm_states: RNNStates, 
        episode_starts: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """根据当前策略评估动作"""
        # 预处理观察
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
            
        latent_pi, _ = self._process_sequence_gru(pi_features, lstm_states.pi, episode_starts, self.gru_actor)
        
        if self.gru_critic is not None:
            latent_vf, _ = self._process_sequence_gru(vf_features, lstm_states.vf, episode_starts, self.gru_critic)
        elif self.shared_gru:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)
        
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
    
    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        """获取给定观察的动作"""
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states


class GRUMultiInputActorCriticPolicy(GRURecurrentActorCriticPolicy):
    """
    GRU-based Multi Input Actor Critic policy class for actor-critic algorithms.
    支持字典观察空间的GRU循环策略
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        gru_hidden_size: int = 256,
        n_gru_layers: int = 1,
        shared_gru: bool = True,
        enable_critic_gru: bool = False,
        gru_kwargs: Optional[dict[str, Any]] = None,
    ):
        # 如果没有指定特征提取器，使用CombinedExtractor
        if features_extractor_class is None:
            from stable_baselines3.common.torch_layers import CombinedExtractor
            features_extractor_class = CombinedExtractor
            
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            gru_hidden_size,
            n_gru_layers,
            shared_gru,
            enable_critic_gru,
            gru_kwargs,
        )
