from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


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
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
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
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )
        
        # 用GRU替换LSTM
        # 替换actor的LSTM为GRU
        self.lstm_actor = nn.GRU(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        
        # 如果启用了critic的GRU，则替换
        if self.enable_critic_lstm:
            self.lstm_critic = nn.GRU(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    
    @staticmethod
    def _process_sequence(
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
