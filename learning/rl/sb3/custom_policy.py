"""
Custom policy wrapping selfAttn_srnn_temp_node for SB3 compatibility
Supports both Gaussian continuous and discrete action spaces
"""

import torch as th
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any, Type, List
from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.distributions import (
    Distribution, 
    CategoricalDistribution, 
    DiagGaussianDistribution,
    make_proba_distribution
)
from stable_baselines3.common.type_aliases import Schedule

from rl.networks.selfAttn_srnn_temp_node import selfAttn_merge_SRNN


class CustomSelfAttnPolicy(BasePolicy):
    """
    自定义策略类，包装 selfAttn_merge_SRNN 网络
    支持高斯分布连续动作空间和离散动作空间
    与SB3实现保持一致：actor_features直接作为latent_pi使用
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        args: Any,  # Your original args object
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
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        self.args = args
        self.lr_schedule = lr_schedule
        
        # Initialize the base network (selfAttn_merge_SRNN)
        self.base_network = selfAttn_merge_SRNN(observation_space, args, infer=False)
        self.is_recurrent = True
        
        # Get network output size (this is our latent_dim_pi)
        self.latent_dim_pi = self.base_network.output_size
        
        # Setup action distribution
        self._setup_action_distribution()
        
        # Initialize the optimizer (like SB3)
        self._build(lr_schedule)
        
    def _setup_action_distribution(self):
        """Setup action distribution based on action space"""
        if isinstance(self.action_space, spaces.Discrete):
            # Discrete action space
            num_outputs = self.action_space.n
            self.action_dist = CategoricalDistribution(num_outputs)
            
        elif isinstance(self.action_space, spaces.Box):
            # Continuous action space
            num_outputs = self.action_space.shape[0]
            self.action_dist = DiagGaussianDistribution(num_outputs)
            
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")
            
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        与SB3的_build方法完全一致
        
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Create action_net and log_std (like SB3)
        latent_dim_pi = self.latent_dim_pi
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=0.0
            )
            # 对DiagGaussianDistribution的action_net做正交初始化，gain=1，bias=0
            nn.init.orthogonal_(self.action_net.weight, gain=1)
            nn.init.constant_(self.action_net.bias, 0)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            # 对CategoricalDistribution的action_net做正交初始化，gain=0.01，bias=0
            nn.init.orthogonal_(self.action_net.weight, gain=0.01)
            nn.init.constant_(self.action_net.bias, 0)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
            
        # Setup optimizer with initial learning rate (like SB3)
        # 直接使用self.parameters()，与SB3完全一致
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get parameters for policy reconstruction"""
        data = super()._get_constructor_parameters()
        data.update({
            "lr_schedule": self.lr_schedule,
            "args": self.args,
            "optimizer_class": self.optimizer_class,
            "optimizer_kwargs": self.optimizer_kwargs,
        })
        return data
        
    def reset_noise(self, n_envs: int = 1) -> None:
        """Reset noise if using gSDE"""
        pass  # Not implemented for this policy
        
    def forward(
        self, 
        obs: Dict[str, th.Tensor], 
        hidden_states: Dict[str, th.Tensor],
        masks: th.Tensor,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        前向传播
        
        Args:
            obs: 观察数据字典
            hidden_states: 隐藏状态字典
            masks: 掩码
            deterministic: 是否确定性输出
            
        Returns:
            values, actions, log_probs, new_hidden_states
        """
        # Forward through base network
        values, actor_features, new_hidden_states = self.base_network(
            obs, hidden_states, masks, infer=True
        )
        
        # Get action distribution (like SB3: actor_features -> action_net -> distribution)
        distribution = self._get_action_dist_from_latent(actor_features)
        
        # Sample actions
        if deterministic:
            actions = distribution.get_actions(deterministic=True)
        else:
            actions = distribution.get_actions(deterministic=False)
            
        # Compute log probabilities
        log_probs = distribution.log_prob(actions)
        
        return values, actions, log_probs, new_hidden_states
        
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        与SB3的_get_action_dist_from_latent方法完全一致
        
        :param latent_pi: Latent code for the actor (our actor_features)
        :return: Action distribution
        """
        # 先通过action_net，与SB3完全一致
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")
            
    def predict_values(
        self, 
        obs: Dict[str, th.Tensor],
        hidden_states: Dict[str, th.Tensor],
        masks: th.Tensor
    ) -> th.Tensor:
        """Predict values only"""
        values, _, _ = self.base_network(obs, hidden_states, masks, infer=True)
        return values
        
    def evaluate_actions(
        self, 
        obs: Dict[str, th.Tensor],
        actions: th.Tensor,
        hidden_states: Dict[str, th.Tensor],
        masks: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions for training
        
        Returns:
            values, log_probs, entropy
        """
        # Forward through base network (training mode)
        values, actor_features, _ = self.base_network(obs, hidden_states, masks, infer=False)
        
        # Get action distribution (like SB3)
        distribution = self._get_action_dist_from_latent(actor_features)
        
        # Compute log probabilities and entropy
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_probs, entropy
    
    def _predict(
        self,
        observation: Dict[str, th.Tensor],
        hidden_states: Dict[str, th.Tensor],
        masks: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, hidden_states = self.get_distribution(observation, hidden_states, masks)
        return distribution.get_actions(deterministic=deterministic), hidden_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        hidden_states: Optional[Dict[str, np.ndarray]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Predict action for single environment
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(obs_tensor, dict):
            n_envs = obs_tensor[list(obs_tensor.keys())[0]].shape[0]
        else:
            n_envs = obs_tensor.shape[0]
        if hidden_states is None:
            # Initialize hidden states
            hidden_states_tensor = self._init_hidden_states(n_envs)
        else:
            hidden_states_tensor = {}
            # 区分hidden_states是不是tensor
            if isinstance(hidden_states, dict):
                for key, value in hidden_states.items():
                    hidden_states_tensor[key] = th.as_tensor(value, device=self.device)
            else:
                hidden_states_tensor = th.as_tensor(hidden_states, device=self.device)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])
        masks_tensor = th.tensor(1 - episode_start, dtype=th.float32, device=self.device)
        if len(masks_tensor.shape) < 2:
            masks_tensor = masks_tensor.view(-1, 1)

        with th.no_grad():
            # Convert to tensors
            # Forward pass
            _, actions, _, new_hidden_states = self.forward(
                obs_tensor, hidden_states_tensor, masks_tensor, deterministic
            )
            
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            new_hidden_states_np = {}
            for key, value in new_hidden_states.items():
                new_hidden_states_np[key] = value.cpu().numpy()

            # if not vectorized_env:
            #     actions = actions.squeeze(axis=0)     
        return clipped_actions, new_hidden_states_np
        
    def _init_hidden_states(self, n_envs: int) -> Dict[str, th.Tensor]:
        """Initialize hidden states for n_envs environments"""
        hidden_states = {}
        
        # Get human_num from observation space
        if 'spatial_edges' in self.observation_space.spaces:
            human_num = self.observation_space.spaces['spatial_edges'].shape[0]
        else:
            human_num = 1
            
        node_num = 1
        edge_num = human_num + 1
        
        hidden_states['human_node_rnn'] = th.zeros(
            n_envs, node_num, self.args.human_node_rnn_size,
            device=self.device
        )
        hidden_states['human_human_edge_rnn'] = th.zeros(
            n_envs, edge_num, self.args.human_human_edge_rnn_size,
            device=self.device
        )
        
        return hidden_states
        
    def get_distribution(
        self, 
        obs: Dict[str, th.Tensor],
        hidden_states: Dict[str, th.Tensor],
        masks: th.Tensor
    ) -> Tuple[Distribution, Dict[str, th.Tensor]]:
        """Get action distribution"""
        _, actor_features, new_hidden_states = self.base_network(
            obs, hidden_states, masks, infer=True
        )
        distribution = self._get_action_dist_from_latent(actor_features)
        return distribution, new_hidden_states


# Compatibility alias for easier import
CustomRecurrentActorCriticPolicy = CustomSelfAttnPolicy
