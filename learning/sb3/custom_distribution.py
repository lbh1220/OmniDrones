"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal, Beta

from stable_baselines3.common.preprocessing import get_action_dim

SelfBetaDistribution = TypeVar("SelfBetaDistribution", bound="BetaDistribution")

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.distributions import sum_independent_dims


class BetaDistribution(Distribution):
    """
    Beta distribution for continuous actions with natural bounds in [0, 1].
    
    Beta distribution is particularly useful for continuous control tasks where actions
    are naturally bounded. The distribution has two parameters: alpha (concentration1) 
    and beta (concentration0).
    
    :param action_dim: Dimension of the action space.
    :param min_alpha: Minimum value for alpha parameter to ensure numerical stability.
    :param min_beta: Minimum value for beta parameter to ensure numerical stability.
    """

    def __init__(self, action_dim: int, min_alpha: float = 1e-6, min_beta: float = 1e-6):
        super().__init__()
        self.action_dim = action_dim
        self.min_alpha = min_alpha
        self.min_beta = min_beta
        self.alpha = None
        self.beta = None

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers that represent the distribution:
        Two separate networks will output the alpha and beta parameters of the Beta distribution.
        We use softplus activation to ensure both parameters are positive.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param alpha_init: Initial value for alpha parameter (concentration1)
        :param beta_init: Initial value for beta parameter (concentration0)
        :return: Tuple of (alpha_net, beta_net)
        """
        alpha_net = nn.Linear(latent_dim, self.action_dim)
        beta_net = nn.Linear(latent_dim, self.action_dim)
        

        
        return alpha_net, beta_net

    def proba_distribution(
        self: SelfBetaDistribution, alpha_logits: th.Tensor, beta_logits: th.Tensor
    ) -> SelfBetaDistribution:
        """
        Create the Beta distribution given its parameters (alpha, beta)

        :param alpha_logits: Raw output from alpha network
        :param beta_logits: Raw output from beta network
        :return: self
        """
        # Use softplus to ensure positive parameters and add minimum values for stability
        self.alpha = th.nn.functional.softplus(alpha_logits) + self.min_alpha
        self.beta = th.nn.functional.softplus(beta_logits) + self.min_beta
        
        self.distribution = Beta(self.alpha, self.beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the Beta distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: Actions in [0, 1] range
        :return: Log probabilities
        """
        # Ensure actions are in valid range [0, 1]
        actions = th.clamp(actions, min=1e-8, max=1.0 - 1e-8)
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        """
        Calculate the entropy of the Beta distribution.
        """
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        """
        Sample from the Beta distribution using reparameterization trick.
        """
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """
        Return the mode of the Beta distribution.
        Mode = (alpha - 1) / (alpha + beta - 2) when alpha > 1 and beta > 1
        Otherwise, we return the mean.
        """
        # Check if we can compute the mode (alpha > 1 and beta > 1)
        valid_mode = (self.alpha > 1.0) & (self.beta > 1.0)
        
        # Compute mode where valid
        mode = (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        
        # Use mean where mode is not valid
        mean = self.alpha / (self.alpha + self.beta)
        
        return th.where(valid_mode, mode, mean)

    def actions_from_params(
        self, alpha_logits: th.Tensor, beta_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """
        Generate actions from distribution parameters.
        
        :param alpha_logits: Raw output from alpha network
        :param beta_logits: Raw output from beta network
        :param deterministic: Whether to use deterministic (mode) or stochastic (sample) actions
        :return: Actions in [0, 1] range
        """
        # Update the probability distribution
        self.proba_distribution(alpha_logits, beta_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, alpha_logits: th.Tensor, beta_logits: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action given the distribution parameters.

        :param alpha_logits: Raw output from alpha network
        :param beta_logits: Raw output from beta network
        :return: Tuple of (actions, log_probabilities)
        """
        actions = self.actions_from_params(alpha_logits, beta_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob



# Usage example for BetaDistribution:
#
# 1. 在策略类中使用Beta分布:
#    policy = CustomSelfAttnPolicy(
#        observation_space=obs_space,
#        action_space=action_space,  # 确保是spaces.Box类型
#        lr_schedule=lr_schedule,
#        args=args,
#        use_beta=True,  # 启用Beta分布
#        # 其他参数...
#    )
#
# 2. Beta分布的优势:
#    - 天然有界在[0,1]区间，适合连续控制任务
#    - 可以通过alpha和beta参数控制分布形状
#    - 当alpha=beta=1时是均匀分布，适合初始探索
#    - 当alpha>1和beta>1时是单峰分布，适合收敛阶段
#
# 3. 动作空间映射:
#    - Beta分布输出[0,1]范围的动作
#    - 可以通过线性变换映射到任意范围[low, high]:
#      action_scaled = action * (high - low) + low
#
# 4. 参数选择建议:
#    - min_alpha和min_beta: 保持默认值1.0即可
#    - alpha_init和beta_init: 初始化为1.0-2.0之间较好
#    - 网络会自动学习合适的alpha和beta值
