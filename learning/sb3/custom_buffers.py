"""
Custom rollout buffer based on the original storage.py
Compatible with SB3 interface while maintaining the original sampling mechanism
"""

import torch
import numpy as np
from typing import Dict, Generator, Optional, Union, NamedTuple
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class CustomRolloutBufferSamples(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    hidden_states: Dict[str, torch.Tensor]


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].view(T * N, *(_tensor[key].size()[2:]))
        return _tensor
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])


class CustomRecurrentRolloutBuffer(BaseBuffer):
    """
    自定义循环神经网络 Rollout Buffer
    基于原有 storage.py 的实现，保持原有的采样机制和 hidden state 管理
    只修改错位存储问题，与SB3保持一致
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        human_node_rnn_size: int,
        human_human_edge_rnn_size: int,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.human_node_rnn_size = human_node_rnn_size
        self.human_human_edge_rnn_size = human_human_edge_rnn_size
        
        # Get human_num from observation space
        if 'spatial_edges' in observation_space.spaces:
            self.human_num = observation_space.spaces['spatial_edges'].shape[0]
        else:
            self.human_num = 1
            
        # Initialize observation dict (与SB3保持一致，不使用错位)
        self.observations = {}
        for key, space in observation_space.spaces.items():
            self.observations[key] = torch.zeros(
                (self.buffer_size, self.n_envs, *space.shape), 
                dtype=torch.float32, 
                device=self.device
            )
            
        # Initialize hidden states dict (与SB3保持一致，不使用错位)
        self.recurrent_hidden_states = {}
        node_num = 1
        edge_num = self.human_num + 1
        
        self.recurrent_hidden_states['human_node_rnn'] = torch.zeros(
            self.buffer_size, self.n_envs, node_num, human_node_rnn_size,
            device=self.device
        )
        self.recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(
            self.buffer_size, self.n_envs, edge_num, human_human_edge_rnn_size,
            device=self.device
        )
        
        # Initialize other buffers (与SB3保持一致，不使用错位)
        self.rewards = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        self.values = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        self.returns = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        self.log_prob = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        
        # Action space handling
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = (1,)
            action_dtype = torch.long
        else:
            action_shape = action_space.shape
            action_dtype = torch.float32
            
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, *action_shape), 
            dtype=action_dtype, 
            device=self.device
        )
        
        # 保留原有的mask系统（关键模块）
        self.masks = torch.ones(self.buffer_size, self.n_envs, 1, device=self.device)
        
        self.episode_starts = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        # Advantages
        self.advantages = torch.zeros(self.buffer_size, self.n_envs, 1, device=self.device)
        
        self.pos = 0
        self.full = False
        
    def reset(self):
        """Reset the buffer"""
        # Reset observations
        for key in self.observations:
            self.observations[key].zero_()
            
        # Reset hidden states
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key].zero_()
            
        # Reset other tensors
        self.rewards.zero_()
        self.values.zero_()
        self.returns.zero_()
        self.log_prob.zero_()
        self.actions.zero_()
        self.masks.fill_(1.0)
        self.episode_starts.zero_()
        self.advantages.zero_()
        
        self.pos = 0
        self.full = False
        
    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        masks: torch.Tensor,
        hidden_states: Dict[str, torch.Tensor],
    ):
        """
        Add new data to the buffer
        与SB3保持一致，不使用错位机制，但保留mask系统
        """
        if len(log_prob.shape) < 2:
            # Reshape scalar to 1D tensor
            log_prob = log_prob.reshape(-1, 1)
        if len(value.shape) < 2:
            # Reshape scalar to 1D tensor
            value = value.reshape(-1, 1)
        if len(reward.shape) < 2:
            # Reshape scalar to 1D tensor
            reward = reward.reshape(-1, 1)
        if len(episode_start.shape) < 2:
            # Reshape scalar to 1D tensor
            episode_start = episode_start.reshape(-1, 1)
        if len(masks.shape) < 2:
            # Reshape scalar to 1D tensor
            masks = masks.reshape(-1, 1)
        # 与SB3保持一致：所有数据都存储在 pos 位置
        for key in self.observations:
            if key in obs:
                self.observations[key][self.pos].copy_(obs[key])
                
        for key in self.recurrent_hidden_states:
            if key in hidden_states:
                self.recurrent_hidden_states[key][self.pos].copy_(hidden_states[key])
                
        # 动作、奖励、价值、log_probs 存储在 pos 位置
        self.actions[self.pos].copy_(action)
        self.log_prob[self.pos].copy_(log_prob)
        self.values[self.pos].copy_(value)
        self.rewards[self.pos].copy_(reward)
        
        # 保留原有的mask系统
        self.masks[self.pos].copy_(masks)
        self.episode_starts[self.pos].copy_(episode_start)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor):
        """
        计算 returns 和 advantages
        采用SB3的逻辑，但保持tensor操作
        """
        # Convert to tensor if needed
        last_values = last_values.clone()
        
        last_gae_lam = torch.zeros_like(last_values)
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                
            delta = (
                self.rewards[step] 
                + self.gamma * next_values * next_non_terminal 
                - self.values[step]
            )
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        
        
    def get(self, batch_size: Optional[int] = None) -> Generator[CustomRolloutBufferSamples, None, None]:
        """
        Generator that yields batches of data
        使用原有的 recurrent_generator 采样策略
        """
        assert self.full, "Buffer must be full before sampling"
        
        num_mini_batch = max(1, (self.buffer_size * self.n_envs) // batch_size if batch_size else 4)
        
        # Use the original recurrent generator logic
        yield from self._recurrent_generator(num_mini_batch)
        
    def _recurrent_generator(self, num_mini_batch: int):
        """
        原有的 recurrent generator 逻辑
        """
        num_processes = self.n_envs
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = {}
            for key in self.observations:
                obs_batch[key] = []
                
            hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                hidden_states_batch[key] = []
                
            actions_batch = []
            values_batch = []
            returns_batch = []
            masks_batch = []
            log_probs_batch = []
            adv_batch = []
            
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                
                # Collect data for this environment
                for key in self.observations:
                    obs_batch[key].append(self.observations[key][:, ind])
                    
                for key in self.recurrent_hidden_states:
                    hidden_states_batch[key].append(self.recurrent_hidden_states[key][0:1, ind])
                    
                actions_batch.append(self.actions[:, ind])
                values_batch.append(self.values[:, ind])
                returns_batch.append(self.returns[:, ind])
                masks_batch.append(self.masks[:, ind])
                log_probs_batch.append(self.log_prob[:, ind])
                adv_batch.append(self.advantages[:, ind])
                
            T, N = self.buffer_size, num_envs_per_batch
            
            # Stack tensors
            actions_batch = torch.stack(actions_batch, 1)
            values_batch = torch.stack(values_batch, 1)
            returns_batch = torch.stack(returns_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            log_probs_batch = torch.stack(log_probs_batch, 1)
            adv_batch = torch.stack(adv_batch, 1)
            
            for key in obs_batch:
                obs_batch[key] = torch.stack(obs_batch[key], 1)
                
            for key in hidden_states_batch:
                temp = torch.stack(hidden_states_batch[key], 1)
                hidden_states_batch[key] = temp.view(N, *(temp.size()[2:]))
                
            # Flatten tensors using the original helper
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            values_batch = _flatten_helper(T, N, values_batch)
            returns_batch = _flatten_helper(T, N, returns_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            log_probs_batch = _flatten_helper(T, N, log_probs_batch)
            adv_batch = _flatten_helper(T, N, adv_batch)
            
            yield CustomRolloutBufferSamples(
                observations=obs_batch,
                actions=actions_batch,
                old_values=values_batch,
                old_log_prob=log_probs_batch,
                advantages=adv_batch,
                returns=returns_batch,
                masks=masks_batch,
                hidden_states=hidden_states_batch,
            ) 
    def _get_samples(self, batch_inds, env=None):
        raise NotImplementedError("CustomRecurrentRolloutBuffer does not support _get_samples; use get() instead.")