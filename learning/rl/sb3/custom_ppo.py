"""
Custom PPO algorithm compatible with custom policy and buffer
Based on original PPO implementation while conforming to SB3 interface
"""

import warnings
import torch as th
import numpy as np
import time
import sys
from typing import Any, Dict, Optional, Type, Union, Callable, Tuple
import argparse
from gymnasium import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from .custom_policy import CustomSelfAttnPolicy
from .custom_buffers import CustomRecurrentRolloutBuffer

def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, dict]:
    """
    Moves the observation to the given device and converts to float32 tensor.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device, dtype float32.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device, dtype=th.float32)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device, dtype=th.float32) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

class CustomPPO(OnPolicyAlgorithm):
    """
    自定义 PPO 算法
    基于原有 PPO 流程，适配自定义的 policy 和 buffer
    """
    
    def __init__(
        self,
        policy: Union[str, Type[CustomSelfAttnPolicy]],
        env: Union[GymEnv, str],
        args: Optional[Any] = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # if args is None:
        #     args = policy_kwargs.get('args', None)
        # if args is None:
        #     raise ValueError("args is not provided")
        # if isinstance(args, dict):
        #     args = argparse.Namespace(**args)
        self.args = args
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        if isinstance(self.args, argparse.Namespace):
            data['args'] = vars(self.args)
        else:
            data['args'] = self.args
        return data
    
    def _setup_model(self) -> None:
        """Setup model components"""
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        
        # Setup policy
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
            
        self.policy_kwargs["args"] = self.args
        self.policy_kwargs["use_beta"] = self.args.action_space_type == "beta"
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
            
        # Setup rollout buffer
        buffer_size = self.n_steps
        
        self.rollout_buffer = CustomRecurrentRolloutBuffer(
            buffer_size,
            self.observation_space,
            self.action_space,
            human_node_rnn_size=self.args.human_node_rnn_size,
            human_human_edge_rnn_size=self.args.human_human_edge_rnn_size,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        # Initialize hidden states if needed
        # if not hasattr(self, '_last_hidden_states') or self._last_hidden_states is None:
        self._last_hidden_states = self.policy._init_hidden_states(self.n_envs)
            # this is a dict tensor
            # self._last_hidden_states['human_node_rnn'] = th.zeros(1, self.n_envs, self.args.human_node_rnn_size)
            # self._last_hidden_states['human_human_edge_rnn'] = th.zeros(1, self.n_envs, self.args.human_human_edge_rnn_size)
            
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomRecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a rollout buffer.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        
        # Switch to eval mode
        self.policy.eval()
        
        n_steps = 0
        rollout_buffer.reset()
        

        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to tensor
                # self._last_obs and self._last_episode_starts are initialized in _setup_learn()
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # masks_tensor is a tensor of shape (n_envs, 1), 1 is the number of robots
                masks_tensor = th.tensor(1 - self._last_episode_starts, dtype=th.float32, device=self.device).unsqueeze(-1)
                # episode_starts is a tensor of shape (n_envs,), following the sb3 standard
                #episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                # Get actions and values
                last_hidden_states = self._last_hidden_states.copy()
                values, actions, log_probs, new_hidden_states = self.policy.forward(
                    obs_tensor, 
                    last_hidden_states,
                    masks_tensor,
                    deterministic=False
                )
                
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)

                    
            # Step environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            self.num_timesteps += env.num_envs
            
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
                
            self._update_info_buffer(infos, dones)
            n_steps += 1
            
            # Handle timeout terminations
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            time_limit_truncated_idx = []
            if dones.any():
                for idx, done_ in enumerate(dones):
                    if (
                        done_
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        # new_obs已经是reset之后的了， infos中terminal_observation是reset之前的
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_lstm_state = {}
                            terminal_lstm_state['human_node_rnn'] = new_hidden_states['human_node_rnn'][idx : idx + 1].contiguous()
                            terminal_lstm_state['human_human_edge_rnn'] = new_hidden_states['human_human_edge_rnn'][idx : idx + 1].contiguous()
                            # terminal_episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                            terminal_masks = th.tensor([1], dtype=th.float32, device=self.device).unsqueeze(-1)
                            terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, terminal_masks)[0].cpu().numpy().item()
                            # terminal_value = self.policy.predict_values(terminal_obs, new_hidden_states, masks_tensor)
                        rewards[idx] += self.gamma * terminal_value


            # Store data in buffer
            episode_starts = th.as_tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
            rewards_tensor = th.as_tensor(rewards, dtype=th.float32, device=self.device).view(-1, 1)
            
            rollout_buffer.add(
                obs_tensor,
                actions,
                rewards_tensor,
                episode_starts,
                values,
                log_probs,
                masks_tensor,
                self._last_hidden_states,
            )
            
            # Update last observations and states
            self._last_obs = new_obs
            self._last_hidden_states = {
                key: value.clone() for key, value in new_hidden_states.items()
            }
            self._last_episode_starts = dones
            
        # Compute value for the last timestep
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.device)
            # episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            masks_tensor = th.tensor(1 - dones, dtype=th.float32, device=self.device).unsqueeze(-1)
            last_hidden_states = self._last_hidden_states.copy()
            values = self.policy.predict_values(
                obs_tensor, 
                last_hidden_states,
                masks_tensor
            )
            
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=th.tensor(dones, dtype=th.float32, device=self.device).unsqueeze(-1))
        
        callback.on_rollout_end()
        
        return True
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode
        self.policy.train()
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range
        if callable(self.clip_range):
            clip_range = self.clip_range(self._current_progress_remaining)
            
        # Optional: clip range for value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            if callable(self.clip_range_vf):
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
            else:
                clip_range_vf = self.clip_range_vf
                
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        continue_training = True
        
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    
                # Evaluate actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.hidden_states,
                    rollout_data.masks,
                )
                # if isinstance(self.action_space, spaces.Discrete):
                log_prob = log_prob.unsqueeze(-1)
                # values = values.flatten()
                
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Value loss
                if clip_range_vf is not None:
                    value_pred_clipped = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    value_loss_unclipped = (values - rollout_data.returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - rollout_data.returns).pow(2)
                    value_loss = 0.5 * th.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * th.nn.functional.mse_loss(rollout_data.returns, values)
                    
                # value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # Entropy loss
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                    
                entropy_losses.append(entropy_loss.item())
                
                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                    
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1    
            if not continue_training:
                break
                
        explained_var = explained_variance(self.rollout_buffer.values.flatten().cpu().numpy(), self.rollout_buffer.returns.flatten().cpu().numpy())
        
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CustomPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        This function is designed for evaluation, the input is episode_start;
        Policy needs to convert episode_start to masks;
        """
        return self.policy.predict(observation, state, episode_start, deterministic)