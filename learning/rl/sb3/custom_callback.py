import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque
import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CheckpointCallback

class RewardCallback(BaseCallback):
    def __init__(self, 
                 check_freq=100, 
                 save_path: str = "checkpoints",
                 name_prefix: str = "sb3_train",
                 verbose=0):
        super(RewardCallback, self).__init__(verbose)

        self.episode_num = 0
        self.success_num = 0
        self.collision_num = 0
        self.timeout_num = 0
        self.check_freq = check_freq
        self.check_num = 0
        self.best_mean_reward = -np.inf
        self.best_success_rate = 0.0
        self.save_path = save_path
        self.name_prefix = name_prefix
    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_rollout_end(self):
        self.check_num += 1
        if self.check_num % self.check_freq == 0:
            if self.episode_num > 0:
                self.logger.record("train/success_rate", self.success_num / self.episode_num)
                self.logger.record("train/collision_rate", self.collision_num / self.episode_num)
                self.logger.record("train/timeout_rate", self.timeout_num / self.episode_num)
                if self.success_num / self.episode_num > self.best_success_rate:
                    self.best_success_rate = self.success_num / self.episode_num
                    model_path = self._checkpoint_path(extension="zip")
                    self.model.save(model_path)
                    self.logger.info(f"Best success rate: {self.best_success_rate}, save to {model_path}")
                    if self.model.get_vec_normalize_env() is not None:
                        # Save the VecNormalize statistics
                        vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                        self.model.get_vec_normalize_env().save(vec_normalize_path)

            else:
                self.logger.record("train/success_rate", 0)
                self.logger.record("train/collision_rate", 0)
                self.logger.record("train/timeout_rate", 0)
            self.total_episode_reward = 0
            self.episode_num = 0
            self.success_num = 0
            self.collision_num = 0
            self.timeout_num = 0
            self.check_num = 0
    def _on_step(self) -> bool:
        # 获取当前环境的reward和info
        for i, done in enumerate(self.locals['dones']):
            if done:
                if self.locals['infos'][i]['goal_reached']:
                    self.success_num += 1
                elif self.locals['infos'][i]['collision']:
                    self.collision_num += 1
                else:
                    self.timeout_num += 1
                self.episode_num += 1
        return True

class SucessRateCallback(BaseCallback):
    def __init__(self, 
                 check_freq=100, 
                 save_path: str = "checkpoints",
                 name_prefix: str = "best_sr",
                 queue_size=100,
                 verbose=0):
        super(SucessRateCallback, self).__init__(verbose)

        self.episode_num = 0
        self.success_num = 0
        self.collision_num = 0
        self.timeout_num = 0
        self.check_freq = check_freq
        self.check_num = 0
        self.best_mean_reward = -np.inf
        self.best_success_rate = 0.5
        self.save_path = save_path
        self.name_prefix = name_prefix
        # 维护最近100次episode的结果
        if queue_size < 100:
            queue_size = 100
        self.episode_results = deque(maxlen=queue_size)
        self.episode_rewards = deque(maxlen=queue_size)
        
    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_rollout_end(self):
        self.check_num += 1
        if self.check_num % self.check_freq == 0:
            # 计算最近100次episode的成功率
            if len(self.episode_results) > 0:
                recent_success = sum(1 for result in self.episode_results if result == 'success')
                recent_collision = sum(1 for result in self.episode_results if result == 'collision')
                recent_timeout = sum(1 for result in self.episode_results if result == 'timeout')
                total_recent = len(self.episode_results)
                mean_recent_reward = np.mean(self.episode_rewards)
                success_rate = recent_success / total_recent
                collision_rate = recent_collision / total_recent
                timeout_rate = recent_timeout / total_recent

                self.logger.record("val/success_rate", success_rate)
                self.logger.record("val/collision_rate", collision_rate)
                self.logger.record("val/timeout_rate", timeout_rate)
                self.logger.record("val/mean_recent_reward", mean_recent_reward)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    model_path = self._checkpoint_path(extension="zip")
                    self.model.save(model_path)
                    self.logger.info(f"Best success rate: {self.best_success_rate}, save to {model_path}")
                    if self.model.get_vec_normalize_env() is not None:
                        # Save the VecNormalize statistics
                        vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                        self.model.get_vec_normalize_env().save(vec_normalize_path)

            else:
                self.logger.record("val/success_rate", 0)
                self.logger.record("val/collision_rate", 0)
                self.logger.record("val/timeout_rate", 0)
                self.logger.record("val/mean_recent_reward", 0)
                
    def _on_step(self) -> bool:
        # 获取当前环境的reward和info
        for i, done in enumerate(self.locals['dones']):
            if done:
                # 将episode结果添加到deque中
                if self.locals['infos'][i]['goal_reached']:
                    self.episode_results.append('success')
                elif self.locals['infos'][i]['collision']:
                    self.episode_results.append('collision')
                else:
                    self.episode_results.append('timeout')
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                else:
                    self.episode_rewards.append(0)
        return True


class CourseWithSuccessRateCallback(SucessRateCallback):
    def __init__(self, 
                 check_freq=100, 
                 save_path: str = "checkpoints",
                 name_prefix: str = "sb3_train",
                 queue_size=100,
                 verbose=0,
                 success_rate_threshold=0.8,
                 min_episodes_for_curriculum=20,
                 initial_lr=2e-5,
                 min_lr=5e-6,
                 total_timesteps_per_course=1000000,
                 warmup_steps=50000,
                 warmup_start_lr_factor=0.1,
                 course_num=1
                 ):
        super(CourseWithSuccessRateCallback, self).__init__(check_freq, save_path, name_prefix, verbose)
        
        # Curriculum learning parameters
        self.success_rate_threshold = success_rate_threshold
        self.min_episodes_for_curriculum = min_episodes_for_curriculum
        self.current_course = 0
        self.course_switched = False
        
        # Learning rate parameters for flexible scheduling
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_timesteps_per_course = total_timesteps_per_course
        self.mini_timesteps_per_course = total_timesteps_per_course * 0.4
        self.curriculum_start_timesteps = 0  # Track when each curriculum starts
        
        # Warmup parameters
        self.warmup_steps = warmup_steps
        self.warmup_start_lr_factor = warmup_start_lr_factor  # Start from this factor of initial_lr
        self.warmup_start_lr = self.initial_lr * self.warmup_start_lr_factor
        
        # Learning rate scheduling state
        self.current_phase = "warmup"  # "warmup", "learning", "decay"
        self.phase_start_timesteps = 0
        
        # Initialize lr_schedule after model is set
        self._lr_schedule_initialized = False

        # 课程列表，每个课程包含成功率阈值、无人机数量、电动飞行器数量

        self.course_num = course_num

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{self.current_course}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")
    
    def _check_curriculum_switch(self):
        """
        Check if curriculum should be switched based on success rate threshold
        and minimum episode requirements
        """
        steps_in_current_course = self.num_timesteps - self.curriculum_start_timesteps
        if steps_in_current_course <= self.mini_timesteps_per_course:
            return False
        if self.current_course >= (self.course_num - 1):
            return False
        if len(self.episode_results) >= self.min_episodes_for_curriculum:
            recent_success = sum(1 for result in self.episode_results if result == 'success')
            total_recent = len(self.episode_results)
            current_success_rate = recent_success / total_recent
            
            if current_success_rate >= self.success_rate_threshold:
                self._switch_to_next_course()
                return True
        steps_in_current_course = self.num_timesteps - self.curriculum_start_timesteps
        if steps_in_current_course > self.total_timesteps_per_course:
            return True
        return False
    
    def _switch_to_next_course(self):
        """
        Switch to next course and reset episode tracking
        """
        self.current_course += 1
        if self.current_course >= self.course_num:
            self.current_course = self.course_num - 1
        self.course_switched = True
        # base_env = self.model.get_env().unwrapped.envs[0].env
        # # if hasattr(base_env, 'state'):
        # #     base_env.state.uav_num = self.course_list[self.current_course]['uav_num']
        # #     base_env.state.evtol_num = self.course_list[self.current_course]['evtol_num']
        # # Clear episode results for new course
        base_env = self.model.get_env().unwrapped
        base_env.set_course(self.current_course)
        self.episode_results.clear()
        self.episode_rewards.clear()
        
        # Reset learning rate for new curriculum
        self._reset_learning_rate()
        
        # Log curriculum switch
        self.logger.info(f"Curriculum switched to course {self.current_course}")
        self.logger.record("train/current_course", self.current_course)
        
        # Reset episode counters for new course
        self.best_success_rate = 0.5
        self.episode_num = 0
        self.success_num = 0
        self.collision_num = 0
        self.timeout_num = 0
        
    def _reset_learning_rate(self):
        """
        Reset learning rate and start warmup phase for new curriculum
        """
        try:
            # Reset learning rate scheduling state
            self.current_phase = "warmup"
            self.phase_start_timesteps = self.num_timesteps
            self.curriculum_start_timesteps = self.num_timesteps
            
            # Update the lr_schedule to return current learning rate
            self._update_lr_schedule(self.warmup_start_lr)
            
            self.logger.info(f"Learning rate reset to {self.warmup_start_lr} for new curriculum (warmup phase)")
            self.logger.record("curriculum/learning_rate", self.warmup_start_lr)
            self.logger.record("curriculum/lr_phase", "warmup")
                
        except Exception as e:
            self.logger.warn(f"Failed to reset learning rate: {e}")
            
    def _update_learning_rate(self):
        """
        Update learning rate based on current phase (warmup -> learning -> decay)
        """
        try:
            # Calculate steps in current course
            steps_in_current_course = self.num_timesteps - self.curriculum_start_timesteps
            steps_in_current_phase = self.num_timesteps - self.phase_start_timesteps
            
            current_lr = None
            
            # Phase 1: Warmup - linear increase from warmup_start_lr to initial_lr
            if self.current_phase == "warmup":
                if steps_in_current_phase >= self.warmup_steps:
                    # Warmup complete, switch to learning phase
                    self.current_phase = "learning"
                    self.phase_start_timesteps = self.num_timesteps
                    current_lr = self.initial_lr
                    self.logger.info(f"Warmup complete, switching to learning phase with LR: {current_lr}")
                    self.logger.record("curriculum/lr_phase", "learning")
                else:
                    # Linear warmup
                    warmup_progress = steps_in_current_phase / self.warmup_steps
                    current_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * warmup_progress
            
            # Phase 2: Learning - maintain initial_lr
            elif self.current_phase == "learning":
                # Check if we should start decay phase
                # Start decay after warmup + some learning time
                decay_start_steps = self.warmup_steps + (self.total_timesteps_per_course - self.warmup_steps) * 0.3  # Start decay at 30% of remaining time
                
                if steps_in_current_course >= decay_start_steps:
                    # Start decay phase
                    self.current_phase = "decay"
                    self.phase_start_timesteps = self.num_timesteps
                    self.logger.info(f"Starting learning rate decay phase")
                    self.logger.record("curriculum/lr_phase", "decay")
                
                current_lr = self.initial_lr
            
            # Phase 3: Decay - linear decrease from initial_lr to min_lr
            elif self.current_phase == "decay":
                # Calculate decay progress within decay phase
                decay_phase_steps = self.total_timesteps_per_course - self.warmup_steps - (self.total_timesteps_per_course - self.warmup_steps) * 0.3
                if decay_phase_steps > 0:
                    decay_progress = min((steps_in_current_course - (self.warmup_steps + (self.total_timesteps_per_course - self.warmup_steps) * 0.3)) / decay_phase_steps, 1.0)
                    current_lr = self.initial_lr + (self.min_lr - self.initial_lr) * decay_progress
                else:
                    current_lr = self.min_lr
            
            # Update the lr_schedule to return current learning rate
            if current_lr is not None:
                self._update_lr_schedule(current_lr)
                
                # Log learning rate information
                self.logger.record("curriculum/learning_rate", current_lr)
                self.logger.record("curriculum/lr_phase", self.current_phase)
                self.logger.record("curriculum/steps_in_current_course", steps_in_current_course)
                self.logger.record("curriculum/steps_in_current_phase", steps_in_current_phase)
                
                # Log phase-specific information
                if self.current_phase == "warmup":
                    warmup_progress = steps_in_current_phase / self.warmup_steps
                    self.logger.record("curriculum/warmup_progress", warmup_progress)
                elif self.current_phase == "decay":
                    decay_phase_steps = self.total_timesteps_per_course - self.warmup_steps - (self.total_timesteps_per_course - self.warmup_steps) * 0.3
                    if decay_phase_steps > 0:
                        decay_progress = min((steps_in_current_course - (self.warmup_steps + (self.total_timesteps_per_course - self.warmup_steps) * 0.3)) / decay_phase_steps, 1.0)
                        self.logger.record("curriculum/decay_progress", decay_progress)
                
        except Exception as e:
            self.logger.warn(f"Failed to update learning rate: {e}")
            
    def _update_lr_schedule(self, current_lr):
        """
        Update the model's lr_schedule to return the current learning rate
        This ensures that when PPO calls _update_learning_rate, it gets our curriculum-based LR
        
        Args:
            current_lr: The learning rate value to set (already calculated)
        """
        try:
            # Create a constant function that always returns the current learning rate
            def curriculum_lr_schedule(progress_remaining):
                return current_lr
            
            # Update the model's lr_schedule
            if hasattr(self.model, 'lr_schedule'):
                self.model.lr_schedule = curriculum_lr_schedule
                self.logger.debug(f"Updated lr_schedule to return constant LR: {current_lr}")
            else:
                self.logger.warn("Model does not have lr_schedule attribute")
                
        except Exception as e:
            self.logger.warn(f"Failed to update lr_schedule: {e}")
            
    def _on_rollout_end(self):
        # Check for curriculum switch before regular evaluation
        if self._check_curriculum_switch():
            # If curriculum switched, skip regular evaluation this time
            # as we need to collect new episode data
            return
            
        # Update learning rate based on current course progress
        self._update_learning_rate()
            
        # Regular evaluation logic from parent class
        super()._on_rollout_end()
        
        # Log current course information
        self.logger.record("curriculum/current_course", self.current_course)
        
    def _on_step(self) -> bool:
        # Call parent method to collect episode data
        result = super()._on_step()

        # Initialize lr_schedule on first step if not done yet
        if not self._lr_schedule_initialized and hasattr(self, 'model') and self.model is not None:
            self._initialize_lr_schedule()
        
        # Reset course_switched flag after one step
        if self.course_switched:
            self.course_switched = False
            
        return result
        
    def _initialize_lr_schedule(self):
        """
        Initialize the lr_schedule when the model is first available
        """
        try:
            if hasattr(self.model, 'lr_schedule'):
                # Set initial warmup learning rate
                self._update_lr_schedule(self.warmup_start_lr)
                self._lr_schedule_initialized = True
                self.logger.info(f"Initialized curriculum lr_schedule with warmup LR: {self.warmup_start_lr}")
            else:
                self.logger.warn("Model does not have lr_schedule attribute")
        except Exception as e:
            self.logger.warn(f"Failed to initialize lr_schedule: {e}")
    

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.best_success_rate = 0.1

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("goal_reached")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True
        if self.num_timesteps < 0.5 * self.model._total_timesteps:
            eval_freq = self.eval_freq*100  # 前半程评估稀疏
        else:
            eval_freq = self.eval_freq   # 后半程评估密集
        if eval_freq > 0 and self.n_calls % eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            import time
            eval_start_time = time.time()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time
            total_steps = sum(episode_lengths)
            eval_fps = total_steps / eval_duration if eval_duration > 0 else float('inf')

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(f"Eval FPS: {eval_fps:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/fps", eval_fps)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self.logger.info(f"New best success rate: {self.best_success_rate}")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_success_rate_model"))
                        self.logger.info(f"Best success rate model saved to {os.path.join(self.best_model_save_path, 'best_success_rate_model')}")
                        if self.model.get_vec_normalize_env() is not None:
                            vec_normalize_path = os.path.join(self.best_model_save_path, "best_success_rate_model_vecnormalize.pkl")
                            self.model.get_vec_normalize_env().save(vec_normalize_path)
                            self.logger.info(f"Best success rate model VecNormalize saved to {vec_normalize_path}")

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.logger.info(f"Best mean reward model saved to {os.path.join(self.best_model_save_path, 'best_model')}")
                    if self.model.get_vec_normalize_env() is not None:
                        vec_normalize_path = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
                        self.model.get_vec_normalize_env().save(vec_normalize_path)
                        self.logger.info(f"Best mean reward model VecNormalize saved to {vec_normalize_path}")
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

class CustomCheckpointCallback(CheckpointCallback):
    """
    Custom checkpoint callback for saving the model slower in the first half of training.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(save_freq=save_freq, 
                         save_path=save_path, 
                         name_prefix=name_prefix, 
                         save_replay_buffer=save_replay_buffer, 
                         save_vecnormalize=save_vecnormalize, 
                         verbose=verbose)
        self.original_save_freq = save_freq
        self.save_freq = save_freq
    def _on_step(self) -> bool:
        if self.num_timesteps < 0.5 * self.model._total_timesteps:
            self.save_freq = self.original_save_freq*10  # 前半程评估稀疏
        else:
            self.save_freq = self.original_save_freq   # 后半程评估密集
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            return super()._on_step()
        return True