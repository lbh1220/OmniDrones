"""
SKRL训练的自定义回调函数
包括success rate监控、TensorBoard和Wandb日志记录
"""

import os
import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

# SKRL imports
from skrl.utils import set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseCallback:
    """基础回调类"""
    
    def __init__(self, name: str = "BaseCallback"):
        self.name = name
        self.timestep = 0
        self.episode_count = 0
        
    def on_training_start(self, trainer, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_timestep_end(self, trainer, **kwargs):
        """每个时间步结束时调用"""
        self.timestep += 1
        
    def on_episode_end(self, trainer, **kwargs):
        """每个episode结束时调用"""
        self.episode_count += 1
        
    def on_training_end(self, trainer, **kwargs):
        """训练结束时调用"""
        pass


class SuccessRateCallback(BaseCallback):
    """
    Success Rate监控回调
    监控成功率、碰撞率和超时率，保存最佳模型
    """
    
    def __init__(
        self,
        check_freq: int = 100,
        save_path: str = "checkpoints",
        queue_size: int = 100,
        name_prefix: str = "best_sr",
        success_threshold: float = 0.8
    ):
        super().__init__(name="SuccessRateCallback")
        
        self.check_freq = check_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.success_threshold = success_threshold
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 维护最近episodes的结果
        if queue_size < 100:
            queue_size = 100
        self.episode_results = deque(maxlen=queue_size)
        self.episode_rewards = deque(maxlen=queue_size)
        
        # 统计信息
        self.best_success_rate = 0.0
        self.best_mean_reward = -np.inf
        self.check_count = 0
        
        # 临时存储当前episode的信息
        self.current_episode_info = {}
        
    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "pt") -> str:
        """获取检查点保存路径"""
        return os.path.join(
            self.save_path, 
            f"{self.name_prefix}_{self.timestep}_steps{checkpoint_type}.{extension}"
        )
    
    def on_timestep_end(self, trainer, **kwargs):
        """检查episode是否结束并收集统计信息"""
        super().on_timestep_end(trainer, **kwargs)
        
        # 从环境获取信息
        if hasattr(trainer, 'env') and hasattr(trainer.env, 'infos'):
            infos = trainer.env.infos
            dones = trainer.env.dones if hasattr(trainer.env, 'dones') else None
            rewards = trainer.env.rewards if hasattr(trainer.env, 'rewards') else None
            
            if dones is not None and infos is not None:
                for i, done in enumerate(dones):
                    if done:
                        info = infos[i] if isinstance(infos, (list, tuple)) else infos
                        
                        # 收集episode结果
                        if isinstance(info, dict):
                            if info.get('goal_reached', False):
                                self.episode_results.append('success')
                            elif info.get('collision', False):
                                self.episode_results.append('collision')
                            else:
                                self.episode_results.append('timeout')
                            
                            # 收集episode奖励
                            episode_reward = info.get('episode_reward', 0)
                            if episode_reward != 0:
                                self.episode_rewards.append(episode_reward)
                            elif rewards is not None:
                                self.episode_rewards.append(float(rewards[i]))
    
    def on_episode_end(self, trainer, **kwargs):
        """Episode结束时的处理"""
        super().on_episode_end(trainer, **kwargs)
        
        self.check_count += 1
        
        # 定期检查并记录统计信息
        if self.check_count % self.check_freq == 0:
            self._log_and_save_stats(trainer)
    
    def _log_and_save_stats(self, trainer):
        """记录统计信息并保存最佳模型"""
        if len(self.episode_results) == 0:
            return
        
        # 计算统计信息
        recent_success = sum(1 for result in self.episode_results if result == 'success')
        recent_collision = sum(1 for result in self.episode_results if result == 'collision') 
        recent_timeout = sum(1 for result in self.episode_results if result == 'timeout')
        total_recent = len(self.episode_results)
        
        success_rate = recent_success / total_recent
        collision_rate = recent_collision / total_recent
        timeout_rate = recent_timeout / total_recent
        
        mean_recent_reward = np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
        
        # 记录到trainer的日志中
        if hasattr(trainer, 'write_interval') and hasattr(trainer, 'writer'):
            trainer.writer.add_scalar("Success_Rate/success_rate", success_rate, self.timestep)
            trainer.writer.add_scalar("Success_Rate/collision_rate", collision_rate, self.timestep)
            trainer.writer.add_scalar("Success_Rate/timeout_rate", timeout_rate, self.timestep)
            trainer.writer.add_scalar("Success_Rate/mean_recent_reward", mean_recent_reward, self.timestep)
        
        # 打印统计信息
        print(f"Timestep {self.timestep}:")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Collision Rate: {collision_rate:.3f}")
        print(f"  Timeout Rate: {timeout_rate:.3f}")
        print(f"  Mean Recent Reward: {mean_recent_reward:.3f}")
        
        # 保存最佳模型
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            model_path = self._checkpoint_path("_best_success", "pt")
            
            try:
                # 保存模型
                if hasattr(trainer, 'agents'):
                    if isinstance(trainer.agents, (list, tuple)):
                        agent = trainer.agents[0]
                    else:
                        agent = trainer.agents
                    
                    torch.save({
                        'policy_state_dict': agent.policy.state_dict(),
                        'value_state_dict': agent.value.state_dict() if hasattr(agent, 'value') else None,
                        'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None,
                        'timestep': self.timestep,
                        'success_rate': success_rate,
                        'mean_reward': mean_recent_reward
                    }, model_path)
                    
                    print(f"  New best success rate! Model saved to: {model_path}")
                    
            except Exception as e:
                print(f"  Warning: Failed to save model: {e}")
        
        if mean_recent_reward > self.best_mean_reward:
            self.best_mean_reward = mean_recent_reward
            model_path = self._checkpoint_path("_best_reward", "pt")
            
            try:
                if hasattr(trainer, 'agents'):
                    if isinstance(trainer.agents, (list, tuple)):
                        agent = trainer.agents[0]
                    else:
                        agent = trainer.agents
                    
                    torch.save({
                        'policy_state_dict': agent.policy.state_dict(),
                        'value_state_dict': agent.value.state_dict() if hasattr(agent, 'value') else None,
                        'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None,
                        'timestep': self.timestep,
                        'success_rate': success_rate,
                        'mean_reward': mean_recent_reward
                    }, model_path)
                    
                    print(f"  New best mean reward! Model saved to: {model_path}")
                    
            except Exception as e:
                print(f"  Warning: Failed to save model: {e}")


class TensorboardLogger(BaseCallback):
    """TensorBoard日志记录回调"""
    
    def __init__(self, log_dir: str = "tensorboard_logs"):
        super().__init__(name="TensorboardLogger")
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        print(f"TensorBoard日志将保存到: {log_dir}")
        
    def on_timestep_end(self, trainer, **kwargs):
        """记录训练指标"""
        super().on_timestep_end(trainer, **kwargs)
        
        # 如果trainer有write_interval属性，使用它来控制写入频率
        if hasattr(trainer, 'write_interval'):
            if self.timestep % trainer.write_interval != 0:
                return
        
        # 记录基本训练指标
        if hasattr(trainer, 'agents'):
            if isinstance(trainer.agents, (list, tuple)):
                agent = trainer.agents[0]
            else:
                agent = trainer.agents
                
            # 记录损失
            if hasattr(agent, 'tracking_data'):
                for key, value in agent.tracking_data.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        scalar_value = float(value)
                        self.writer.add_scalar(f"Training/{key}", scalar_value, self.timestep)
            
            # 记录学习率
            if hasattr(agent, 'scheduler') and hasattr(agent.scheduler, 'get_last_lr'):
                lr = agent.scheduler.get_last_lr()[0]
                self.writer.add_scalar("Training/learning_rate", lr, self.timestep)
        
        # 记录环境指标
        if hasattr(trainer, 'env'):
            if hasattr(trainer.env, 'rewards'):
                rewards = trainer.env.rewards
                if rewards is not None:
                    mean_reward = float(torch.mean(rewards))
                    self.writer.add_scalar("Environment/mean_reward", mean_reward, self.timestep)
        
        # 强制写入
        self.writer.flush()
    
    def on_training_end(self, trainer, **kwargs):
        """训练结束时关闭writer"""
        super().on_training_end(trainer, **kwargs)
        self.writer.close()
        print("TensorBoard日志记录完成")


class WandbLogger(BaseCallback):
    """Weights & Biases日志记录回调"""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        dir: Optional[str] = None,
        tags: Optional[list] = None
    ):
        super().__init__(name="WandbLogger")
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not available. Install with: pip install wandb")
        
        # 初始化wandb
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            dir=dir,
            tags=tags
        )
        
        print(f"Wandb初始化完成: {wandb.run.url}")
        
    def on_timestep_end(self, trainer, **kwargs):
        """记录训练指标到wandb"""
        super().on_timestep_end(trainer, **kwargs)
        
        # 控制记录频率
        if hasattr(trainer, 'write_interval'):
            if self.timestep % trainer.write_interval != 0:
                return
        
        log_dict = {"timestep": self.timestep}
        
        # 记录训练指标
        if hasattr(trainer, 'agents'):
            if isinstance(trainer.agents, (list, tuple)):
                agent = trainer.agents[0]
            else:
                agent = trainer.agents
                
            # 记录损失和指标
            if hasattr(agent, 'tracking_data'):
                for key, value in agent.tracking_data.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        log_dict[f"training/{key}"] = float(value)
            
            # 记录学习率
            if hasattr(agent, 'scheduler') and hasattr(agent.scheduler, 'get_last_lr'):
                log_dict["training/learning_rate"] = agent.scheduler.get_last_lr()[0]
        
        # 记录环境指标
        if hasattr(trainer, 'env'):
            if hasattr(trainer.env, 'rewards'):
                rewards = trainer.env.rewards
                if rewards is not None:
                    log_dict["environment/mean_reward"] = float(torch.mean(rewards))
        
        # 记录到wandb
        if len(log_dict) > 1:  # 确保有实际数据要记录
            wandb.log(log_dict, step=self.timestep)
    
    def on_training_end(self, trainer, **kwargs):
        """训练结束时完成wandb记录"""
        super().on_training_end(trainer, **kwargs)
        wandb.finish()
        print("Wandb日志记录完成")


class EpisodeStatsCallback(BaseCallback):
    """Episode统计信息回调"""
    
    def __init__(self, log_interval: int = 100):
        super().__init__(name="EpisodeStatsCallback")
        
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_start_time = time.time()
        
    def on_episode_end(self, trainer, **kwargs):
        """收集episode统计信息"""
        super().on_episode_end(trainer, **kwargs)
        
        # 获取episode信息
        if hasattr(trainer, 'env'):
            if hasattr(trainer.env, 'infos'):
                infos = trainer.env.infos
                if isinstance(infos, (list, tuple)) and len(infos) > 0:
                    info = infos[0]
                else:
                    info = infos
                
                if isinstance(info, dict):
                    episode_reward = info.get('episode_reward', 0)
                    episode_length = info.get('episode_length', 0)
                    
                    if episode_reward != 0:
                        self.episode_rewards.append(episode_reward)
                    if episode_length != 0:
                        self.episode_lengths.append(episode_length)
        
        # 定期打印统计信息
        if self.episode_count % self.log_interval == 0 and len(self.episode_rewards) > 0:
            current_time = time.time()
            elapsed_time = current_time - self.episode_start_time
            
            mean_reward = np.mean(self.episode_rewards[-self.log_interval:])
            mean_length = np.mean(self.episode_lengths[-self.log_interval:]) if self.episode_lengths else 0
            
            print(f"Episode {self.episode_count}:")
            print(f"  Mean Reward (last {self.log_interval}): {mean_reward:.3f}")
            print(f"  Mean Length (last {self.log_interval}): {mean_length:.1f}")
            print(f"  Elapsed Time: {elapsed_time:.1f}s")
            print(f"  Episodes per second: {self.log_interval / elapsed_time:.2f}")
            
            self.episode_start_time = current_time


class ModelCheckpointCallback(BaseCallback):
    """模型定期保存回调"""
    
    def __init__(
        self,
        save_interval: int = 10000,
        save_path: str = "checkpoints",
        name_prefix: str = "model_checkpoint"
    ):
        super().__init__(name="ModelCheckpointCallback")
        
        self.save_interval = save_interval
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        os.makedirs(save_path, exist_ok=True)
        
    def on_timestep_end(self, trainer, **kwargs):
        """定期保存模型"""
        super().on_timestep_end(trainer, **kwargs)
        
        if self.timestep % self.save_interval == 0:
            model_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.timestep}_steps.pt"
            )
            
            try:
                if hasattr(trainer, 'agents'):
                    if isinstance(trainer.agents, (list, tuple)):
                        agent = trainer.agents[0]
                    else:
                        agent = trainer.agents
                    
                    torch.save({
                        'policy_state_dict': agent.policy.state_dict(),
                        'value_state_dict': agent.value.state_dict() if hasattr(agent, 'value') else None,
                        'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None,
                        'timestep': self.timestep
                    }, model_path)
                    
                    print(f"模型检查点已保存: {model_path}")
                    
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
