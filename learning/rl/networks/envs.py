import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize
from rl.vec_env.running_mean_std import RunningMeanStd
import json

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets, config=None, envNum=1, ax=None, test_case=-1, is_train=True):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.configure(config)

        envSeed = seed + rank if seed is not None else None
        # environment.render_axis = ax
        env.thisSeed = envSeed
        env.nenv = envNum

        env.phase = 'train' if is_train else 'test'
        # if envNum > 1:
        #     env.phase = 'train'
        # else:
        #     env.phase = 'test'

        if ax:
            env.render_axis = ax
            if test_case >= 0:
                env.test_case = test_case
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        # if log_dir is not None:
        env = bench.Monitor(
            env,
            None,
            allow_early_resets=allow_early_resets)
        print(env)

        if isinstance(env.observation_space, Box):
            if is_atari:
                if len(env.observation_space.shape) == 3:
                    env = wrap_deepmind(env)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example.")

            # If the input has shape (W,H,3), wrap for PyTorch convolutions

            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  config=None,
                  ax=None, test_case=-1, wrap_pytorch=True, pretext_wrapper=False, is_train=True):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, config=config,
                 envNum=num_processes, ax=ax, test_case=test_case, is_train=is_train)
        for i in range(num_processes)
    ]
    test = False if len(envs) > 1 else True

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
        # envs = ShmemVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)
    # for collect data in supervised learning, we don't need to wrap pytorch
    if wrap_pytorch:
        if hasattr(config.env, 'use_obs_norm'):
            if config.env.use_obs_norm or config.env.use_reward_norm:
                # 使用新的normalize wrapper
                envs = VecPyTorchNormalize(
                    envs, 
                    device,
                    ob=config.env.use_obs_norm,
                    ret=config.env.use_reward_norm,
                    clipob=config.env.norm_clip,
                    cliprew=config.env.norm_clip,
                    gamma=gamma,
                    epsilon=config.env.norm_epsilon,
                    training=is_train  # 训练模式下才更新统计量
                )
        else:
            envs = VecPyTorch(envs, device)
    if pretext_wrapper:
        if gamma is None:
            envs = VecPretextNormalize(envs, ret=False, ob=False, config=config, test=test)
        else:
            envs = VecPretextNormalize(envs, gamma=gamma, ob=False, ret=False, config=config, test=test)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif isinstance(envs.observation_space, Box):
        if len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def render_traj(self, path, episode_num):
        if self.venv.num_envs == 1:
            return self.venv.envs[0].env.render_traj(path, episode_num)
        else:
            for i, curr_env in enumerate(self.venv.envs):
                curr_env.env.render_traj(path, str(episode_num) + '.' + str(i))


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class VecPyTorchNormalize(VecPyTorch):
    """
    在VecPyTorch基础上增加observation和reward的标准化功能
    先进行归一化，再转换为torch tensor
    """
    def __init__(self, venv, device, ob=True, ret=True, 
                 clipob=None, cliprew=None, gamma=0.99, epsilon=1e-8,
                 training=True):
        super().__init__(venv, device)
        self.ob_rms = None
        self.ret_rms = None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs, np.float32)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.ob = ob
        self.ret_norm = ret
        
        # 定义不需要归一化的observation keys
        self.skip_norm_keys = {
            'detected_human_num',
            'detected_evtol_num', 
            'detected_uav_num',
            'agent_types'
        }
        
        # 初始化rms统计对象
        if self.ob:
            if isinstance(self.observation_space, Dict):
                self.ob_rms = {}
                for key, space in self.observation_space.spaces.items():
                    # 只为需要归一化的observation创建rms对象
                    if isinstance(space, Box) and key not in self.skip_norm_keys:
                        self.ob_rms[key] = RunningMeanStd(shape=space.shape)
            elif isinstance(self.observation_space, Box):
                self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
                
        if self.ret_norm:
            self.ret_rms = RunningMeanStd(shape=())

    def _obfilt(self, obs):
        """对numpy格式的observation进行归一化，跳过不需要归一化的keys"""
        if self.ob_rms is None:
            return obs
            
        if isinstance(obs, dict):
            obs_filtered = {}
            for key in obs:
                if key in self.ob_rms:  # 只处理需要归一化的observation
                    if self.training:
                        self.ob_rms[key].update(obs[key])
                    normalized = (obs[key] - self.ob_rms[key].mean) / \
                               np.sqrt(self.ob_rms[key].var + self.epsilon)
                    if self.clipob is not None:
                        normalized = np.clip(normalized, -self.clipob, self.clipob)
                    obs_filtered[key] = normalized
                else:
                    obs_filtered[key] = obs[key]  # 不需要归一化的直接复制
            return obs_filtered
        else:
            if self.training:
                self.ob_rms.update(obs)
            normalized = (obs - self.ob_rms.mean) / \
                        np.sqrt(self.ob_rms.var + self.epsilon)
            if self.clipob is not None:
                normalized = np.clip(normalized, -self.clipob, self.clipob)
            return normalized

    def step_wait(self):
        """重写step_wait，先归一化再转tensor"""
        obs, rews, done, info = self.venv.step_wait()
        
        # normalize rewards (in numpy)
        if self.ret_norm:
            self.ret = self.ret * self.gamma + rews
            if self.training:
                self.ret_rms.update(self.ret)
                # 不是train的时候，就不要修改输出的reward的值了
                normalized_rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)
                if self.cliprew is not None:
                    normalized_rews = np.clip(normalized_rews, -self.cliprew, self.cliprew)
                rews = normalized_rews
                self.ret[done] = 0.
            
        # normalize observations (in numpy)
        obs = self._obfilt(obs)
        
        # 转换为torch tensor (继承VecPyTorch的功能)
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        rews = torch.from_numpy(rews).unsqueeze(dim=1).float()
        
        return obs, rews, done, info

    def reset(self):
        """重写reset，先归一化再转tensor"""
        self.ret = np.zeros(self.num_envs,np.float32)
        obs = self.venv.reset()
        obs = self._obfilt(obs)
        
        # 转换为torch tensor
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            
        return obs

    def save_running_stats(self, save_dir, save_name='normalize_stats.json'):
        """保存归一化统计量"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        stats = {}
        if self.ob_rms is not None:
            if isinstance(self.ob_rms, dict):
                stats['ob_rms'] = {}
                for key, rms in self.ob_rms.items():
                    stats['ob_rms'][key] = {
                        'mean': rms.mean.tolist(),
                        'var': rms.var.tolist(),
                        'count': float(rms.count)
                    }
            else:
                stats['ob_rms'] = {
                    'mean': self.ob_rms.mean.tolist(),
                    'var': self.ob_rms.var.tolist(),
                    'count': float(self.ob_rms.count)
                }
                
        if self.ret_rms is not None:
            stats['ret_rms'] = {
                'mean': self.ret_rms.mean.tolist(),
                'var': self.ret_rms.var.tolist(),
                'count': float(self.ret_rms.count)
            }
            
        with open(os.path.join(save_dir, save_name), 'w') as f:
            json.dump(stats, f)
            
    def load_running_stats(self, save_dir, load_name='normalize_stats.json'):
        """加载归一化统计量"""
        load_path = os.path.join(save_dir, load_name)
        with open(load_path, 'r') as f:
            stats = json.load(f)
            
        if 'ob_rms' in stats and self.ob_rms is not None:
            if isinstance(self.ob_rms, dict):
                for key, rms in self.ob_rms.items():
                    if key in stats['ob_rms']:
                        rms.mean = np.array(stats['ob_rms'][key]['mean'],np.float32)
                        rms.var = np.array(stats['ob_rms'][key]['var'],np.float32)
                        rms.count = stats['ob_rms'][key]['count']
            else:
                self.ob_rms.mean = np.array(stats['ob_rms']['mean'],np.float32)
                self.ob_rms.var = np.array(stats['ob_rms']['var'],np.float32)
                self.ob_rms.count = stats['ob_rms']['count']
                
        if 'ret_rms' in stats and self.ret_rms is not None:
            self.ret_rms.mean = np.array(stats['ret_rms']['mean'],np.float32)
            self.ret_rms.var = np.array(stats['ret_rms']['var'],np.float32)
            self.ret_rms.count = stats['ret_rms']['count']

    def load_running_stats_from_path(self, load_path):
        """从路径中加载归一化统计量"""
        with open(load_path, 'r') as f:
            stats = json.load(f)

        if 'ob_rms' in stats and self.ob_rms is not None:
            if isinstance(self.ob_rms, dict):
                for key, rms in self.ob_rms.items():
                    if key in stats['ob_rms']:
                        rms.mean = np.array(stats['ob_rms'][key]['mean'],np.float32)
                        rms.var = np.array(stats['ob_rms'][key]['var'],np.float32)
                        rms.count = stats['ob_rms'][key]['count']
            else:
                self.ob_rms.mean = np.array(stats['ob_rms']['mean'],np.float32)
                self.ob_rms.var = np.array(stats['ob_rms']['var'],np.float32)
                self.ob_rms.count = stats['ob_rms']['count']
                
        if 'ret_rms' in stats and self.ret_rms is not None:
            self.ret_rms.mean = np.array(stats['ret_rms']['mean'],np.float32)
            self.ret_rms.var = np.array(stats['ret_rms']['var'],np.float32)
            self.ret_rms.count = stats['ret_rms']['count']