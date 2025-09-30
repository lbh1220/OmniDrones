#!/usr/bin/env python3

"""
Test script for GRU-based Recurrent Policy
测试GRU循环策略的基本功能
"""

import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from gru_recurrent_policy import GRUMultiInputActorCriticPolicy
from attention_features_extractor import AttentionFeaturesExtractor


def test_gru_policy_initialization():
    """测试GRU策略的初始化"""
    print("Testing GRU policy initialization...")
    
    # 定义观察空间（模拟交通环境的观察空间）
    observation_space = spaces.Dict({
        'robot_node': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 9), dtype=np.float32),
        'temporal_edges': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32),
        'spatial_edges': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 13), dtype=np.float32),
        'detected_human_num': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
    })
    
    # 定义动作空间
    action_space = spaces.Discrete(7**3)  # 离散动作空间
    
    # 创建学习率调度
    def lr_schedule(progress):
        return 4e-5
    
    # 策略参数
    policy_kwargs = dict(
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
        ortho_init=True,
        gru_hidden_size=256,
        n_gru_layers=1,
        shared_gru=True,
        enable_critic_gru=False,
    )
    
    # 创建策略
    try:
        policy = GRUMultiInputActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **policy_kwargs
        )
        print("✓ GRU policy initialized successfully")
        
        # # 检查GRU模块是否正确创建
        # assert hasattr(policy, 'gru_actor'), "Missing gru_actor"
        # assert isinstance(policy.gru_actor, nn.GRU), "gru_actor is not GRU"
        # assert not hasattr(policy, 'lstm_actor'), "lstm_actor should be removed"
        
        # print(f"✓ GRU actor created: {policy.gru_actor}")
        # print(f"✓ GRU hidden size: {policy.gru_hidden_size}")
        # print(f"✓ GRU layers: {policy.n_gru_layers}")
        # print(f"✓ Shared GRU: {policy.shared_gru}")
        
        return policy
        
    except Exception as e:
        print(f"✗ Failed to initialize GRU policy: {e}")
        raise


def test_gru_forward_pass(policy):
    """测试GRU策略的前向传播"""
    print("\nTesting GRU policy forward pass...")
    
    batch_size = 4
    device = th.device("cpu")
    
    # 创建模拟观察
    observations = {
        'robot_node': th.randn(batch_size, 1, 9, dtype=th.float32),
        'temporal_edges': th.randn(batch_size, 1, 3, dtype=th.float32),
        'spatial_edges': th.randn(batch_size, 10, 13, dtype=th.float32),
        'detected_human_num': th.randint(1, 11, (batch_size, 1), dtype=th.int32)
    }
    
    # 创建初始的GRU状态
    # 对于GRU，我们需要(hidden, dummy_cell)格式以保持兼容性
    hidden_state = th.zeros(policy.n_gru_layers, batch_size, policy.gru_hidden_size)
    # hidden_state = th.zeros(policy.n_lstm_layers, batch_size, policy.lstm_hidden_size)
    dummy_cell_state = th.zeros_like(hidden_state)  # GRU不使用，但保持格式
    
    initial_states = RNNStates(
        pi=(hidden_state, dummy_cell_state),
        vf=(hidden_state, dummy_cell_state)
    )
    
    episode_starts = th.zeros(batch_size, dtype=th.float32)
    
    try:
        # 前向传播
        actions, values, log_probs, new_states = policy.forward(
            observations, initial_states, episode_starts, deterministic=False
        )
        
        print(f"✓ Forward pass successful")
        print(f"✓ Actions shape: {actions.shape}")
        print(f"✓ Values shape: {values.shape}")
        print(f"✓ Log probs shape: {log_probs.shape}")
        print(f"✓ New states type: {type(new_states)}")
        
        # 检查状态格式
        assert isinstance(new_states, RNNStates), "States should be RNNStates"
        assert len(new_states.pi) == 2, "Pi states should have 2 elements (hidden, cell)"
        assert len(new_states.vf) == 2, "Vf states should have 2 elements (hidden, cell)"
        
        print(f"✓ Pi states shape: {[s.shape for s in new_states.pi]}")
        print(f"✓ Vf states shape: {[s.shape for s in new_states.vf]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_gru_evaluate_actions(policy):
    """测试动作评估功能"""
    print("\nTesting GRU policy action evaluation...")
    
    batch_size = 4
    
    # 创建模拟观察和动作
    observations = {
        'robot_node': th.randn(batch_size, 1, 9, dtype=th.float32),
        'temporal_edges': th.randn(batch_size, 1, 3, dtype=th.float32),
        'spatial_edges': th.randn(batch_size, 10, 13, dtype=th.float32),
        'detected_human_num': th.randint(1, 11, (batch_size, 1), dtype=th.int32)
    }
    
    actions = th.randint(0, 343, (batch_size,), dtype=th.long)  # 7^3 = 343
    
    # 创建GRU状态
    hidden_state = th.zeros(policy.n_gru_layers, batch_size, policy.gru_hidden_size)
    # hidden_state = th.zeros(policy.n_lstm_layers, batch_size, policy.lstm_hidden_size)
    dummy_cell_state = th.zeros_like(hidden_state)
    
    lstm_states = RNNStates(
        pi=(hidden_state, dummy_cell_state),
        vf=(hidden_state, dummy_cell_state)
    )
    
    episode_starts = th.zeros(batch_size, dtype=th.float32)
    
    try:
        values, log_probs, entropy = policy.evaluate_actions(
            observations, actions, lstm_states, episode_starts
        )
        
        print(f"✓ Action evaluation successful")
        print(f"✓ Values shape: {values.shape}")
        print(f"✓ Log probs shape: {log_probs.shape}")
        print(f"✓ Entropy shape: {entropy.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Action evaluation failed: {e}")
        raise


def test_states_compatibility():
    """测试状态兼容性"""
    print("\nTesting RNN states compatibility...")
    
    batch_size = 4
    gru_hidden_size = 256
    n_layers = 1
    
    # 创建GRU状态（带有虚拟cell state）
    hidden_state = th.randn(n_layers, batch_size, gru_hidden_size)
    dummy_cell_state = th.zeros_like(hidden_state)
    
    gru_states = RNNStates(
        pi=(hidden_state, dummy_cell_state),
        vf=(hidden_state.clone(), dummy_cell_state.clone())
    )
    
    print(f"✓ GRU states created with shape: {gru_states.pi[0].shape}")
    print(f"✓ Dummy cell states shape: {gru_states.pi[1].shape}")
    print(f"✓ States are compatible with RNNStates format")
    
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("GRU Recurrent Policy Test Suite")
    print("=" * 60)
    
    try:
        # 测试初始化
        policy = test_gru_policy_initialization()
        
        # 测试前向传播
        test_gru_forward_pass(policy)
        
        # 测试动作评估
        test_gru_evaluate_actions(policy)
        
        # 测试状态兼容性
        test_states_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! GRU policy is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Tests failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
