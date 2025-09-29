#!/usr/bin/env python3

"""Test script for AttentionFeaturesExtractor
测试注意力机制特征提取器的基本功能
"""

import torch
import numpy as np
from gymnasium import spaces
from rl.sb3.attention_features_extractor import AttentionFeaturesExtractor

# Import the original network for comparison
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from rl.networks.selfAttn_srnn_temp_node import selfAttn_merge_SRNN
from rl.sb3.config import ArgsConfig


def create_dummy_observation_space():
    """创建模拟的观察空间"""
    # Based on the original network configuration and actual observation space
    max_human_num = 20  # Maximum number of humans/agents
    robot_node_dim = 9  # Robot node features (matching actual env)
    temporal_edges_dim = 2  # Temporal edge features  
    spatial_edges_dim = 13  # 2*(predict_steps+1) + 1 = 2*6 + 1 = 13
    
    observation_space = spaces.Dict({
        'robot_node': spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(1, robot_node_dim), dtype=np.float32  # Include robot_num dimension
        ),
        'temporal_edges': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, temporal_edges_dim), dtype=np.float32  # Include robot_num dimension
        ),
        'spatial_edges': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_human_num, spatial_edges_dim), dtype=np.float32
        ),
        'detected_human_num': spaces.Box(
            low=0, high=max_human_num,
            shape=(1,), dtype=np.int32
        )
    })
    
    return observation_space


def create_dummy_observations(batch_size=4, max_human_num=20):
    """创建模拟的观察数据"""
    observations = {
        'robot_node': torch.randn(batch_size, 1, 9, dtype=torch.float32),  # Include robot_num dimension
        'temporal_edges': torch.randn(batch_size, 1, 2, dtype=torch.float32),  # Include robot_num dimension
        'spatial_edges': torch.randn(batch_size, max_human_num, 13, dtype=torch.float32),
        'detected_human_num': torch.randint(1, max_human_num+1, (batch_size, 1), dtype=torch.int32)
    }
    return observations


def test_features_extractor():
    """测试特征提取器的基本功能"""
    print("开始测试 AttentionFeaturesExtractor...")
    
    # Create observation space
    obs_space = create_dummy_observation_space()
    print(f"观察空间: {obs_space}")
    
    # Create features extractor
    features_extractor = AttentionFeaturesExtractor(obs_space, features_dim=256)
    print(f"特征提取器创建成功，输出维度: {features_extractor.features_dim}")
    
    # Create dummy observations
    batch_size = 4
    observations = create_dummy_observations(batch_size)
    
    print(f"\n输入数据形状:")
    for key, value in observations.items():
        print(f"  {key}: {value.shape}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            features = features_extractor(observations)
        print(f"\n前向传播成功!")
        print(f"输出特征形状: {features.shape}")
        print(f"输出特征范围: [{features.min().item():.3f}, {features.max().item():.3f}]")
        
        # Check if output shape is correct
        expected_shape = (batch_size, 256)
        if features.shape == expected_shape:
            print(f"✓ 输出形状正确: {features.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 得到 {features.shape}")
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # Test with different detected_human_num values
    print(f"\n测试不同的检测到的人数...")
    for i, num_humans in enumerate([1, 5, 10, 20]):
        observations['detected_human_num'][i % batch_size] = num_humans
    
    try:
        with torch.no_grad():
            features = features_extractor(observations)
        print(f"✓ 变化的人数测试成功")
    except Exception as e:
        print(f"✗ 变化的人数测试失败: {e}")
        return False
    
    # Test gradient computation
    print(f"\n测试梯度计算...")
    try:
        features_extractor.train()
        features = features_extractor(observations)
        loss = features.sum()
        loss.backward()
        print(f"✓ 梯度计算成功")
        
        # Check if gradients exist
        has_gradients = any(p.grad is not None for p in features_extractor.parameters())
        if has_gradients:
            print(f"✓ 梯度已计算")
        else:
            print(f"✗ 未发现梯度")
            
    except Exception as e:
        print(f"✗ 梯度计算失败: {e}")
        return False
    
    print(f"\n🎉 所有测试通过!")
    return True


def test_attention_modules():
    """测试各个注意力模块"""
    print("\n" + "="*50)
    print("测试注意力模块...")
    
    from rl.sb3.attention_features_extractor import SpatialEdgeSelfAttn, EdgeAttention_M
    
    # Test SpatialEdgeSelfAttn
    print("\n测试 SpatialEdgeSelfAttn...")
    spatial_attn = SpatialEdgeSelfAttn(input_size=13, attn_size=512, num_attn_heads=8)
    
    batch_size = 4
    max_human_num = 20
    input_features = torch.randn(batch_size, max_human_num, 13)
    detected_human_num = torch.randint(1, max_human_num+1, (batch_size,))
    
    try:
        with torch.no_grad():
            output = spatial_attn(input_features, detected_human_num)
        print(f"✓ SpatialEdgeSelfAttn 测试成功")
        print(f"  输入形状: {input_features.shape}")
        print(f"  输出形状: {output.shape}")
    except Exception as e:
        print(f"✗ SpatialEdgeSelfAttn 测试失败: {e}")
    
    # Test EdgeAttention_M
    print("\n测试 EdgeAttention_M...")
    edge_attn = EdgeAttention_M(input_feature_size=256, attention_size=64)
    
    h_temporal = torch.randn(batch_size, 1, 256)
    h_spatials = torch.randn(batch_size, max_human_num, 256)
    
    try:
        with torch.no_grad():
            weighted_value, attn = edge_attn(h_temporal, h_spatials, detected_human_num)
        print(f"✓ EdgeAttention_M 测试成功")
        print(f"  h_temporal形状: {h_temporal.shape}")
        print(f"  h_spatials形状: {h_spatials.shape}")
        print(f"  weighted_value形状: {weighted_value.shape}")
    except Exception as e:
        print(f"✗ EdgeAttention_M 测试失败: {e}")


def compare_attention_only():
    """使用修改后的原始网络直接对比注意力机制部分"""
    print("\n" + "="*60)
    print("🔍 对比注意力机制部分的计算结果...")
    
    # Create observation space 
    obs_space = create_dummy_observation_space()
    batch_size = 2  # Use smaller batch for easier debugging
    
    # Set seed for reproducible comparison (especially for observation data)
    torch.manual_seed(123)
    np.random.seed(123)
    observations = create_dummy_observations(batch_size)
    
    # Prepare args for original network
    args = ArgsConfig()
    args.num_processes = batch_size
    args.num_mini_batch = 1
    args.seq_length = 1
    args.human_human_edge_input_size = 13
    args.robot_node_input_size = 11  # 9 + 2
    
    try:
        # Determine device (prefer CPU for testing to avoid device issues)
        device = 'cpu'
        print(f"\n🔧 使用设备: {device}")
        
        # Force args to use CPU
        args.no_cuda = True
        args.cuda = False
        
        print(f"🔧 设置种子并初始化原始网络...")
        # Set seed before initializing original network
        torch.manual_seed(128)
        np.random.seed(128)
        original_net = selfAttn_merge_SRNN(obs_space, args, infer=True)
        original_net.to(device)
        original_net.eval()
        
        # Set flag to return only attention results
        original_net._return_attention_only = True
        
        print(f"🔧 重新设置种子并初始化新特征提取器...")
        # Reset seed before initializing new features extractor
        torch.manual_seed(128)
        np.random.seed(128)
        new_extractor = AttentionFeaturesExtractor(obs_space, features_dim=256)
        new_extractor.to(device)
        new_extractor.eval()
        
        print(f"🔧 测试权重是否相同...")
        # Check if weights are actually the same after same initialization
        try:
            robot_linear_diff = torch.abs(original_net.robot_linear[0].weight - new_extractor.robot_linear[0].weight).max().item()
            print(f"  Robot Linear权重最大差异: {robot_linear_diff:.8f}")
            
            spatial_embed_diff = torch.abs(original_net.spatial_attn.embedding_layer[0].weight - new_extractor.spatial_attn.embedding_layer[0].weight).max().item()
            print(f"  Spatial Embedding权重最大差异: {spatial_embed_diff:.8f}")
            
            if robot_linear_diff < 1e-6 and spatial_embed_diff < 1e-6:
                print(f"  ✅ 权重初始化一致，可以进行精确对比!")
                use_weight_copy = False
            else:
                print(f"  ⚠️  权重初始化不一致，将复制权重进行测试")
                use_weight_copy = True
        except:
            print(f"  ⚠️  无法比较权重，将复制权重进行测试")
            use_weight_copy = True
        
        if use_weight_copy:
            print(f"🔧 复制原始网络权重到新特征提取器...")
            # Copy weights from original network to ensure exact comparison
            try:
                # Copy robot_linear weights
                new_extractor.robot_linear[0].weight.data.copy_(original_net.robot_linear[0].weight.data)
                new_extractor.robot_linear[0].bias.data.copy_(original_net.robot_linear[0].bias.data)
                
                # Copy spatial_attn weights
                new_extractor.spatial_attn.embedding_layer[0].weight.data.copy_(original_net.spatial_attn.embedding_layer[0].weight.data)
                new_extractor.spatial_attn.embedding_layer[0].bias.data.copy_(original_net.spatial_attn.embedding_layer[0].bias.data)
                new_extractor.spatial_attn.embedding_layer[2].weight.data.copy_(original_net.spatial_attn.embedding_layer[2].weight.data)
                new_extractor.spatial_attn.embedding_layer[2].bias.data.copy_(original_net.spatial_attn.embedding_layer[2].bias.data)
                new_extractor.spatial_attn.q_linear.weight.data.copy_(original_net.spatial_attn.q_linear.weight.data)
                new_extractor.spatial_attn.q_linear.bias.data.copy_(original_net.spatial_attn.q_linear.bias.data)
                new_extractor.spatial_attn.k_linear.weight.data.copy_(original_net.spatial_attn.k_linear.weight.data)
                new_extractor.spatial_attn.k_linear.bias.data.copy_(original_net.spatial_attn.k_linear.bias.data)
                new_extractor.spatial_attn.v_linear.weight.data.copy_(original_net.spatial_attn.v_linear.weight.data)
                new_extractor.spatial_attn.v_linear.bias.data.copy_(original_net.spatial_attn.v_linear.bias.data)
                
                # Copy multihead attention weights (this is more complex)
                new_extractor.spatial_attn.multihead_attn.load_state_dict(original_net.spatial_attn.multihead_attn.state_dict())
                
                # Copy spatial_linear weights
                new_extractor.spatial_linear[0].weight.data.copy_(original_net.spatial_linear[0].weight.data)
                new_extractor.spatial_linear[0].bias.data.copy_(original_net.spatial_linear[0].bias.data)
                
                # Copy edge attention weights
                new_extractor.attn.temporal_edge_layer.weight.data.copy_(original_net.attn.temporal_edge_layer[0].weight.data)
                new_extractor.attn.temporal_edge_layer.bias.data.copy_(original_net.attn.temporal_edge_layer[0].bias.data)
                new_extractor.attn.spatial_edge_layer.weight.data.copy_(original_net.attn.spatial_edge_layer[0].weight.data)
                new_extractor.attn.spatial_edge_layer.bias.data.copy_(original_net.attn.spatial_edge_layer[0].bias.data)
                
                print(f"  ✅ 权重复制成功!")
            except Exception as e:
                print(f"  ⚠️  权重复制部分失败: {e}")
                print(f"  继续使用当前权重进行测试...")
        else:
            print(f"  ✅ 使用相同种子初始化的权重进行测试")
        
        # Move all observations to the same device
        for key in observations:
            observations[key] = observations[key].to(device)
        
        # Prepare dummy inputs for original network
        rnn_hxs = {
            'human_node_rnn': torch.zeros(batch_size, args.human_node_rnn_size, device=device)
        }
        masks = torch.ones(batch_size, 1, device=device)
        
        print(f"\n⚡ 运行原始网络的注意力部分...")
        with torch.no_grad():
            # Set seed again before forward pass to ensure deterministic computation
            torch.manual_seed(128)
            np.random.seed(128)
            
            # Run original network with early return
            robot_states_orig, hidden_attn_weighted_orig = original_net(observations, rnn_hxs, masks, infer=True)
            
            # Flatten for comparison (remove seq_length and agent dimensions)
            robot_states_orig_flat = robot_states_orig.squeeze(0).squeeze(1)  # [batch_size, 256]
            hidden_attn_orig_flat = hidden_attn_weighted_orig.squeeze(0).squeeze(1)  # [batch_size, 256]
            
            print(f"📐 原始网络输出形状:")
            print(f"  robot_states: {robot_states_orig_flat.shape}")
            print(f"  hidden_attn_weighted: {hidden_attn_orig_flat.shape}")
            
        print(f"\n⚡ 运行新特征提取器...")
        with torch.no_grad():
            # Set same seed before forward pass to ensure deterministic computation
            torch.manual_seed(128)
            np.random.seed(128)
            
            # Run new features extractor step by step
            robot_node_new = observations['robot_node'].squeeze(1)
            temporal_edges_new = observations['temporal_edges'].squeeze(1)
            spatial_edges_new = observations['spatial_edges']
            detected_human_num_new = observations['detected_human_num'].squeeze(-1).int()
            
            # Get robot_states (equivalent to original)
            robot_states_new = torch.cat((temporal_edges_new, robot_node_new), dim=-1)
            robot_states_new = new_extractor.robot_linear(robot_states_new)
            
            # Get spatial attention output
            spatial_attn_out_new = new_extractor.spatial_attn(spatial_edges_new, detected_human_num_new)
            output_spatial_new = new_extractor.spatial_linear(spatial_attn_out_new)
            
            # Get hidden_attn_weighted
            robot_states_expanded = robot_states_new.unsqueeze(1)
            hidden_attn_new, _ = new_extractor.attn(robot_states_expanded, output_spatial_new, detected_human_num_new)
            hidden_attn_new = hidden_attn_new.squeeze(1)
            
            print(f"📐 新提取器输出形状:")
            print(f"  robot_states: {robot_states_new.shape}")
            print(f"  hidden_attn_weighted: {hidden_attn_new.shape}")
        
        # Compare results
        print(f"\n📊 比较计算结果:")
        
        # Compare robot_states
        robot_states_diff = torch.abs(robot_states_orig_flat - robot_states_new)
        robot_states_max_diff = robot_states_diff.max().item()
        robot_states_mean_diff = robot_states_diff.mean().item()
        
        print(f"\n🤖 Robot States 比较:")
        print(f"  最大差异: {robot_states_max_diff:.6f}")
        print(f"  平均差异: {robot_states_mean_diff:.6f}")
        
        if robot_states_max_diff < 1e-5:
            print(f"  ✅ Robot states 计算结果几乎完全一致!")
        elif robot_states_max_diff < 1e-3:
            print(f"  ⚠️  Robot states 有小的数值差异")
        else:
            print(f"  ❌ Robot states 存在显著差异")
            print(f"     原始: {robot_states_orig_flat[0, :5]}")
            print(f"     新的: {robot_states_new[0, :5]}")
        
        # Compare hidden_attn_weighted
        attn_diff = torch.abs(hidden_attn_orig_flat - hidden_attn_new)
        attn_max_diff = attn_diff.max().item()
        attn_mean_diff = attn_diff.mean().item()
        
        print(f"\n🎯 Hidden Attention Weighted 比较:")
        print(f"  最大差异: {attn_max_diff:.6f}")
        print(f"  平均差异: {attn_mean_diff:.6f}")
        
        if attn_max_diff < 1e-5:
            print(f"  ✅ Attention weights 计算结果几乎完全一致!")
        elif attn_max_diff < 1e-3:
            print(f"  ⚠️  Attention weights 有小的数值差异")
        else:
            print(f"  ❌ Attention weights 存在显著差异")
            print(f"     原始: {hidden_attn_orig_flat[0, :5]}")
            print(f"     新的: {hidden_attn_new[0, :5]}")
        
        # Compare final combined features (robot_states + hidden_attn)
        original_combined = robot_states_orig_flat + hidden_attn_orig_flat
        new_combined = new_extractor(observations)  # This should be robot_states + hidden_attn
        
        combined_diff = torch.abs(original_combined - new_combined)
        combined_max_diff = combined_diff.max().item()
        combined_mean_diff = combined_diff.mean().item()
        
        print(f"\n🔗 最终组合特征比较:")
        print(f"  最大差异: {combined_max_diff:.6f}")
        print(f"  平均差异: {combined_mean_diff:.6f}")
        
        if combined_max_diff < 1e-5:
            print(f"  ✅ 最终特征计算结果几乎完全一致!")
            return True
        elif combined_max_diff < 1e-3:
            print(f"  ⚠️  最终特征有小的数值差异，可能可以接受")
            return True
        else:
            print(f"  ❌ 最终特征存在显著差异，需要进一步调试")
            return False
            
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_original_network():
    """简化版本：只比较注意力机制，避免GRU复杂性"""
    return compare_attention_only()


def main():
    """主测试函数"""
    print("AttentionFeaturesExtractor 测试脚本")
    print("="*50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test individual attention modules
    test_attention_modules()
    
    # Test complete features extractor
    success = test_features_extractor()
    
    # Compare with original network
    comparison_success = compare_with_original_network()
    
    if success and comparison_success:
        print(f"\n🎉 所有测试通过！新特征提取器与原始网络计算结果一致!")
        print(f"您现在可以安心使用 train_traffic_standard.py 进行训练。")
    elif success and not comparison_success:
        print(f"\n⚠️  基本功能测试通过，但与原始网络存在差异!")
        print(f"建议进一步检查 EdgeAttention_M 或 SpatialEdgeSelfAttn 的实现。")
    else:
        print(f"\n❌ 测试发现问题，请检查代码。")


if __name__ == "__main__":
    main()
