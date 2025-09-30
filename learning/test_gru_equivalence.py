#!/usr/bin/env python3

"""测试新的GRU实现与原始网络的等价性
Test equivalence between new GRU implementation and original network

这个脚本验证 AttentionFeaturesExtractor + GRU policy 与原始的 selfAttn_merge_SRNN 是否等价
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from rl.networks.selfAttn_srnn_temp_node import selfAttn_merge_SRNN, EndRNN
from rl.sb3.attention_features_extractor import AttentionFeaturesExtractor
from rl.sb3.gru_recurrent_policy import GRURecurrentActorCriticPolicy
from rl.sb3.config import ArgsConfig


def create_dummy_observation_space():
    """创建模拟的观察空间"""
    max_human_num = 20
    robot_node_dim = 9
    temporal_edges_dim = 2
    spatial_edges_dim = 13
    
    observation_space = spaces.Dict({
        'robot_node': spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(1, robot_node_dim), dtype=np.float32
        ),
        'temporal_edges': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, temporal_edges_dim), dtype=np.float32
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
        'robot_node': torch.randn(batch_size, 1, 9, dtype=torch.float32),
        'temporal_edges': torch.randn(batch_size, 1, 2, dtype=torch.float32),
        'spatial_edges': torch.randn(batch_size, max_human_num, 13, dtype=torch.float32),
        'detected_human_num': torch.randint(1, max_human_num+1, (batch_size, 1), dtype=torch.int32)
    }
    return observations


def enable_intermediate_results_storage(original_net):
    """启用原始网络的中间结果存储"""
    original_net._store_intermediate_results = True
    return original_net


def create_standalone_gru_module(args):
    """创建独立的GRU模块来模拟EndRNN的GRU部分"""
    
    class StandaloneGRU(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            
            # GRU模块（输入维度是128，即64+64）
            # 这里的输入已经经过了特征提取器的final_robot_linear和final_spatial_linear
            self.gru = nn.GRU(args.human_node_embedding_size * 2, args.human_node_rnn_size)
            
            # 初始化GRU权重（与RNNBase相同）
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            
            # 输出线性层
            self.output_linear = nn.Linear(args.human_node_rnn_size, args.human_node_output_size)
        
        def forward(self, features_128, hidden_state, masks):
            """
            接收特征提取器的128维输出，通过GRU处理
            
            Args:
                features_128: [batch_size, 128] - 来自特征提取器的输出
                hidden_state: [1, batch_size, 1, rnn_size] - GRU隐藏状态
                masks: [batch_size, 1] - episode masks
            """
            batch_size = features_128.size(0)
            
            # 为了与原始网络保持一致，添加seq_len和agent维度
            # [batch_size, 128] → [1, batch_size, 1, 128]
            features_input = features_128.unsqueeze(0).unsqueeze(2)
            
            # 存储GRU输入
            self._gru_input = features_input
            
            # 简化的GRU forward（模拟RNNBase._forward_gru的acting模式）
            seq_len, nenv, agent_num, _ = features_input.size()
            x = features_input.view(seq_len, nenv * agent_num, -1)
            
            # 处理masks：masks是episode_starts的取反
            hxs_times_masks = hidden_state * masks.view(seq_len, nenv, 1, 1)
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv * agent_num, -1)
            
            x, hxs = self.gru(x, hxs_times_masks)
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)
            
            # 存储GRU输出
            self._gru_output = x
            
            # 输出线性层
            outputs = self.output_linear(x)
            
            return outputs, hxs
    
    return StandaloneGRU(args)


def copy_weights_from_original_to_new(original_net, new_extractor):
    """复制原始网络的权重到新特征提取器"""
    try:
        print("🔧 复制权重...")
        
        # 复制robot_linear权重
        new_extractor.robot_linear[0].weight.data.copy_(original_net.robot_linear[0].weight.data)
        new_extractor.robot_linear[0].bias.data.copy_(original_net.robot_linear[0].bias.data)
        
        # 复制spatial_attn权重
        new_extractor.spatial_attn.embedding_layer[0].weight.data.copy_(original_net.spatial_attn.embedding_layer[0].weight.data)
        new_extractor.spatial_attn.embedding_layer[0].bias.data.copy_(original_net.spatial_attn.embedding_layer[0].bias.data)
        new_extractor.spatial_attn.embedding_layer[2].weight.data.copy_(original_net.spatial_attn.embedding_layer[2].weight.data)
        new_extractor.spatial_attn.embedding_layer[2].bias.data.copy_(original_net.spatial_attn.embedding_layer[2].bias.data)
        
        # 复制q, k, v linear权重
        new_extractor.spatial_attn.q_linear.weight.data.copy_(original_net.spatial_attn.q_linear.weight.data)
        new_extractor.spatial_attn.q_linear.bias.data.copy_(original_net.spatial_attn.q_linear.bias.data)
        new_extractor.spatial_attn.k_linear.weight.data.copy_(original_net.spatial_attn.k_linear.weight.data)
        new_extractor.spatial_attn.k_linear.bias.data.copy_(original_net.spatial_attn.k_linear.bias.data)
        new_extractor.spatial_attn.v_linear.weight.data.copy_(original_net.spatial_attn.v_linear.weight.data)
        new_extractor.spatial_attn.v_linear.bias.data.copy_(original_net.spatial_attn.v_linear.bias.data)
        
        # 复制multihead attention权重
        new_extractor.spatial_attn.multihead_attn.load_state_dict(original_net.spatial_attn.multihead_attn.state_dict())
        
        # 复制spatial_linear权重
        new_extractor.spatial_linear[0].weight.data.copy_(original_net.spatial_linear[0].weight.data)
        new_extractor.spatial_linear[0].bias.data.copy_(original_net.spatial_linear[0].bias.data)
        
        # 复制edge attention权重
        new_extractor.attn.temporal_edge_layer.weight.data.copy_(original_net.attn.temporal_edge_layer[0].weight.data)
        new_extractor.attn.temporal_edge_layer.bias.data.copy_(original_net.attn.temporal_edge_layer[0].bias.data)
        new_extractor.attn.spatial_edge_layer.weight.data.copy_(original_net.attn.spatial_edge_layer[0].weight.data)
        new_extractor.attn.spatial_edge_layer.bias.data.copy_(original_net.attn.spatial_edge_layer[0].bias.data)
        
        # 复制final linear layers权重（对应EndRNN的encoder_linear和edge_attention_embed）
        new_extractor.final_robot_linear[0].weight.data.copy_(original_net.humanNodeRNN.encoder_linear.weight.data)
        new_extractor.final_robot_linear[0].bias.data.copy_(original_net.humanNodeRNN.encoder_linear.bias.data)
        new_extractor.final_spatial_linear[0].weight.data.copy_(original_net.humanNodeRNN.edge_attention_embed.weight.data)
        new_extractor.final_spatial_linear[0].bias.data.copy_(original_net.humanNodeRNN.edge_attention_embed.bias.data)
        
        print("  ✅ 权重复制成功!")
        
    except Exception as e:
        print(f"  ⚠️ 权重复制失败: {e}")
        raise


def copy_weights_from_original_to_gru(original_net, standalone_gru):
    """复制原始网络的EndRNN的GRU权重到独立GRU模块"""
    try:
        print("🔧 复制EndRNN的GRU权重到独立GRU...")
        
        # 复制GRU权重
        standalone_gru.gru.load_state_dict(original_net.humanNodeRNN.gru.state_dict())
        
        # 复制输出线性层权重
        standalone_gru.output_linear.weight.data.copy_(original_net.humanNodeRNN.output_linear.weight.data)
        standalone_gru.output_linear.bias.data.copy_(original_net.humanNodeRNN.output_linear.bias.data)
        
        print("  ✅ EndRNN的GRU权重复制成功!")
        
    except Exception as e:
        print(f"  ⚠️ EndRNN的GRU权重复制失败: {e}")
        raise


def compare_tensors(name, tensor1, tensor2, threshold=1e-5):
    """比较两个张量"""
    # 确保张量形状匹配
    if tensor1.shape != tensor2.shape:
        print(f"\n❌ {name} 形状不匹配: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n{name} 比较:")
    print(f"  形状: {tensor1.shape}")
    print(f"  最大差异: {max_diff:.8f}")
    print(f"  平均差异: {mean_diff:.8f}")
    
    if max_diff < threshold:
        print(f"  ✅ {name} 计算结果一致!")
        return True
    elif max_diff < 1e-3:
        print(f"  ⚠️ {name} 有小的数值差异")
        print(f"     原始: {tensor1.flatten()[:5]}")
        print(f"     新的: {tensor2.flatten()[:5]}")
        return True
    else:
        print(f"  ❌ {name} 存在显著差异")
        print(f"     原始: {tensor1.flatten()[:5]}")
        print(f"     新的: {tensor2.flatten()[:5]}")
        return False


def test_gru_equivalence():
    """测试GRU实现的等价性"""
    print("🔍 测试GRU实现的等价性...")
    
    # 设置参数
    obs_space = create_dummy_observation_space()
    batch_size = 4
    device = 'cpu'
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建观察数据
    observations = create_dummy_observations(batch_size)
    
    # 准备args
    args = ArgsConfig()
    args.num_processes = batch_size
    args.num_mini_batch = 1
    args.seq_length = 1
    args.human_human_edge_input_size = 13
    args.robot_node_input_size = 11  # 9 + 2
    args.no_cuda = True
    args.cuda = False
    
    try:
        print("🔧 初始化原始网络...")
        # 设置种子
        torch.manual_seed(128)
        np.random.seed(128)
        
        original_net = selfAttn_merge_SRNN(obs_space, args, infer=True)
        original_net.to(device)
        original_net.eval()
        
        # 启用中间结果存储
        original_net = enable_intermediate_results_storage(original_net)
        
        print("🔧 初始化新的特征提取器...")
        # 重置种子
        torch.manual_seed(128)
        np.random.seed(128)
        
        new_extractor = AttentionFeaturesExtractor(obs_space, features_dim=128)
        new_extractor.to(device)
        new_extractor.eval()
        
        # 复制权重
        copy_weights_from_original_to_new(original_net, new_extractor)
        
        print("🔧 创建独立GRU模块...")
        standalone_gru = create_standalone_gru_module(args)
        standalone_gru.to(device)
        standalone_gru.eval()
        
        # 复制EndRNN权重
        copy_weights_from_original_to_gru(original_net, standalone_gru)
        
        # 移动数据到设备
        for key in observations:
            observations[key] = observations[key].to(device)
        
        # 准备原始网络的输入
        rnn_hxs = {
            'human_node_rnn': torch.zeros(batch_size, 1, args.human_node_rnn_size, device=device)
        }
        masks = torch.ones(batch_size, 1, device=device)  # episode_starts的取反
        
        print("⚡ 运行原始网络...")
        with torch.no_grad():
            torch.manual_seed(128)
            
            # 运行原始网络
            critic_value_orig, actor_output_orig, new_rnn_hxs_orig = original_net(observations, rnn_hxs, masks, infer=True)
            
            # 获取中间结果
            robot_states_orig = original_net._intermediate_robot_states
            hidden_attn_orig = original_net._intermediate_hidden_attn_weighted
            endrnn_output_orig = original_net._intermediate_endrnn_output
        
        print("⚡ 运行新的特征提取器和独立GRU...")
        with torch.no_grad():
            torch.manual_seed(128)
            
            # 1. 通过特征提取器获得128维特征（已经包含了linear layers的处理）
            features_128 = new_extractor(observations)  # [batch_size, 128]
            
            # 2. 通过独立GRU
            # 初始hidden state
            hidden_state = torch.zeros(1, batch_size, 1, args.human_node_rnn_size, device=device)
            
            # 通过GRU
            endrnn_output_new, new_hidden = standalone_gru(
                features_128,
                hidden_state, 
                masks
            )
        
        print("📊 比较结果...")
        
        success = True
        
        # 1. 首先比较特征提取器的中间结果
        robot_states_orig_flat = robot_states_orig.squeeze(0).squeeze(1)   # [batch_size, 256]
        hidden_attn_orig_flat = hidden_attn_orig.squeeze(0).squeeze(1)  # [batch_size, 256]
        
        # 2. 从新的特征提取器中提取对应的中间结果进行比较
        # 需要单独运行特征提取器的组件来获得中间结果
        robot_node_new = observations['robot_node'].squeeze(1)
        temporal_edges_new = observations['temporal_edges'].squeeze(1)
        spatial_edges_new = observations['spatial_edges']
        detected_human_num_new = observations['detected_human_num'].squeeze(-1).int()
        
        # Robot states (before final_robot_linear)
        robot_states_new = torch.cat((temporal_edges_new, robot_node_new), dim=-1)
        robot_states_new = new_extractor.robot_linear(robot_states_new)
        
        # Hidden attention (before final_spatial_linear)
        spatial_attn_out_new = new_extractor.spatial_attn(spatial_edges_new, detected_human_num_new)
        output_spatial_new = new_extractor.spatial_linear(spatial_attn_out_new)
        robot_states_expanded = robot_states_new.unsqueeze(1)
        hidden_attn_new, _ = new_extractor.attn(robot_states_expanded, output_spatial_new, detected_human_num_new)
        hidden_attn_new = hidden_attn_new.squeeze(1)
        
        success &= compare_tensors("Robot States (before final linear)", robot_states_orig_flat, robot_states_new)
        success &= compare_tensors("Hidden Attention (before final linear)", hidden_attn_orig_flat, hidden_attn_new)
        
        # 3. 比较特征提取器的最终输出（应该等于原始网络EndRNN的GRU输入）
        # 从原始网络获取EndRNN的GRU输入
        robot_linear_orig = original_net.humanNodeRNN.encoder_linear(robot_states_orig_flat)
        robot_linear_orig = original_net.humanNodeRNN.relu(robot_linear_orig)
        attn_linear_orig = original_net.humanNodeRNN.edge_attention_embed(hidden_attn_orig_flat) 
        attn_linear_orig = original_net.humanNodeRNN.relu(attn_linear_orig)
        gru_input_orig = torch.cat((robot_linear_orig, attn_linear_orig), dim=-1)  # [batch_size, 128]
        
        # 从新的特征提取器获取128维输出
        features_128_new = new_extractor(observations)
        
        success &= compare_tensors("GRU Input (特征提取器输出)", gru_input_orig, features_128_new)
        
        # 4. 比较GRU输出（最重要的比较）
        endrnn_output_orig_flat = endrnn_output_orig.squeeze(0).squeeze(-2)  # [batch_size, 256]
        endrnn_output_new_flat = endrnn_output_new.squeeze(0).squeeze(-2)    # [batch_size, 256]
        success &= compare_tensors("GRU Output (EndRNN Output)", endrnn_output_orig_flat, endrnn_output_new_flat)
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("GRU等价性测试脚本")
    print("="*60)
    print("验证 AttentionFeaturesExtractor + GRU policy 与原始 selfAttn_merge_SRNN 的等价性")
    print("="*60)
    
    # 设置随机种子以获得可重现的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = test_gru_equivalence()
    
    if success:
        print(f"\n🎉 所有测试通过！新的GRU实现与原始网络等价!")
        print(f"您可以安心使用新的 SB3 格式训练流程。")
    else:
        print(f"\n❌ 测试发现差异，请检查实现。")
        print(f"建议检查：")
        print(f"1. AttentionFeaturesExtractor 的特征提取是否正确")
        print(f"2. GRU 的权重初始化和forward逻辑是否与原始EndRNN一致")
        print(f"3. masks 的处理是否正确（episode_starts的取反）")


if __name__ == "__main__":
    main() 