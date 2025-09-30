#!/usr/bin/env python3

"""测试GRU前向计算方法的等价性
Test equivalence between original and optimized GRU forward methods

验证 _process_sequence_gru 和 _process_sequence_gru_optimized 的计算结果是否相同
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from rl.sb3.gru_recurrent_policy import GRURecurrentActorCriticPolicy


class GRUMethodsTester:
    """测试两种GRU方法的等价性"""
    
    def __init__(self, input_size=128, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 创建GRU模块用于测试
        self.gru = nn.GRU(input_size, hidden_size)
        
        # 固定权重以确保可重现性
        torch.manual_seed(42)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    
    def compare_tensors_detailed(self, name, tensor1, tensor2, threshold=1e-8):
        """详细比较两个张量"""
        # 确保张量形状匹配
        if tensor1.shape != tensor2.shape:
            print(f"\n❌ {name} 形状不匹配: {tensor1.shape} vs {tensor2.shape}")
            return False
        
        # 计算差异
        diff = torch.abs(tensor1 - tensor2)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 数值统计
        t1_min, t1_max = tensor1.min().item(), tensor1.max().item()
        t2_min, t2_max = tensor2.min().item(), tensor2.max().item()
        t1_mean, t1_std = tensor1.mean().item(), tensor1.std().item()
        t2_mean, t2_std = tensor2.mean().item(), tensor2.std().item()
        
        # 相对差异
        magnitude = max(abs(t1_max), abs(t1_min), abs(t2_max), abs(t2_min))
        relative_diff = max_diff / (magnitude + 1e-8)
        
        print(f"\n{name} 详细比较:")
        print(f"  形状: {tensor1.shape}")
        print(f"  绝对差异: 最大={max_diff:.2e}, 平均={mean_diff:.2e}")
        print(f"  相对差异: {relative_diff:.2e}")
        
        print(f"  原始张量: 范围=[{t1_min:.4f}, {t1_max:.4f}], 均值={t1_mean:.4f}±{t1_std:.4f}")
        print(f"  新张量:   范围=[{t2_min:.4f}, {t2_max:.4f}], 均值={t2_mean:.4f}±{t2_std:.4f}")
        print(f"  样本对比: 原始{tensor1.flatten()[:3].detach().numpy()} vs 新{tensor2.flatten()[:3].detach().numpy()}")
        
        # 判断结果
        if max_diff < threshold:
            print(f"  ✅ {name} 一致! (阈值: {threshold:.2e})")
            return True
        elif max_diff < 1e-5:
            print(f"  ⚠️ {name} 有微小差异，可能可接受")
            return True
        else:
            print(f"  ❌ {name} 差异显著")
            return False
    
    def create_test_data(self, n_seq, seq_len, scenario="normal"):
        """创建测试数据"""
        batch_size = n_seq * seq_len
        
        # 创建特征数据
        features = torch.randn(batch_size, self.input_size)
        
        # 创建初始隐藏状态 (1, n_seq, hidden_size)
        hidden_state = torch.randn(1, n_seq, self.hidden_size)
        dummy_cell_state = torch.zeros_like(hidden_state)  # GRU不用，但保持接口兼容
        gru_states = (hidden_state, dummy_cell_state)
        
        # 创建episode_starts
        if scenario == "no_resets":
            episode_starts = torch.zeros(batch_size)
        elif scenario == "single_reset":
            episode_starts = torch.zeros(batch_size)
            episode_starts[seq_len//2] = 1.0  # 在序列中间重置
        elif scenario == "multiple_resets":
            episode_starts = torch.zeros(batch_size)
            episode_starts[seq_len//3] = 1.0
            episode_starts[2*seq_len//3] = 1.0
        elif scenario == "random_resets":
            episode_starts = torch.bernoulli(torch.full((batch_size,), 0.2))  # 20%概率重置
        elif scenario == "first_step_reset":
            episode_starts = torch.zeros(batch_size)
            episode_starts[0] = 1.0  # 第一步重置
        else:  # "normal"
            episode_starts = torch.zeros(batch_size)
            if seq_len > 1:
                episode_starts[torch.randint(1, seq_len, (1,))] = 1.0
        
        return features, gru_states, episode_starts
    
    def test_scenario(self, n_seq, seq_len, scenario_name):
        """测试特定场景"""
        print(f"\n📋 测试场景: {scenario_name}")
        print(f"   n_seq={n_seq}, seq_len={seq_len}, batch_size={n_seq*seq_len}")
        
        # 创建测试数据
        features, gru_states, episode_starts = self.create_test_data(n_seq, seq_len, scenario_name)
        
        # 运行原始方法
        torch.manual_seed(123)  # 确保deterministic
        output1, states1 = GRURecurrentActorCriticPolicy._process_sequence_gru(
            features, gru_states, episode_starts, self.gru
        )
        
        # 运行优化方法
        torch.manual_seed(123)  # 相同种子
        output2, states2 = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
            features, gru_states, episode_starts, self.gru
        )
        
        # 比较结果
        success = True
        success &= self.compare_tensors_detailed(f"{scenario_name}_输出", output1, output2)
        success &= self.compare_tensors_detailed(f"{scenario_name}_隐藏状态", states1[0], states2[0])
        
        # 额外检查：episode_starts的分布
        resets_count = episode_starts.sum().item()
        print(f"   Episode重置次数: {resets_count}/{len(episode_starts)}")
        print(f"   Episode重置位置: {episode_starts.nonzero().squeeze().tolist()}")
        
        return success


def run_comprehensive_tests():
    """运行全面的测试套件"""
    print("🧪 GRU方法等价性测试")
    print("="*60)
    
    tester = GRUMethodsTester()
    all_success = True
    
    # 测试不同的批处理配置
    test_configs = [
        # (n_seq, seq_len, scenario)
        (1, 1, "no_resets"),           # 最简单情况
        (4, 1, "no_resets"),           # 多环境，单步
        (1, 10, "no_resets"),          # 单环境，多步
        (4, 10, "no_resets"),          # 多环境，多步
        
        (4, 10, "single_reset"),       # 单次重置
        (4, 10, "multiple_resets"),    # 多次重置
        (4, 10, "first_step_reset"),   # 首步重置
        (4, 20, "random_resets"),      # 随机重置
        
        # 边界情况
        (1, 30, "multiple_resets"),    # 长序列
        (5, 128, "random_resets"),       # 更多环境
    ]
    
    print(f"将运行 {len(test_configs)} 个测试场景...")
    
    for n_seq, seq_len, scenario in test_configs:
        try:
            success = tester.test_scenario(n_seq, seq_len, scenario)
            all_success &= success
            
            if not success:
                print(f"❌ 场景 {scenario} (n_seq={n_seq}, seq_len={seq_len}) 失败")
            else:
                print(f"✅ 场景 {scenario} 通过")
                
        except Exception as e:
            print(f"❌ 场景 {scenario} 出现异常: {e}")
            all_success = False
    
    return all_success


def test_edge_cases():
    """测试边界情况"""
    print(f"\n🔍 测试边界情况...")
    
    tester = GRUMethodsTester()
    success = True
    
    # 测试1: 全部都是episode重置
    print(f"\n测试1: 全部episode重置")
    n_seq, seq_len = 3, 5
    features = torch.randn(n_seq * seq_len, tester.input_size)
    hidden_state = torch.randn(1, n_seq, tester.hidden_size)
    dummy_cell = torch.zeros_like(hidden_state)
    gru_states = (hidden_state, dummy_cell)
    episode_starts = torch.ones(n_seq * seq_len)  # 全部重置
    
    try:
        output1, states1 = GRURecurrentActorCriticPolicy._process_sequence_gru(
            features, gru_states, episode_starts, tester.gru
        )
        output2, states2 = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
            features, gru_states, episode_starts, tester.gru
        )
        
        success &= tester.compare_tensors_detailed("全重置_输出", output1, output2)
        success &= tester.compare_tensors_detailed("全重置_状态", states1[0], states2[0])
        
    except Exception as e:
        print(f"❌ 全重置测试失败: {e}")
        success = False
    
    # 测试2: 零输入
    print(f"\n测试2: 零输入")
    features_zero = torch.zeros(n_seq * seq_len, tester.input_size)
    episode_starts_zero = torch.zeros(n_seq * seq_len)
    
    try:
        output1, states1 = GRURecurrentActorCriticPolicy._process_sequence_gru(
            features_zero, gru_states, episode_starts_zero, tester.gru
        )
        output2, states2 = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
            features_zero, gru_states, episode_starts_zero, tester.gru
        )
        
        success &= tester.compare_tensors_detailed("零输入_输出", output1, output2)
        success &= tester.compare_tensors_detailed("零输入_状态", states1[0], states2[0])
        
    except Exception as e:
        print(f"❌ 零输入测试失败: {e}")
        success = False
    
    # 测试3: 大批量
    print(f"\n测试3: 大批量处理")
    n_seq, seq_len = 16, 30
    features_large = torch.randn(n_seq * seq_len, tester.input_size)
    hidden_large = torch.randn(1, n_seq, tester.hidden_size)
    dummy_large = torch.zeros_like(hidden_large)
    gru_states_large = (hidden_large, dummy_large)
    episode_starts_large = torch.bernoulli(torch.full((n_seq * seq_len,), 0.1))  # 10%重置概率
    
    try:
        torch.manual_seed(999)
        output1, states1 = GRURecurrentActorCriticPolicy._process_sequence_gru(
            features_large, gru_states_large, episode_starts_large, tester.gru
        )
        
        torch.manual_seed(999)
        output2, states2 = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
            features_large, gru_states_large, episode_starts_large, tester.gru
        )
        
        success &= tester.compare_tensors_detailed("大批量_输出", output1, output2)
        success &= tester.compare_tensors_detailed("大批量_状态", states1[0], states2[0])
        
    except Exception as e:
        print(f"❌ 大批量测试失败: {e}")
        success = False
    
    return success


def test_performance_difference():
    """测试性能差异（可选）"""
    print(f"\n⚡ 性能对比测试...")
    
    import time
    
    tester = GRUMethodsTester()
    n_seq, seq_len = 16, 50
    features = torch.randn(n_seq * seq_len, tester.input_size)
    hidden_state = torch.randn(1, n_seq, tester.hidden_size)
    dummy_cell = torch.zeros_like(hidden_state)
    gru_states = (hidden_state, dummy_cell)
    episode_starts = torch.bernoulli(torch.full((n_seq * seq_len,), 0.15))
    
    num_runs = 10
    
    try:
        # 测试原始方法
        start_time = time.time()
        for _ in range(num_runs):
            torch.manual_seed(42)
            output1, _ = GRURecurrentActorCriticPolicy._process_sequence_gru(
                features, gru_states, episode_starts, tester.gru
            )
        original_time = time.time() - start_time
        
        # 测试优化方法
        start_time = time.time()
        for _ in range(num_runs):
            torch.manual_seed(42)
            output2, _ = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
                features, gru_states, episode_starts, tester.gru
            )
        optimized_time = time.time() - start_time
        
        print(f"  原始方法平均耗时: {original_time/num_runs*1000:.2f} ms")
        print(f"  优化方法平均耗时: {optimized_time/num_runs*1000:.2f} ms")
        
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"  🚀 加速比: {speedup:.2f}x")
        else:
            slowdown = optimized_time / original_time
            print(f"  🐌 变慢: {slowdown:.2f}x")
        
        # 验证结果一致性
        return tester.compare_tensors_detailed("性能测试_结果一致性", output1, output2)
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def analyze_episode_starts_patterns():
    """分析不同episode_starts模式的处理"""
    print(f"\n🔍 分析episode_starts模式...")
    
    patterns = {
        "全零": torch.zeros(20),
        "首个重置": torch.cat([torch.tensor([1.0]), torch.zeros(19)]),
        "末尾重置": torch.cat([torch.zeros(19), torch.tensor([1.0])]),
        "中间重置": torch.cat([torch.zeros(10), torch.tensor([1.0]), torch.zeros(9)]),
        "连续重置": torch.cat([torch.zeros(8), torch.ones(3), torch.zeros(9)]),
        "交替重置": torch.tensor([1.0 if i % 4 == 0 else 0.0 for i in range(20)]),
    }
    
    tester = GRUMethodsTester()
    n_seq = 4
    
    for pattern_name, episode_starts in patterns.items():
        print(f"\n  测试模式: {pattern_name}")
        print(f"    重置位置: {episode_starts.nonzero().squeeze().tolist()}")
        
        seq_len = len(episode_starts) // n_seq
        features = torch.randn(len(episode_starts), tester.input_size)
        hidden_state = torch.randn(1, n_seq, tester.hidden_size)
        dummy_cell = torch.zeros_like(hidden_state)
        gru_states = (hidden_state, dummy_cell)
        
        try:
            torch.manual_seed(555)
            output1, states1 = GRURecurrentActorCriticPolicy._process_sequence_gru(
                features, gru_states, episode_starts, tester.gru
            )
            
            torch.manual_seed(555)
            output2, states2 = GRURecurrentActorCriticPolicy._process_sequence_gru_optimized(
                features, gru_states, episode_starts, tester.gru
            )
            
            success = tester.compare_tensors_detailed(f"{pattern_name}_模式", output1, output2, threshold=1e-7)
            
            if success:
                print(f"    ✅ {pattern_name} 模式通过")
            else:
                print(f"    ❌ {pattern_name} 模式失败")
                
        except Exception as e:
            print(f"    ❌ {pattern_name} 模式异常: {e}")


def main():
    """主测试函数"""
    print("GRU前向计算方法等价性测试")
    print("="*80)
    print("比较 _process_sequence_gru 和 _process_sequence_gru_optimized 的计算结果")
    print("="*80)
    
    # 设置全局种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 运行主要测试套件
        print("\n🚀 运行主要测试套件...")
        main_success = run_comprehensive_tests()
        
        # 2. 测试边界情况
        edge_success = test_edge_cases()
        
        # 3. 分析episode_starts模式
        analyze_episode_starts_patterns()
        
        # 4. 性能对比（可选）
        perf_success = test_performance_difference()
        
        # 总结
        print(f"\n" + "="*80)
        print(f"📊 测试总结:")
        print(f"  主要测试: {'✅ 通过' if main_success else '❌ 失败'}")
        print(f"  边界测试: {'✅ 通过' if edge_success else '❌ 失败'}")
        print(f"  性能测试: {'✅ 通过' if perf_success else '❌ 失败'}")
        
        overall_success = main_success and edge_success and perf_success
        
        if overall_success:
            print(f"\n🎉 所有测试通过！优化版本与原始版本完全等价!")
            print(f"您可以安心使用优化版本的GRU实现。")
        else:
            print(f"\n⚠️ 部分测试失败，请检查优化版本的实现。")
            print(f"\n🔧 调试建议:")
            print(f"1. 检查episode_starts的处理逻辑")
            print(f"2. 验证序列分块的边界条件")
            print(f"3. 确认状态重置的时机和方式")
            print(f"4. 检查张量维度变换的正确性")
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 