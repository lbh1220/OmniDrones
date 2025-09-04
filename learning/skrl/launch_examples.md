# SKRL Training Launch Examples

这个文件包含了各种SKRL训练的启动示例，帮助您快速开始不同场景的训练。

## 基本训练示例

### 1. 默认配置训练

```bash
# 使用默认配置文件进行训练
./learning/skrl/run_skrl_traffic.sh
```

### 2. 快速测试（小规模）

```bash
# 使用小规模配置进行快速测试
./learning/skrl/run_skrl_traffic.sh --config learning/skrl/configs/small_scale.yaml

# 或者直接使用Python
python learning/skrl/skrl_train_traffic.py --config learning/skrl/configs/small_scale.yaml
```

### 3. 生产环境训练（大规模）

```bash
# 使用大规模配置进行生产训练
./learning/skrl/run_skrl_traffic.sh --config learning/skrl/configs/large_scale.yaml --use_wandb
```

## 参数覆盖示例

### 4. 调整环境数量和学习率

```bash
# 覆盖环境数量和学习率
./learning/skrl/run_skrl_traffic.sh \
    --num_envs 2048 \
    --learning_rate 3e-5 \
    --experiment_name "high_lr_experiment"
```

### 5. 设置交通环境参数

```bash
# 设置特定的无人机和eVTOL数量
./learning/skrl/run_skrl_traffic.sh \
    --drones_num 10 \
    --evtols_num 3 \
    --experiment_name "complex_traffic"
```

### 6. 启用Wandb记录

```bash
# 启用Wandb并设置项目名
./learning/skrl/run_skrl_traffic.sh \
    --use_wandb \
    --wandb_project "my_traffic_research" \
    --experiment_name "baseline_experiment"
```

## 研究和实验示例

### 7. 学习率扫描实验

```bash
# 不同学习率的实验
for lr in 1e-5 3e-5 5e-5 1e-4; do
    ./learning/skrl/run_skrl_traffic.sh \
        --learning_rate $lr \
        --experiment_name "lr_sweep_${lr}" \
        --use_wandb \
        --wandb_project "lr_ablation_study"
done
```

### 8. 网络规模对比实验

```bash
# 小规模网络
./learning/skrl/run_skrl_traffic.sh \
    --config learning/skrl/configs/small_scale.yaml \
    --experiment_name "small_network" \
    --use_wandb

# 大规模网络
./learning/skrl/run_skrl_traffic.sh \
    --config learning/skrl/configs/large_scale.yaml \
    --experiment_name "large_network" \
    --use_wandb
```

### 9. 交通复杂度递增实验

```bash
# 简单场景
./learning/skrl/run_skrl_traffic.sh \
    --drones_num 2 \
    --evtols_num 1 \
    --experiment_name "simple_traffic"

# 中等复杂度
./learning/skrl/run_skrl_traffic.sh \
    --drones_num 5 \
    --evtols_num 2 \
    --experiment_name "medium_traffic"

# 高复杂度
./learning/skrl/run_skrl_traffic.sh \
    --drones_num 10 \
    --evtols_num 5 \
    --experiment_name "complex_traffic"
```

## 性能优化示例

### 10. GPU内存优化训练

```bash
# 适合GPU内存较小的情况
./learning/skrl/run_skrl_traffic.sh \
    --num_envs 512 \
    --config learning/skrl/configs/small_scale.yaml \
    --experiment_name "memory_optimized"
```

### 11. 高吞吐量训练

```bash
# 最大化训练吞吐量
./learning/skrl/run_skrl_traffic.sh \
    --num_envs 8192 \
    --config learning/skrl/configs/large_scale.yaml \
    --experiment_name "high_throughput"
```

## 调试和开发示例

### 12. 调试模式

```bash
# 快速调试配置
python learning/skrl/skrl_train_traffic.py \
    --config learning/skrl/configs/small_scale.yaml \
    --num_envs 64 \
    --total_timesteps 100000 \
    --experiment_name "debug_run"
```

### 13. 模型验证

```bash
# 验证模型训练流程
./learning/skrl/run_skrl_traffic.sh \
    --config learning/skrl/configs/small_scale.yaml \
    --num_envs 32 \
    --experiment_name "model_validation" \
    --use_wandb
```

## 长期训练示例

### 14. 24小时训练任务

```bash
# 设置长期训练任务（建议使用screen或tmux）
screen -S skrl_training
./learning/skrl/run_skrl_traffic.sh \
    --config learning/skrl/configs/large_scale.yaml \
    --experiment_name "long_term_training_$(date +%Y%m%d)" \
    --use_wandb \
    --wandb_project "long_term_experiments"
```

### 15. 多GPU训练准备

```bash
# 为多GPU训练设置大批次
./learning/skrl/run_skrl_traffic.sh \
    --num_envs 16384 \
    --config learning/skrl/configs/large_scale.yaml \
    --experiment_name "multi_gpu_ready"
```

## 对比基线示例

### 16. 与SB3对比基线

```bash
# SKRL基线实验，用于与原SB3代码对比
./learning/skrl/run_skrl_traffic.sh \
    --drones_num 5 \
    --evtols_num 1 \
    --learning_rate 4e-5 \
    --num_envs 1024 \
    --experiment_name "skrl_baseline_vs_sb3" \
    --use_wandb \
    --wandb_project "skrl_vs_sb3_comparison"
```

### 17. 不同随机种子重复实验

```bash
# 多次运行以验证结果稳定性
for seed in 42 123 456 789 999; do
    python learning/skrl/skrl_train_traffic.py \
        --config learning/skrl/configs/large_scale.yaml \
        --experiment_name "reproducibility_seed_${seed}" \
        --use_wandb \
        --wandb_project "reproducibility_study"
    # 注意：需要在配置中修改seed值
done
```

## 自定义配置示例

### 18. 创建自定义配置并运行

```bash
# 复制并修改配置文件
cp learning/skrl/skrl_config.yaml my_custom_config.yaml
# 编辑my_custom_config.yaml...

# 使用自定义配置运行
./learning/skrl/run_skrl_traffic.sh --config my_custom_config.yaml
```

## 批量实验管理

### 19. 使用配置模板进行批量实验

```bash
#!/bin/bash
# 批量实验脚本示例

experiments=(
    "small_scale:learning/skrl/configs/small_scale.yaml"
    "large_scale:learning/skrl/configs/large_scale.yaml"
)

for exp in "${experiments[@]}"; do
    name="${exp%%:*}"
    config="${exp##*:}"
    
    echo "Starting experiment: $name"
    ./learning/skrl/run_skrl_traffic.sh \
        --config "$config" \
        --experiment_name "${name}_$(date +%Y%m%d_%H%M%S)" \
        --use_wandb \
        --wandb_project "batch_experiments"
    
    echo "Completed experiment: $name"
    sleep 10  # 间隔时间
done
```

## 常用命令组合

### 20. 完整的研究实验命令

```bash
# 完整的研究实验设置
./learning/skrl/run_skrl_traffic.sh \
    --config learning/skrl/configs/large_scale.yaml \
    --experiment_name "gat_ppo_traffic_final" \
    --num_envs 4096 \
    --learning_rate 2e-5 \
    --drones_num 10 \
    --evtols_num 3 \
    --use_wandb \
    --wandb_project "traffic_gat_research" \
    2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt
```

这个命令会：
- 使用大规模配置
- 设置特定的实验名称
- 调整关键训练参数
- 启用Wandb记录
- 将所有输出保存到日志文件

---

**提示：**
- 使用`screen`或`tmux`进行长时间训练
- 定期检查GPU内存使用情况
- 监控Wandb或TensorBoard了解训练进度
- 保存重要的配置文件以便复现实验
