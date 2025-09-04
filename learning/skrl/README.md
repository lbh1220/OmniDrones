# SKRL Training Framework for Traffic Environment

这个目录包含了从SB3迁移到SKRL架构的完整训练框架，专为交通环境设计，使用Graph Attention Network (GAT)作为特征提取器。

## 架构概览

### 主要组件

1. **`skrl_train_traffic.py`** - 主训练脚本，基于SKRL框架
2. **`skrl_models.py`** - GAT-based策略网络实现
3. **`skrl_callbacks.py`** - 自定义回调函数（success rate监控、日志记录等）
4. **`skrl_config.yaml`** - 训练配置文件

### 相比SB3的优势

- ✅ **原生GPU训练**: SKRL提供完全GPU原生的训练pipeline
- ✅ **更好的性能**: 避免了SB3的CPU-GPU数据传输瓶颈
- ✅ **现代化架构**: 基于Isaac Lab推荐的训练框架
- ✅ **灵活配置**: 使用YAML配置文件管理所有参数
- ✅ **完整回调系统**: Success rate监控、TensorBoard、Wandb集成
- ✅ **Graph Attention Network**: 保留了原有的GAT特征提取能力

## 快速开始

### 1. 安装依赖

```bash
# 安装SKRL
./isaaclab.sh -i skrl

# 或者手动安装
pip install skrl[torch]
```

### 2. 基本训练

```bash
# 使用默认配置
cd /path/to/OmniDrones
python learning/skrl/skrl_train_traffic.py

# 指定自定义配置
python learning/skrl/skrl_train_traffic.py --config my_config.yaml

# 覆盖关键参数
python learning/skrl/skrl_train_traffic.py \
    --num_envs 2048 \
    --learning_rate 3e-5 \
    --drones_num 10 \
    --evtols_num 2
```

### 3. 启用Wandb记录

```bash
python learning/skrl/skrl_train_traffic.py \
    --use_wandb \
    --wandb_project "my_traffic_project" \
    --wandb_run_name "gat_ppo_experiment_1"
```

## 配置说明

### 配置文件结构

```yaml
# 环境设置
environment:
  name: "TrafficEnv"
  num_envs: 1024
  max_episode_length: 1000

# PPO智能体配置
agent:
  rollouts: 64                    # 每次rollout的步数
  mini_batches: 16               # mini-batch数量
  learning_epochs: 4             # 每次更新的学习epoch数
  learning_rate: 4.0e-5          # 学习率
  ratio_clip: 0.1                # PPO策略裁剪
  entropy_loss_scale: 0.001      # 熵奖励系数

# GAT网络架构
network:
  human_node_embedding_size: 64   # 节点嵌入维度
  attention_size: 256            # 注意力层大小
  num_heads: 4                   # 注意力头数量
  dropout: 0.1                   # Dropout率

# 训练参数
training:
  total_timesteps: 20000000      # 总训练步数
  log_interval: 10               # 日志记录间隔
  save_interval: 1000            # 模型保存间隔

# 交通环境专用参数
traffic:
  drones_num: 5                  # 无人机数量
  evtols_num: 1                  # 电动飞行器数量
  drone_future_penalty: 0.0      # 无人机未来惩罚
  evtol_future_penalty: 0.0      # eVTOL未来惩罚

# 日志配置
logging:
  use_tensorboard: true          # 启用TensorBoard
  use_wandb: false              # 启用Wandb
  wandb_project: "traffic_skrl"  # Wandb项目名
```

### 关键参数说明

- **rollouts**: 决定每次更新前收集多少步的经验
- **mini_batches**: 每次更新时将经验分成多少个mini-batch
- **learning_epochs**: 每批经验要训练多少个epoch
- **ratio_clip**: PPO的策略裁剪参数，防止策略更新过大
- **entropy_loss_scale**: 熵奖励系数，鼓励探索

## Graph Attention Network架构

### 网络结构

```
输入观测 → Robot/Human编码器 → GAT多头注意力 → Actor/Critic输出
```

1. **节点编码**: 分别编码robot状态和human状态
2. **图注意力**: 使用多头注意力机制处理节点间关系
3. **掩码处理**: 处理不可见的human节点
4. **决策输出**: 基于robot节点特征进行动作决策

### GAT参数

- `num_heads`: 注意力头数量，控制模型的表达能力
- `attention_size`: 注意力层的隐藏维度
- `dropout`: 防止过拟合的dropout率
- `alpha`: LeakyReLU的负斜率参数

## 回调系统

### Success Rate Callback

监控训练过程中的成功率、碰撞率和超时率：

```python
SuccessRateCallback(
    check_freq=10,              # 每10个rollout检查一次
    queue_size=1000,            # 维护最近1000个episode的统计
    success_threshold=0.8       # 成功率阈值
)
```

### TensorBoard Logging

自动记录以下指标：
- Training losses (policy, value, entropy)
- Environment rewards
- Learning rate
- Success/collision/timeout rates

### Wandb Integration

支持完整的实验跟踪：
- 超参数记录
- 实时指标监控  
- 模型工件保存
- 实验对比

## 模型保存

### 自动保存策略

1. **最佳成功率模型**: 当成功率达到新高时自动保存
2. **最佳奖励模型**: 当平均奖励达到新高时自动保存
3. **定期检查点**: 按配置的间隔定期保存
4. **最终模型**: 训练结束时保存

### 保存格式

```python
{
    'policy_state_dict': ...,     # 策略网络参数
    'value_state_dict': ...,      # 价值网络参数（如果分离）
    'optimizer_state_dict': ...,  # 优化器状态
    'timestep': ...,              # 当前时间步
    'success_rate': ...,          # 当前成功率
    'mean_reward': ...            # 平均奖励
}
```

## 与SB3代码的对应关系

| SB3组件 | SKRL组件 | 文件位置 |
|---------|----------|----------|
| `CustomPPO` | `PPO` + config | `skrl_train_traffic.py` |
| `CustomSelfAttnPolicy` | `GraphAttentionPolicy` | `skrl_models.py` |
| `SucessRateCallback` | `SuccessRateCallback` | `skrl_callbacks.py` |
| `ArgsConfig` | YAML config | `skrl_config.yaml` |
| Manual logging | `TensorboardLogger`, `WandbLogger` | `skrl_callbacks.py` |

## 性能优化建议

### 内存优化
- 适当调整`num_envs`和`rollouts`的平衡
- 较大的`num_envs`提高并行度，较大的`rollouts`提高样本效率

### 训练稳定性
- 使用较小的`learning_rate`和`ratio_clip`提高稳定性
- 适当的`entropy_loss_scale`维持探索

### GAT网络调优
- `num_heads`通常在4-8之间效果较好
- `dropout`在0.1-0.3之间防止过拟合
- `attention_size`影响模型容量，需要平衡性能和计算成本

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`num_envs`
   - 减少GAT的`attention_size`
   - 使用梯度累积

2. **训练不稳定**
   - 降低`learning_rate`
   - 增加`mini_batches`
   - 检查`ratio_clip`设置

3. **成功率不提升**
   - 检查奖励函数设计
   - 调整GAT网络架构
   - 增加训练时间

### 调试模式

```bash
# 启用详细日志
python learning/skrl/skrl_train_traffic.py --verbose

# 使用小规模测试
python learning/skrl/skrl_train_traffic.py \
    --num_envs 64 \
    --total_timesteps 100000
```

## 扩展开发

### 添加新的回调

```python
class CustomCallback(BaseCallback):
    def on_timestep_end(self, trainer, **kwargs):
        # 自定义逻辑
        pass
```

### 修改GAT架构

在`skrl_models.py`中的`GraphAttentionNetwork`类中添加新的层或修改现有结构。

### 集成新的环境

修改`create_env`函数以支持新的环境类型。

## 致谢

本框架基于以下开源项目：
- [SKRL](https://github.com/Toni-SM/skrl) - 强化学习库
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - 仿真平台
- 原有的SB3训练代码作为迁移基础
