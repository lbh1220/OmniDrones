"""
SKRL兼容的Graph Attention Network Policy
基于现有的GAT网络结构，适配SKRL框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Tuple
from gymnasium import spaces

# SKRL imports
from skrl.models.torch import Model, CategoricalMixin, GaussianMixin
from skrl.utils import set_seed


def init(module, weight_init, bias_init, gain=1):
    """Initialize network weights"""
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class GraphAttentionLayer(nn.Module):
    """单个GAT层的实现，基于现有代码"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        # 使用正交初始化
        self.W = init_(nn.Linear(in_features, out_features, bias=False))
        self.a = init_(nn.Linear(2 * out_features, 1, bias=False))
        
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        x: 输入特征 [batch_size, N, in_features]
        adj: 邻接矩阵 [batch_size, N, N]
        """
        # 线性变换
        h = self.W(x)  # [batch_size, N, out_features]
        
        # 准备注意力计算
        batch_size, N = h.size(0), h.size(1)
        
        # 重复张量以计算所有节点对的注意力
        a_input = torch.cat([h.repeat_interleave(N, dim=1),
                           h.repeat(1, N, 1)], dim=2)  # [batch_size, N*N, 2*out_features]
        
        # 计算注意力系数
        e = self.leakyrelu(self.a(a_input).view(batch_size, N, N))  # [batch_size, N, N]
        
        # 掩码处理
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力
        h_prime = torch.bmm(attention, h)  # [batch_size, N, out_features]
        
        return h_prime


class MultiHeadGAT(nn.Module):
    """多头注意力GAT"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super().__init__()
        
        self.dropout = dropout
        
        # 多头注意力层
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)
        ])
        
        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        """
        x: [batch_size, N, nfeat]
        adj: [batch_size, N, N]
        """
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 多头注意力的结果拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 输出层
        x = F.elu(self.out_att(x, adj))
        
        return x


class GraphAttentionNetwork(nn.Module):
    """完整的GAT网络，基于现有的GAT_singlelayer和GAT_single"""
    def __init__(self, obs_space_dict, args):
        super().__init__()
        
        # 从观测空间获取维度信息
        robot_dim = args.robot_node_input_size
        human_dim = args.human_node_input_size
        max_human_num = args.max_human_num if hasattr(args, 'max_human_num') else 10
        
        encoding_dim = args.human_node_embedding_size
        hidden_dim = args.attention_size
        self.output_size = args.human_node_output_size
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        # Robot状态编码器
        self.robot_encoder = nn.Sequential(
            init_(nn.Linear(robot_dim, encoding_dim)),
            nn.ReLU()
        )
        
        # Human状态编码器
        self.human_encoder = nn.Sequential(
            init_(nn.Linear(human_dim, encoding_dim)),
            nn.ReLU()
        )
        
        # 不可见human的padding embedding
        self.padding_embedding = nn.Parameter(torch.zeros(1, max_human_num, encoding_dim))
        
        # GAT网络
        self.gat = MultiHeadGAT(
            nfeat=encoding_dim,
            nhid=hidden_dim // 4,  # 每个头的维度
            nclass=hidden_dim,
            dropout=0.1,
            alpha=0.2,
            nheads=4
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh()
        )

        # Critic网络
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

    def forward(self, ob, return_features=False):
        """
        Forward pass
        
        Args:
            ob: 观测字典，包含robot_node, human_nodes, visible_mask
            return_features: 是否返回特征向量
            
        Returns:
            如果return_features=False: (critic_value, actor_features)
            如果return_features=True: features
        """
        # 提取观测值
        robot_node = ob['robot_node']  # [batch_size, 1, input_dim]
        human_nodes = ob['human_nodes']  # [batch_size, max_human_num, input_dim]
        visible_mask = ob['visible_mask'].bool()  # [batch_size, max_human_num]
        
        # 编码robot状态
        robot_encoded = self.robot_encoder(robot_node)  # [batch_size, 1, encoding_dim]
        
        # 编码human状态并处理不可见的human
        human_nodes_encoded = self.human_encoder(human_nodes)  # [batch_size, max_human_num, encoding_dim]
        human_nodes_encoded = torch.where(
            visible_mask.unsqueeze(-1), 
            human_nodes_encoded,
            self.padding_embedding.expand(human_nodes_encoded.shape[0], -1, -1)
        )
        
        # 构建图的节点特征
        x = torch.cat([robot_encoded, human_nodes_encoded], dim=1)  # [batch_size, N, encoding_dim]
        
        # 构建邻接矩阵和掩码
        batch_size = x.size(0)
        N = x.size(1)
        adj = torch.ones(batch_size, N, N).to(x.device)
        
        # 注意力掩码处理
        mask_matrix = torch.ones(batch_size, N, N).to(x.device)
        mask_matrix[:, 1:, 1:] = torch.einsum('bi,bj->bij', visible_mask, visible_mask)
        mask_matrix[:, 0, 1:] = visible_mask
        mask_matrix[:, 1:, 0] = visible_mask
        adj = adj * mask_matrix
        
        # GAT处理
        node_features = self.gat(x, adj)
        
        # 只使用robot节点的特征进行决策
        robot_features = node_features[:, 0]  # [batch_size, hidden_dim]
        
        if return_features:
            return robot_features
        
        # 计算critic和actor输出
        hidden_critic = self.critic(robot_features)
        hidden_actor = self.actor(robot_features)
        
        critic_value = self.critic_linear(hidden_critic)
        
        return critic_value, hidden_actor


class GraphAttentionPolicy(GaussianMixin, Model):
    """
    SKRL兼容的Graph Attention Policy
    结合了GAT网络和SKRL的Gaussian policy
    """
    
    def __init__(self, observation_space, action_space, device, cfg=None, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions=True)
        
        # 解析观测空间
        if isinstance(observation_space, spaces.Dict):
            # 从观测空间推断参数
            robot_dim = observation_space.spaces['robot_node'].shape[-1]
            human_dim = observation_space.spaces['human_nodes'].shape[-1]
            max_human_num = observation_space.spaces['human_nodes'].shape[-2]
        else:
            # 默认值
            robot_dim = 7
            human_dim = 3
            max_human_num = 10
        
        # 创建args对象来兼容现有的GAT网络
        class Args:
            def __init__(self):
                self.robot_node_input_size = robot_dim
                self.human_node_input_size = human_dim  
                self.max_human_num = max_human_num
                self.human_node_embedding_size = 64
                self.attention_size = 256
                self.human_node_output_size = 256
        
        self.args = Args()
        
        # 动作空间维度
        if isinstance(action_space, spaces.Box):
            self.num_actions = action_space.shape[0]
        else:
            raise ValueError("Only continuous action spaces (Box) are supported")
        
        # 创建GAT网络
        self.gat_network = GraphAttentionNetwork(observation_space, self.args)
        
        # 动作输出层（均值）
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.01)
        self.mean_layer = init_(nn.Linear(self.args.human_node_output_size, self.num_actions))
        
        # 动作标准差（可学习参数）
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
        # 价值函数使用GAT网络的critic输出
        
    def compute(self, inputs, role):
        """
        SKRL模型的compute方法
        
        Args:
            inputs: 输入观测
            role: "policy" 或 "value"
            
        Returns:
            根据role返回相应的输出
        """
        if role == "policy":
            # 前向传播获取actor特征
            _, actor_features = self.gat_network(inputs)
            
            # 计算动作均值
            mean_actions = self.mean_layer(actor_features)
            
            # 计算标准差
            log_std = self.log_std_parameter.expand_as(mean_actions)
            std = torch.exp(log_std)
            
            return mean_actions, log_std, {"mean": mean_actions, "std": std}
            
        elif role == "value":
            # 前向传播获取critic值
            values, _ = self.gat_network(inputs)
            return values, {}
        
    def act(self, inputs, role):
        """执行动作采样"""
        if role == "policy":
            mean_actions, log_std, _ = self.compute(inputs, role)
            return self.sample(mean_actions, log_std), log_std, {}
        elif role == "value":
            return self.compute(inputs, role)


class GraphAttentionPolicyDiscrete(CategoricalMixin, Model):
    """
    SKRL兼容的离散动作GAT Policy
    """
    
    def __init__(self, observation_space, action_space, device, cfg=None, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self)
        
        # 解析观测空间
        if isinstance(observation_space, spaces.Dict):
            robot_dim = observation_space.spaces['robot_node'].shape[-1]
            human_dim = observation_space.spaces['human_nodes'].shape[-1]
            max_human_num = observation_space.spaces['human_nodes'].shape[-2]
        else:
            robot_dim = 7
            human_dim = 3
            max_human_num = 10
        
        # 创建args对象
        class Args:
            def __init__(self):
                self.robot_node_input_size = robot_dim
                self.human_node_input_size = human_dim  
                self.max_human_num = max_human_num
                self.human_node_embedding_size = 64
                self.attention_size = 256
                self.human_node_output_size = 256
        
        self.args = Args()
        
        # 动作空间维度
        if isinstance(action_space, spaces.Discrete):
            self.num_actions = action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")
        
        # 创建GAT网络
        self.gat_network = GraphAttentionNetwork(observation_space, self.args)
        
        # 动作输出层
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.01)
        self.action_layer = init_(nn.Linear(self.args.human_node_output_size, self.num_actions))
        
    def compute(self, inputs, role):
        """SKRL模型的compute方法"""
        if role == "policy":
            _, actor_features = self.gat_network(inputs)
            logits = self.action_layer(actor_features)
            return logits, {}
            
        elif role == "value":
            values, _ = self.gat_network(inputs)
            return values, {}
    
    def act(self, inputs, role):
        """执行动作采样"""
        if role == "policy":
            logits, _ = self.compute(inputs, role)
            return self.sample(logits), {}, {}
        elif role == "value":
            return self.compute(inputs, role)
