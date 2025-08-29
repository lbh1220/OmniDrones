import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """单个GAT层的实现"""
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

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super(GAT, self).__init__()
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

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

class GAT_singlelayer(nn.Module):
    def __init__(self, obs_space_dict, args):
        """
        Args:
            input_dim: 输入状态维度 (5 or 7)
            hidden_dim: GAT中每个注意力头的隐藏层维度，应该是 nheads(4) 的倍数
            output_dim: 输出动作维度
            encoding_dim: 编码后的维度，建议与hidden_dim相同或更小
        """
        super().__init__()
        
        # 初始化函数
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.args = args
        self.is_recurrent = False
        self.output_size = 64

        input_dim_robot = obs_space_dict['robot_node'].shape[1]
        input_dim_human = obs_space_dict['human_nodes'].shape[1]
        hidden_dim = 64
        # output_dim = 2
        encoding_dim = 64

        nheads = 4
        assert hidden_dim % nheads == 0, f"hidden_dim({hidden_dim})应该是注意力头数量({nheads})的倍数"
        
        # 参考原代码的设计，使用更合理的编码器结构
        self.robot_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_robot, encoding_dim)),
            nn.ReLU()
        )
        
        self.human_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_human, encoding_dim)),
            nn.ReLU()
        )
        
        # 可学习的填充嵌入
        self.padding_embedding = nn.Parameter(torch.randn(1, 1, encoding_dim))
        
        # GAT网络
        self.gat = GAT(nfeat=encoding_dim,
                      nhid=hidden_dim, 
                      nclass=hidden_dim,
                      dropout=0.2,
                      nheads=4)
        
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

    def forward(self, ob, rnn_hxs, masks):
        # 提取观测值（已经在环境中归一化）
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
        robot_features = node_features[:, 0]

        hidden_critic = self.critic(robot_features)
        hidden_actor = self.actor(robot_features)
        
        # 输出动作
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class GAT_single(nn.Module):
    def __init__(self, obs_space_dict, args):
        """
        Args:
            input_dim: 输入状态维度 (5 or 7)
            hidden_dim: GAT中每个注意力头的隐藏层维度，应该是 nheads(4) 的倍数
            output_dim: 输出动作维度
            encoding_dim: 编码后的维度，建议与hidden_dim相同或更小
        """
        super().__init__()
        
        # 初始化函数
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.args = args
        self.is_recurrent = False
        if hasattr(args, 'gat_layer_num'):  # 如果设置了gat_layer_num，则使用设置的层数，否则使用1层
            self.layer_num = args.gat_layer_num
        else:
            self.layer_num = 1
        self.output_size = 64

        input_dim_robot = obs_space_dict['robot_node'].shape[1]
        input_dim_human = obs_space_dict['human_nodes'].shape[1]
        hidden_dim = 64
        # output_dim = 2
        encoding_dim = 64

        nheads = 4
        assert hidden_dim % nheads == 0, f"hidden_dim({hidden_dim})应该是注意力头数量({nheads})的倍数"
        
        # 参考原代码的设计，使用更合理的编码器结构
        self.robot_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_robot, encoding_dim)),
            nn.ReLU()
        )
        
        self.human_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_human, encoding_dim)),
            nn.ReLU()
        )
        
        # 可学习的填充嵌入
        self.padding_embedding = nn.Parameter(torch.randn(1, 1, encoding_dim))
        
        # GAT网络
        self.gat = GAT(nfeat=encoding_dim,
                      nhid=hidden_dim, 
                      nclass=hidden_dim,
                      dropout=0.2,
                      nheads=4)
        
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

    def forward(self, ob, rnn_hxs, masks):
        # 提取观测值（已经在环境中归一化）
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
        h_list = []
        for i in range(self.layer_num):
            node_features = self.gat(x, adj)
            h_list.append(node_features[:, 0])
        # # 使用多层的均值作为robot节点的特征
        # robot_features = torch.mean(torch.stack(h_list), dim=0)
        # 使用残差连接融合多层特征：依次累加各层输出
        robot_features = torch.stack(h_list, dim=0).sum(dim=0)

        hidden_critic = self.critic(robot_features)
        hidden_actor = self.actor(robot_features)
        
        # 输出动作
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class HeterogeneousGAT_single(nn.Module):
    def __init__(self, obs_space_dict, args):
        super().__init__()
        
        # 初始化函数
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.args = args
        self.is_recurrent = False
        if hasattr(args, 'gat_layer_num'):  # 如果设置了gat_layer_num，则使用设置的层数，否则使用1层
            self.layer_num = args.gat_layer_num
        else:
            self.layer_num = 1
        self.output_size = 64

        input_dim_robot = obs_space_dict['robot_node'].shape[1]
        input_dim_human = obs_space_dict['human_nodes'].shape[1]
        input_dim_evtol = obs_space_dict['evtol_nodes'].shape[1]
        input_dim_uav = obs_space_dict['uav_nodes'].shape[1]
        hidden_dim = 64
        # output_dim = 2
        encoding_dim = 64

        nheads = 4
        assert hidden_dim % nheads == 0, f"hidden_dim({hidden_dim})应该是注意力头数量({nheads})的倍数"
        
        # 参考原代码的设计，使用更合理的编码器结构
        self.robot_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_robot, encoding_dim)),
            nn.ReLU()
        )
        

        self.evtol_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_evtol, encoding_dim)),
            nn.ReLU()
        )
        self.uav_encoder = nn.Sequential(
            init_(nn.Linear(input_dim_uav, encoding_dim)),
            nn.ReLU()
        )
        
        
        # 可学习的填充嵌入
        self.padding_embedding = nn.Parameter(torch.randn(1, 1, encoding_dim))
        
        # GAT网络
        self.gat_evtol = GAT(nfeat=encoding_dim,
                      nhid=hidden_dim, 
                      nclass=hidden_dim,
                      dropout=0.2,
                      nheads=4)
        
        self.gat_uav = GAT(nfeat=encoding_dim,
                      nhid=hidden_dim, 
                      nclass=hidden_dim,
                      dropout=0.2,
                      nheads=4)
        

        # TODO: 考虑增加actor和critic的层数
        self.actor = nn.Sequential(
            init_(nn.Linear(encoding_dim + hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(encoding_dim + hidden_dim, self.output_size)), 
            nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

    def forward(self, ob, rnn_hxs, masks):
        # 提取观测值（已经在环境中归一化）
        robot_node = ob['robot_node']  # [batch_size, 1, input_dim]
        evtol_nodes = ob['evtol_nodes']  # [batch_size, max_human_num, input_dim]
        uav_nodes = ob['uav_nodes']  # [batch_size, max_human_num, input_dim]
        evtol_mask = ob['evtol_mask'].bool()  # [batch_size, max_human_num]
        uav_mask = ob['uav_mask'].bool()  # [batch_size, max_human_num]
        
        # 编码robot状态
        robot_encoded = self.robot_encoder(robot_node)  # [batch_size, 1, encoding_dim]
        
        # 编码human状态并处理不可见的human
        evtol_nodes_encoded = self.evtol_encoder(evtol_nodes)  # [batch_size, max_human_num, encoding_dim]
        evtol_nodes_encoded = torch.where(
            evtol_mask.unsqueeze(-1), 
            evtol_nodes_encoded,
            self.padding_embedding.expand(evtol_nodes_encoded.shape[0], -1, -1)
        )
        uav_nodes_encoded = self.uav_encoder(uav_nodes)  # [batch_size, max_human_num, encoding_dim]
        uav_nodes_encoded = torch.where(
            uav_mask.unsqueeze(-1), 
            uav_nodes_encoded,
            self.padding_embedding.expand(uav_nodes_encoded.shape[0], -1, -1)
        )
        # 构建图的节点特征
        
        x_evtol = torch.cat([robot_encoded, evtol_nodes_encoded], dim=1)  # [batch_size, N, encoding_dim]
        x_uav = torch.cat([robot_encoded, uav_nodes_encoded], dim=1)  # [batch_size, N, encoding_dim]
        # 构建邻接矩阵和掩码
        batch_size = x_evtol.size(0)
        adj_evtol = torch.ones(batch_size, x_evtol.size(1), x_evtol.size(1)).to(x_evtol.device)
        adj_uav = torch.ones(batch_size, x_uav.size(1), x_uav.size(1)).to(x_uav.device)
        
        # 注意力掩码处理
        mask_matrix_evtol = torch.ones(batch_size, x_evtol.size(1), x_evtol.size(1)).to(x_evtol.device)
        mask_matrix_evtol[:, 1:, 1:] = torch.einsum('bi,bj->bij', evtol_mask, evtol_mask)
        mask_matrix_evtol[:, 0, 1:] = evtol_mask
        mask_matrix_evtol[:, 1:, 0] = evtol_mask
        adj_evtol = adj_evtol * mask_matrix_evtol
        mask_matrix_uav = torch.ones(batch_size, x_uav.size(1), x_uav.size(1)).to(x_uav.device)
        mask_matrix_uav[:, 1:, 1:] = torch.einsum('bi,bj->bij', uav_mask, uav_mask)
        mask_matrix_uav[:, 0, 1:] = uav_mask
        mask_matrix_uav[:, 1:, 0] = uav_mask
        adj_uav = adj_uav * mask_matrix_uav
        
        
        
        # GAT处理
        # 
        h_list_evtol = []
        h_list_uav = []
        for i in range(self.layer_num):
            node_features_evtol = self.gat_evtol(x_evtol, adj_evtol)
            node_features_uav = self.gat_uav(x_uav, adj_uav)
            h_list_evtol.append(node_features_evtol[:, 0])
            h_list_uav.append(node_features_uav[:, 0])
        # 使用残差连接融合多层特征：依次累加各层输出
        robot_features_evtol = torch.stack(h_list_evtol, dim=0).sum(dim=0)
        robot_features_uav = torch.stack(h_list_uav, dim=0).sum(dim=0)
        # 使用mean aggregation融合两个特征
        robot_features = (robot_features_evtol + robot_features_uav) / 2
        
        robot_encoded = robot_encoded.squeeze(1)
        robot_features = torch.cat([robot_encoded, robot_features], dim=1)

        hidden_critic = self.critic(robot_features)
        hidden_actor = self.actor(robot_features)
        
        # 输出动作
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs