import torch.nn.functional as F

from .srnn_model import *
from .selfAttn_srnn_temp_node import *

class EndRNN_double(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN_double, self).__init__(args, edge=False)

        self.args = args
        
        # 重新定义这里的gru的维度
        self.gru = nn.GRU(args.human_node_embedding_size * 3, args.human_node_rnn_size)

        # 初始化 GRU 的参数
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size
        # self.half_embedding_size = self.embedding_size // 2
        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.evtol_edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)
        self.uav_edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_evtol, h_spatial_uav, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_evtol : output of the evtol-evtol attention module
        h_spatial_uav : output of the uav-uav attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        outputs, h_nodes 
            = self.EndRNN_node(robot_states, evtol_hidden_attn_weighted, uav_hidden_attn_weighted, hidden_states_node_RNNs, masks)
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded_evtol = self.relu(self.evtol_edge_attention_embed(h_spatial_evtol))
        h_edges_embedded_uav = self.relu(self.uav_edge_attention_embed(h_spatial_uav))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded_evtol, h_edges_embedded_uav), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new

class double_selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed network
    compared with selfAttn_merge_SRNN:
    I use two selfAttn network and two edge attn network
    to handle two types of aircraft: UAV and eVTOL
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(double_selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the final GRU module
        self.EndRNN_node = EndRNN_double(args)

        # initial robot-UAV attention module
        self.robot_uav_attn = EdgeAttention_M(args)

        # initial robot-eVTOL attention module
        self.robot_evtol_attn = EdgeAttention_M(args)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        # **检查并读取 robot_node_input_size 参数**
        robot_size = getattr(args, 'robot_node_input_size', 9)

        # robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim

        if self.args.use_self_attn:
            self.evtol_spatial_attn = SpatialEdgeSelfAttn(args)
            self.evtol_spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
            self.uav_spatial_attn = SpatialEdgeSelfAttn(args)
            self.uav_spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        else:
            raise NotImplementedError("The policy must use self attention")

        # # 这部分暂时没有用到, 因为我没有部署非sort的情况
        # dummy_human_mask = [0] * self.human_num
        # dummy_human_mask[0] = 1
        # if self.args.no_cuda:
        #     self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        # else:
        #     self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges_evtol = reshapeT(inputs['spatial_edges_evtol'], seq_length, nenv)
        spatial_edges_uav = reshapeT(inputs['spatial_edges_uav'], seq_length, nenv)
        # to prevent errors in old models that does not have sort_humans argument

        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_evtol_num = inputs['detected_evtol_num'].squeeze(-1).cpu().int()
            detected_uav_num = inputs['detected_uav_num'].squeeze(-1).cpu().int()
        else:
            raise NotImplementedError("The policy must use sort")
        

        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)


        # # 试试看如果没有这个, 程序会怎么样
        # if self.args.no_cuda:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        # else:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        
        # 这里没有像原来的代码那样, 部署非sort_humans和非use_self_attn的情况
        # evtol-evtol attention
        evtol_spatial_attn_out = self.evtol_spatial_attn(spatial_edges_evtol, detected_evtol_num).view(seq_length, nenv, self.human_num, -1)
        # uav-uav attention
        uav_spatial_attn_out = self.uav_spatial_attn(spatial_edges_uav, detected_uav_num).view(seq_length, nenv, self.human_num, -1)

        evtol_output_spatial = self.evtol_spatial_linear(evtol_spatial_attn_out)
        uav_output_spatial = self.uav_spatial_linear(uav_spatial_attn_out)
        # robot-human attention
        evtol_hidden_attn_weighted, _ = self.robot_evtol_attn(robot_states, evtol_output_spatial, detected_evtol_num)
 
        uav_hidden_attn_weighted, _ = self.robot_uav_attn(robot_states, uav_output_spatial, detected_uav_num)

        # Do a forward pass through GRU
        outputs, h_nodes \
            = self.EndRNN_node(robot_states, evtol_hidden_attn_weighted, uav_hidden_attn_weighted, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs


        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        # rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs


        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs
