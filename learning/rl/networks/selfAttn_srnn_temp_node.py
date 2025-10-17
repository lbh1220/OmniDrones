import torch.nn.functional as F

from .srnn_model import *

class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, args):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.args = args
        self.input_size = getattr(self.args, 'human_human_edge_input_size', 2)
        self.num_attn_heads=8
        self.attn_size=512


        # Per-type encoders: evtol(0) and drone(1)
        self.drone_embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, self.attn_size), nn.ReLU()
        )
        self.evtol_embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, self.attn_size), nn.ReLU()
        )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask

    
    def forward(self, inp, each_seq_len, spatial_types=None):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, D]
        seq_len, nenv, max_human_num, _ = inp.size()
        # Always use non-sort mask input
        attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)

        # Encode with per-type encoders
        flat_inp = inp.view(seq_len*nenv, max_human_num, -1)
        if spatial_types is not None:
            flat_types = spatial_types.view(seq_len*nenv, max_human_num)
            drone_mask = (flat_types == 1).unsqueeze(-1).float()
            evtol_mask = (flat_types == 2).unsqueeze(-1).float()
            emb_drone = self.drone_embedding_layer(flat_inp)
            emb_evtol = self.evtol_embedding_layer(flat_inp)
            input_emb = emb_drone * drone_mask + emb_evtol * evtol_mask
        else:
            # Fallback: treat all as drone type
            input_emb = self.drone_embedding_layer(flat_inp)
        input_emb = input_emb.view(seq_len*nenv, max_human_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        # 这个torch.transpose, 和np.transpose的用法不同, 表示交换两个通道, 变成(max_human_num, seq_len*nenv, 512)
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        #z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        return z



class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        # 上面的定义是将edge_layer定义为一个列表, 但是实际上长度只有1, 然后这里定义的attention_head, 这样后面forward的循环就不会越界
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed # 这里是一个元素级相称
        attn = torch.sum(attn, dim=3) # 然后sum, 相当与实现了点乘

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size) # 这一步看起来只是一个缩放
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None: # 将attn_mask为0的地方, 换成了极小值-1e9
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, human_masks)
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
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
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

        self.use_rnn = args.use_rnn



    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        outputs, h_nodes 
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        # self.use_rnn = True
        if self.use_rnn:
            x, h_new = self._forward_gru(concat_encoded, h, masks)

            outputs = self.output_linear(x)
        else:
            outputs = self.output_linear(concat_encoded)
            h_new = h


        return outputs, h_new

class selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args
        
        # 使用观测空间里的 traffic slots 数（包含两类混合后的最大数）
        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)

        # Initialize attention modules (type-split)
        self.attn_drone = EdgeAttention_M(args)
        self.attn_evtol = EdgeAttention_M(args)
        # fuse two streams back to 256 using a linear projection


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size
        self.type_fuse = nn.Sequential(
            init_(nn.Linear(self.human_human_edge_rnn_size * 2, self.human_human_edge_rnn_size)), nn.ReLU())

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # self.actor = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        # self.critic = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        if hasattr(self.args, 'robot_node_input_size'):
            robot_size = getattr(self.args, 'robot_node_input_size', 9)
        else:
            robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        # Always use self attention path
        self.spatial_attn = SpatialEdgeSelfAttn(args)
        self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())


        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())



    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        """
        Convert detected counts (lengths) to visibility masks.
        Returns mask of shape [seq_len, nenv, max_human_num] with 1 for visible and 0 for padding.
        """
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        idx = each_seq_len.view(-1).long()
        mask[torch.arange(seq_len * nenv), idx] = 1.0
        mask = torch.logical_not(mask.cumsum(dim=1))
        mask = mask[:, :-1]  # remove sentinel
        mask = mask.view(seq_len, nenv, max_human_num).float()
        return mask

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv
            # add this to support predict_values when TimeLimit.truncated in sb3
            if 'robot_node' in inputs:
                nenv = inputs['robot_node'].shape[0]

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)

        # Prefer visible_masks; fallback to detected_human_num -> mask
        if 'visible_masks' in inputs:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
        elif 'detected_human_num' in inputs:
            lengths = inputs['detected_human_num'].squeeze(-1)
            human_masks = self.create_attn_mask(lengths, seq_length, nenv, self.human_num)
        else:
            raise ValueError("No visible_masks or detected_human_num found in inputs")
        # ensure at least one visible per (seq, env)
        human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask
        spatial_types = reshapeT(inputs['spatial_types'], seq_length, nenv).long() # [seq_len, nenv, max_human_num]


        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)


        # if self.args.no_cuda:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        # else:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)


        # human-human attention (always self-attn + mask)
        use_type_split = getattr(self.args, 'use_type_split_attn', False)
        if use_type_split:
            spatial_attn_out = self.spatial_attn(spatial_edges, human_masks, spatial_types=spatial_types).view(seq_length, nenv, self.human_num, -1)
        else:
            spatial_attn_out = self.spatial_attn(spatial_edges, human_masks, spatial_types=None).view(seq_length, nenv, self.human_num, -1)
        output_spatial = self.spatial_linear(spatial_attn_out)

        # type masks (1=drone, 2=evtol; 0=dummy)
        type_mask_drone = (spatial_types == 1).float()
        type_mask_evtol = (spatial_types == 2).float()

        # Decide single-stream vs dual-stream attention
        use_type_split = getattr(self.args, 'use_type_split_attn', False)
        if use_type_split:
            mask_drone = human_masks * type_mask_drone
            mask_evtol = human_masks * type_mask_evtol

            hidden_drone, _ = self.attn_drone(robot_states, output_spatial, mask_drone)
            hidden_evtol, _ = self.attn_evtol(robot_states, output_spatial, mask_evtol)

            # fuse streams: [seq_len, nenv, 1, 256] concat -> [seq_len, nenv, 1, 512] -> project 256
            hidden_attn_weighted = torch.cat([hidden_drone, hidden_evtol], dim=-1)
            hidden_attn_weighted = self.type_fuse(hidden_attn_weighted)
        else:
            # single stream (no type split): reuse attn_drone with full mask
            hidden_attn_weighted, _ = self.attn_drone(robot_states, output_spatial, human_masks)

        # hidden_attn_weighted computed above based on use_type_split


        # Do a forward pass through GRU
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)


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


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))