import torch.nn.functional as F

from .srnn_model import *
from .selfAttn_srnn_temp_node import EndRNN

class HeterSpatialEdgeSelfAttn(nn.Module):
    """
    Class for the heterogeneous human-human attention,
    uses different encoders for different agent types (evtol vs uav)
    """
    def __init__(self, args):
        super(HeterSpatialEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        if args.env_name in ['CrowdSimPred-v0', 
                             'CrowdSimPredRealGST-v0', 
                             'AirspaceSimPred-v0', 
                             'AirspaceSimPredReward-v0', 
                             'AirspaceSimCourse-v0',
                             'AirspaceSimLarge-v0',
                             'AirspaceSimCity-v0',
                             'AirspaceSimCityDRLVO-v0']:
            self.input_size = self.args.human_human_edge_input_size
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2 # 4
        else:
            raise NotImplementedError
            
        self.num_attn_heads = 8
        self.attn_size = 512
        self.num_agent_types = 2  # evtol (0) and uav (1)

        # Heterogeneous embedding layers for different agent types
        self.heter_embedding_layers = nn.ModuleList()
        for agent_type in range(self.num_agent_types):
            self.heter_embedding_layers.append(
                nn.Sequential(
                    nn.Linear(self.input_size, 128), nn.ReLU(),
                    nn.Linear(128, self.attn_size), nn.ReLU()
                )
            )

        # Heterogeneous Q, K, V linear layers for different agent types
        self.heter_q_linears = nn.ModuleList()
        self.heter_k_linears = nn.ModuleList()
        self.heter_v_linears = nn.ModuleList()
        
        for agent_type in range(self.num_agent_types):
            self.heter_q_linears.append(nn.Linear(self.attn_size, self.attn_size))
            self.heter_k_linears.append(nn.Linear(self.attn_size, self.attn_size))
            self.heter_v_linears.append(nn.Linear(self.attn_size, self.attn_size))

        # Multi-head self attention (shared across types)
        self.multihead_attn = torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask

    def heter_encode_qkv(self, inp, agent_types):
        """
        Encode QKV using heterogeneous encoders based on agent types
        """
        seq_len, nenv, max_human_num, feature_dim = inp.size()
        
        # Flatten for processing
        inp_flat = inp.view(seq_len * nenv, max_human_num, feature_dim)
        agent_types_flat = agent_types.view(seq_len * nenv, max_human_num)
        
        # Initialize output tensors
        device = inp.device
        embedded_features = torch.zeros(seq_len * nenv, max_human_num, self.attn_size, device=device)
        q_features = torch.zeros(seq_len * nenv, max_human_num, self.attn_size, device=device)
        k_features = torch.zeros(seq_len * nenv, max_human_num, self.attn_size, device=device)
        v_features = torch.zeros(seq_len * nenv, max_human_num, self.attn_size, device=device)
        
        # Process each agent type separately
        for agent_type in range(self.num_agent_types):
            # Create mask for current agent type
            type_mask = (agent_types_flat == agent_type)
            
            if type_mask.any():
                # Extract features for current agent type
                type_features = inp_flat[type_mask]  # [num_agents_of_type, feature_dim]
                
                # Apply type-specific embedding
                type_embedded = self.heter_embedding_layers[agent_type](type_features)
                
                # Apply type-specific Q, K, V transforms
                type_q = self.heter_q_linears[agent_type](type_embedded)
                type_k = self.heter_k_linears[agent_type](type_embedded)
                type_v = self.heter_v_linears[agent_type](type_embedded)
                
                # Place back into full tensors
                embedded_features[type_mask] = type_embedded
                q_features[type_mask] = type_q
                k_features[type_mask] = type_k
                v_features[type_mask] = type_v
        
        return embedded_features, q_features, k_features, v_features
    
    def forward(self, inp, each_seq_len, agent_types):
        '''
        Forward pass for the heterogeneous model
        params:
        inp : input edge features
        each_seq_len: sequence lengths or masks
        agent_types: agent type indices [seq_len, nenv, max_human_num]
        '''
        seq_len, nenv, max_human_num, _ = inp.size()
        
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)
            attn_mask = attn_mask.squeeze(1)
        else:
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)

        # Use heterogeneous encoding
        embedded_features, q, k, v = self.heter_encode_qkv(inp, agent_types)
        
        # Transpose for multihead attention
        q = torch.transpose(q, dim0=0, dim1=1)  # (max_human_num, seq_len*nenv, attn_size)
        k = torch.transpose(k, dim0=0, dim1=1)
        v = torch.transpose(v, dim0=0, dim1=1)

        # Apply multihead attention
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))
        z = torch.transpose(z, dim0=0, dim1=1)  # Back to (seq_len*nenv, max_human_num, attn_size)
        
        return z


class HeterEdgeAttention_M(nn.Module):
    '''
    Class for the heterogeneous robot-human attention module
    Uses different encoders for different agent types
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        '''
        super(HeterEdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size
        self.num_agent_types = 2  # evtol (0) and uav (1)

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.ModuleList()
        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Heterogeneous spatial edge layers for different agent types
        self.heter_spatial_edge_layers = nn.ModuleList()
        for agent_type in range(self.num_agent_types):
            self.heter_spatial_edge_layers.append(
                nn.Linear(self.human_human_edge_rnn_size, self.attention_size)
            )

        # Number of agents and attention heads
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def heter_encode_spatial(self, h_spatials, agent_types):
        """
        Encode spatial features using heterogeneous encoders based on agent types
        """
        seq_len, nenv, max_human_num, h_size = h_spatials.size()
        
        # Flatten for processing
        h_spatials_flat = h_spatials.view(seq_len * nenv, max_human_num, h_size)
        agent_types_flat = agent_types.view(seq_len * nenv, max_human_num)
        
        # Initialize output tensor - use same device as input
        device = h_spatials.device
        spatial_embed = torch.zeros(seq_len * nenv, max_human_num, self.attention_size, device=device)
        
        # Process each agent type separately
        for agent_type in range(self.num_agent_types):
            # Create mask for current agent type
            type_mask = (agent_types_flat == agent_type)
            
            if type_mask.any():
                # Extract features for current agent type
                type_features = h_spatials_flat[type_mask]  # [num_agents_of_type, h_size]
                
                # Apply type-specific spatial embedding
                type_spatial_embed = self.heter_spatial_edge_layers[agent_type](type_features)
                
                # Place back into full tensor
                spatial_embed[type_mask] = type_spatial_embed
        
        # Reshape back to original dimensions
        spatial_embed = spatial_embed.view(seq_len, nenv, max_human_num, self.attention_size)
        return spatial_embed

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()
        attn = temporal_embed * spatial_embed  # Element-wise multiplication
        attn = torch.sum(attn, dim=3)  # Sum to get attention scores

        # Temperature scaling
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Compute weighted value
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2, 1)

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)
        weighted_value = torch.bmm(h_spatials, attn)

        # Reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)
        return weighted_value, attn

    def forward(self, h_temporal, h_spatials, each_seq_len, agent_types):
        '''
        Forward pass for the heterogeneous model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node
        each_seq_len: sequence lengths or masks
        agent_types: agent type indices [seq_len, nenv, max_human_num]
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list = [], []
        for i in range(self.num_attention_head):
            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)

            # Embed the spatial edgeRNN hidden states using heterogeneous encoders
            spatial_embed = self.heter_encode_spatial(h_spatials, agent_types)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len

            weighted_value, attn = self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]


class heter_selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed heterogeneous network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(heter_selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args = args
        
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

        # Initialize heterogeneous attention module
        self.heter_attn = HeterEdgeAttention_M(args)

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
        
        if hasattr(self.args, 'robot_node_input_size'):
            robot_size = getattr(self.args, 'robot_node_input_size', 9)
        else:
            robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU())
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))

        if self.args.use_self_attn:
            self.heter_spatial_attn = HeterSpatialEdgeSelfAttn(args)
            self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        else:
            self.spatial_linear = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
                                                init_(nn.Linear(128, 256)), nn.ReLU())

        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

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
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        agent_types = reshapeT(inputs['agent_types'], seq_length, nenv)

        # Convert agent_types to appropriate tensor type and device
        if not torch.is_tensor(agent_types):
            agent_types = torch.from_numpy(agent_types).long()
        else:
            agent_types = agent_types.long()
            
        # Ensure agent_types is on the same device as spatial_edges
        if agent_types.device != spatial_edges.device:
            agent_types = agent_types.to(spatial_edges.device)

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float()
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask

        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        # Attention modules
        if self.args.sort_humans:
            # Human-human heterogeneous attention
            if self.args.use_self_attn:
                spatial_attn_out = self.heter_spatial_attn(spatial_edges, detected_human_num, agent_types).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # Robot-human heterogeneous attention
            hidden_attn_weighted, _ = self.heter_attn(robot_states, output_spatial, detected_human_num, agent_types)
        else:
            # Human-human heterogeneous attention
            if self.args.use_self_attn:
                spatial_attn_out = self.heter_spatial_attn(spatial_edges, human_masks, agent_types).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # Robot-human heterogeneous attention
            hidden_attn_weighted, _ = self.heter_attn(robot_states, output_spatial, human_masks, agent_types)

        # Do a forward pass through GRU
        outputs, h_nodes = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs

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
