import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn.functional as F


class TrafficEdgeSelfAttn(nn.Module):
    """
    Traffic-traffic self attention (multi-head),
    based on https://arxiv.org/abs/1706.03762
    """
    def __init__(self, input_size=13, attn_size=512, num_attn_heads=8):
        super(TrafficEdgeSelfAttn, self).__init__()
        
        self.input_size = input_size
        self.num_attn_heads = num_attn_heads
        self.attn_size = attn_size

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
        self.multihead_attn = torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    def forward(self, inp, visible_mask, traffic_types=None):
        """
        Forward pass
        params:
        inp : input edge features [batch_size, max_traffic_num, feature_dim]
        """
        batch_size, max_traffic_num, _ = inp.size()
        
        # Create attention mask
        attn_mask = visible_mask.reshape(batch_size, max_traffic_num)

        if traffic_types is not None:
            flat_types = traffic_types.view(batch_size, max_traffic_num)
            drone_mask = (flat_types == 1).unsqueeze(-1).float()
            evtol_mask = (flat_types == 2).unsqueeze(-1).float()
            emb_drone = self.drone_embedding_layer(inp)
            emb_evtol = self.evtol_embedding_layer(inp)
            input_emb = emb_drone * drone_mask + emb_evtol * evtol_mask
        else:
            # Fallback: treat all as drone type
            input_emb = self.drone_embedding_layer(inp)
        input_emb = input_emb.view(batch_size, max_traffic_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        # Apply linear transformations
        q = self.q_linear(input_emb)
        k = self.k_linear(input_emb)
        v = self.v_linear(input_emb)

        # Apply multi-head attention
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))
        z=torch.transpose(z, dim0=0, dim1=1)  # [batch_size, max_traffic_num, attn_size]
        
        return z


class EgoTrafficAttention(nn.Module):
    """
    Ego-traffic attention module
    """
    def __init__(self, input_feature_size=256, attention_size=64):
        super(EgoTrafficAttention, self).__init__()
        
        self.input_feature_size = input_feature_size
        self.attention_size = attention_size

        # Linear layers to embed temporal and spatial edge states
        self.temporal_edge_layer = nn.Linear(self.input_feature_size, self.attention_size)
        self.spatial_edge_layer = nn.Linear(self.input_feature_size, self.attention_size)

        self.agent_num = 1
        self.num_attention_head = 1

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        batch_size, num_edges, h_size = h_spatials.size()
        
        # Element-wise multiplication followed by sum (dot product)
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=2)  # [batch_size, num_edges]

        # Temperature scaling
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax across traffics
        attn = attn.view(batch_size, self.agent_num, self.traffic_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Compute weighted value
        h_spatials = h_spatials.view(batch_size, self.agent_num, self.traffic_num, h_size)
        h_spatials = h_spatials.view(batch_size * self.agent_num, self.traffic_num, h_size).permute(0, 2, 1)

        attn = attn.view(batch_size * self.agent_num, self.traffic_num).unsqueeze(-1)
        weighted_value = torch.bmm(h_spatials, attn)  # [batch_size*agent_num, h_size, 1]

        # Reshape back
        weighted_value = weighted_value.squeeze(-1).view(batch_size, self.agent_num, h_size)
        
        return weighted_value, attn

    def forward(self, h_temporal, h_spatials,visible_mask):
        """
        Forward pass
        params:
        h_temporal : Hidden state of the temporal edge [batch_size, 1, 256]
        h_spatials : Hidden states of all spatial edges [batch_size, max_traffic_num, 256]
        """
        batch_size, max_traffic_num, _ = h_spatials.size()
        self.traffic_num = max_traffic_num // self.agent_num

        weighted_value_list, attn_list = [], []
        # Handle the ego dimension (squeeze it since we only have 1 ego)
        if h_temporal.dim() == 2:  # [batch_size, features]
            h_temporal = h_temporal.unsqueeze(1)  # [batch_size, 1, features]
        # Embed the temporal edge hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)  # [batch_size, 1, attention_size]

        # Embed the spatial edge hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)  # [batch_size, max_human_num, attention_size]

        # Repeat temporal embed for each traffic
        temporal_embed = temporal_embed.repeat_interleave(self.traffic_num, dim=1)

        # Create attention mask
        attn_mask = visible_mask
        
        # Apply attention function
        weighted_value, attn = self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
        
        return weighted_value, attn


class DynamicTrafficFeaturesExtractor(nn.Module):
    """
    SB3 compatible features extractor for ego-traffic navigation (attention-based)
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128, use_type_split: bool = True):
        super().__init__()
        
        self.attention_size = 64               # From config: attention_size
        self.use_type_split = use_type_split
        # Calculate ego_node_input_size dynamically
        ego_node_dim = observation_space['robot_node'].shape[-1]  # Last dimension (features)
        self.ego_node_input_size = ego_node_dim
        self.spatial_edge_input_size = observation_space['traffic_states'].shape[-1] # last dimension
        
        # Get traffic_num from observation space
        self.traffic_num = observation_space['traffic_states'].shape[0]
        
        # Initialize attention modules
        internal_features_dim = 256
        spatial_attn_features_dim = 512
        self.spatial_attn = TrafficEdgeSelfAttn(
            input_size=self.spatial_edge_input_size,
            attn_size=spatial_attn_features_dim,
            num_attn_heads=8
        )
        # this 512 followed spatial_attn.attn_size
        self.spatial_linear = nn.Sequential(
            nn.Linear(spatial_attn_features_dim, internal_features_dim), 
            nn.ReLU()
        )
        self.attn_drone = EgoTrafficAttention(
            input_feature_size=internal_features_dim,
            attention_size=self.attention_size
        )
        self.attn_evtol = EgoTrafficAttention(
            input_feature_size=internal_features_dim,
            attention_size=self.attention_size
        )
        self.type_fuse = nn.Sequential(
            nn.Linear(internal_features_dim * 2, internal_features_dim), 
            nn.ReLU()
        )
        # Linear layers
        self.ego_linear = nn.Sequential(
            nn.Linear(self.ego_node_input_size, internal_features_dim), 
            nn.ReLU()
        )
        

        
        half_features_dim = features_dim//2
        # half_features_dim = features_dim
        self.final_ego_linear = nn.Sequential(
            nn.Linear(internal_features_dim, half_features_dim), 
            nn.ReLU()
        )
        # 原来的rnnbase中的encoder_linear

        self.final_spatial_linear = nn.Sequential(
            nn.Linear(internal_features_dim, half_features_dim), 
            nn.ReLU()
        )# 原来rnnbase中的的edge_attention_embed
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)
    
    def forward(self, observations):
        """
        Forward pass of the features extractor
        Args:
            observations: Dictionary containing:
                - robot_node (ego): [batch_size, 1, ego_features]
                - temporal_edges: [batch_size, 1, temporal_features]
                - traffic_states: [batch_size, max_traffic_num, spatial_features]
                - visible_mask: [batch_size, max_traffic_num]
                - traffic_types: [batch_size, max_traffic_num]
        Returns:
            features: [batch_size, features_dim]
        """
        ego_node = observations['robot_node']
        traffic_states = observations['traffic_states']
        visible_mask = observations['visible_masks']
        traffic_types = observations['traffic_types']

        # , need to check if visible_mask are all zeros
        # 1. 找出批次中哪些样本是 "全 0" 掩码
        # all_zero_mask 的形状是 [batch_size]
        all_zero_mask = (visible_mask.sum(dim=-1) == 0)

        # 2. 检查是否 *真的* 有需要修复的样本
        if all_zero_mask.any():
            
            # 3. 动态创建 "dummy_mask" [0, 0, ..., 1]
            # 确保它在正确的设备 (device) 和具有正确的数据类型 (dtype)
            num_traffics = visible_mask.shape[1]
            
            dummy_mask = torch.zeros(
                num_traffics, 
                device=visible_mask.device, 
                dtype=visible_mask.dtype
            )
            dummy_mask[-1] = 1
            
            # 4. 使用布尔索引替换那些 "全 0" 的行
            # PyTorch 会自动将 dummy_mask 广播到 all_zero_mask 中所有为 True 的行
            visible_mask[all_zero_mask] = dummy_mask
        batch_size = ego_node.shape[0]
        
        # Handle the ego dimension (squeeze it since we only have 1 ego)
        if ego_node.dim() == 3:  # [batch_size, 1, features]
            ego_node = ego_node.squeeze(1)  # [batch_size, features]

        ego_states = self.ego_linear(ego_node)  # [batch_size, 256]


        if self.use_type_split:
            spatial_attn_out = self.spatial_attn(traffic_states, visible_mask, traffic_types)  # [batch_size, max_traffic_num, 512]
        else:
            spatial_attn_out = self.spatial_attn(traffic_states, visible_mask, None)  # [batch_size, max_traffic_num, 512]
        # Process spatial features
        output_spatial = self.spatial_linear(spatial_attn_out)  # [batch_size, max_traffic_num, 256]

        # type masks (1=drone, 2=evtol; 0=dummy)
        type_mask_drone = (traffic_types == 1).float()
        type_mask_evtol = (traffic_types == 2).float()

        if self.use_type_split:
            mask_drone = visible_mask * type_mask_drone
            mask_evtol = visible_mask * type_mask_evtol

            hidden_drone, _ = self.attn_drone(ego_states, output_spatial, mask_drone)
            hidden_evtol, _ = self.attn_evtol(ego_states, output_spatial, mask_evtol)

            # fuse streams: [batch_size, 1, 256] concat -> [batch_size, 1, 512] -> project 256
            hidden_attn_weighted = torch.cat([hidden_drone, hidden_evtol], dim=-1)
            hidden_attn_weighted = self.type_fuse(hidden_attn_weighted)
        else:
            # single stream (no type split): reuse attn_drone with full mask
            hidden_attn_weighted, _ = self.attn_drone(ego_states, output_spatial, visible_mask)
        hidden_attn_weighted = hidden_attn_weighted.squeeze(1) # [batch_size, 256]
        # Following original network design: combine robot_states and attention output
        # In original network, this would go through GRU, but we skip that step
        # and directly combine the features for a 256-dimensional output
        ego_states = self.final_ego_linear(ego_states)
        hidden_attn_weighted = self.final_spatial_linear(hidden_attn_weighted) 
        features = torch.cat((ego_states, hidden_attn_weighted), dim=-1)  # [batch_size,  features_dim]
        
        return features
