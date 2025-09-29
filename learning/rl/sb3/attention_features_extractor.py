import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn.functional as F


class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, input_size=13, attn_size=512, num_attn_heads=8):
        super(SpatialEdgeSelfAttn, self).__init__()
        
        self.input_size = input_size
        self.num_attn_heads = num_attn_heads
        self.attn_size = attn_size

        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, 128), 
            nn.ReLU(),
            nn.Linear(128, self.attn_size), 
            nn.ReLU()
        )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn = torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    def create_attn_mask(self, each_seq_len, max_human_num, device):
        """Create attention mask based on actual sequence lengths"""
        batch_size = each_seq_len.shape[0]
        mask = torch.zeros(batch_size, max_human_num + 1, device=device)
        # Ensure arange is on the same device
        arange_indices = torch.arange(batch_size, device=device)
        mask[arange_indices, each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1]  # batch_size, max_human_num
        return mask

    def forward(self, inp, each_seq_len):
        """
        Forward pass for the model
        params:
        inp : input edge features [batch_size, max_human_num, feature_dim]
        each_seq_len: number of detected humans [batch_size]
        """
        batch_size, max_human_num, _ = inp.size()
        
        # Create attention mask
        attn_mask = self.create_attn_mask(each_seq_len, max_human_num, inp.device)

        # Embed input features
        input_emb = self.embedding_layer(inp)  # [batch_size, max_human_num, attn_size]
        input_emb = input_emb.transpose(0, 1)  # [max_human_num, batch_size, attn_size]
        
        # Apply linear transformations
        q = self.q_linear(input_emb)
        k = self.k_linear(input_emb)
        v = self.v_linear(input_emb)

        # Apply multi-head attention
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))
        z = z.transpose(0, 1)  # [batch_size, max_human_num, attn_size]
        
        return z


class EdgeAttention_M(nn.Module):
    """
    Class for the robot-human attention module
    """
    def __init__(self, input_feature_size=256, attention_size=64):
        super(EdgeAttention_M, self).__init__()
        
        self.input_feature_size = input_feature_size
        self.attention_size = attention_size

        # Linear layers to embed temporal and spatial edge states
        self.temporal_edge_layer = nn.Linear(self.input_feature_size, self.attention_size)
        self.spatial_edge_layer = nn.Linear(self.input_feature_size, self.attention_size)

        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, max_human_num, device):
        """Create attention mask based on actual sequence lengths"""
        batch_size = each_seq_len.shape[0]
        mask = torch.zeros(batch_size, max_human_num + 1, device=device)
        # Ensure arange is on the same device
        arange_indices = torch.arange(batch_size, device=device)
        mask[arange_indices, each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1]  # batch_size, max_human_num
        return mask

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

        # Softmax across humans
        attn = attn.view(batch_size, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Compute weighted value
        h_spatials = h_spatials.view(batch_size, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(batch_size * self.agent_num, self.human_num, h_size).permute(0, 2, 1)

        attn = attn.view(batch_size * self.agent_num, self.human_num).unsqueeze(-1)
        weighted_value = torch.bmm(h_spatials, attn)  # [batch_size*agent_num, h_size, 1]

        # Reshape back
        weighted_value = weighted_value.squeeze(-1).view(batch_size, self.agent_num, h_size)
        
        return weighted_value, attn

    def forward(self, h_temporal, h_spatials, each_seq_len):
        """
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edge [batch_size, 1, 256]
        h_spatials : Hidden states of all spatial edges [batch_size, max_human_num, 256]
        each_seq_len: number of detected humans [batch_size]
        """
        batch_size, max_human_num, _ = h_spatials.size()
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list = [], []
        
        # Embed the temporal edge hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)  # [batch_size, 1, attention_size]

        # Embed the spatial edge hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)  # [batch_size, max_human_num, attention_size]

        # Repeat temporal embed for each human
        temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=1)

        # Create attention mask
        attn_mask = self.create_attn_mask(each_seq_len, max_human_num, h_spatials.device)
        
        # Apply attention function
        weighted_value, attn = self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
        
        return weighted_value, attn


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 compatible features extractor based on attention mechanisms
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Use human_node_output_size as features dimension (same as original)
        # actual_features_dim = 256  # From config: human_node_output_size = 256
        super().__init__(observation_space, features_dim)
        
        # Network parameters (matching original config.py)
        # self.spatial_edge_input_size = 13  # From config: 2*(predict_steps+1) + 1

        self.attention_size = 64               # From config: attention_size
        
        # Calculate robot_node_input_size dynamically (like in train_traffic.py)
        robot_node_dim = observation_space['robot_node'].shape[-1]  # Last dimension (features)
        temporal_edges_dim = observation_space['temporal_edges'].shape[-1]  # Last dimension (features)
        self.robot_node_input_size = robot_node_dim + temporal_edges_dim  # Combined features
        self.spatial_edge_input_size = observation_space['spatial_edges'].shape[-1] # last dimension
        
        # Get human_num from observation space
        self.human_num = observation_space['spatial_edges'].shape[0]
        
        # Initialize attention modules
        self.spatial_attn = SpatialEdgeSelfAttn(
            input_size=self.spatial_edge_input_size,
            attn_size=512,
            num_attn_heads=8
        )
        
        self.attn = EdgeAttention_M(
            input_feature_size=features_dim,
            attention_size=self.attention_size
        )
        
        # Linear layers
        self.robot_linear = nn.Sequential(
            nn.Linear(self.robot_node_input_size, features_dim), 
            nn.ReLU()
        )
        
        # this 512 followed spatial_attn.attn_size
        self.spatial_linear = nn.Sequential(
            nn.Linear(512, features_dim), 
            nn.ReLU()
        )
        
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
                - robot_node: [batch_size, 1, robot_features] - with robot_num dimension
                - temporal_edges: [batch_size, 1, temporal_features] - with robot_num dimension
                - spatial_edges: [batch_size, max_human_num, spatial_features]
                - detected_human_num: [batch_size, 1]
        Returns:
            features: [batch_size, features_dim]
        """
        robot_node = observations['robot_node']
        temporal_edges = observations['temporal_edges'] 
        spatial_edges = observations['spatial_edges']
        detected_human_num = observations['detected_human_num'].squeeze(-1).int()
        
        batch_size = robot_node.shape[0]
        
        # Handle the robot_num dimension (squeeze it since we only have 1 robot)
        if robot_node.dim() == 3:  # [batch_size, 1, features]
            robot_node = robot_node.squeeze(1)  # [batch_size, features]
        if temporal_edges.dim() == 3:  # [batch_size, 1, features]
            temporal_edges = temporal_edges.squeeze(1)  # [batch_size, features]
        
        # Combine robot and temporal features
        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)  # [batch_size, 256]
        
        # Human-human self attention
        spatial_attn_out = self.spatial_attn(spatial_edges, detected_human_num)  # [batch_size, max_human_num, 512]
        
        # Process spatial features
        output_spatial = self.spatial_linear(spatial_attn_out)  # [batch_size, max_human_num, 256]
        
        # Robot-human attention
        robot_states_expanded = robot_states.unsqueeze(1)  # [batch_size, 1, 256]
        hidden_attn_weighted, _ = self.attn(robot_states_expanded, output_spatial, detected_human_num)
        hidden_attn_weighted = hidden_attn_weighted.squeeze(1)  # [batch_size, 256]
        
        # Following original network design: combine robot_states and attention output
        # In original network, this would go through GRU, but we skip that step
        # and directly combine the features for a 256-dimensional output
        features = robot_states + hidden_attn_weighted  # [batch_size, 256]
        
        return features
