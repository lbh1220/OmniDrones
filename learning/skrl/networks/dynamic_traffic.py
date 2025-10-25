"""
SKRL Attention Networks following Official Pattern

This implementation follows the official SKRL shared model pattern from torch_ant_ppo.py:
- SharedAttentionContinuous: For continuous action spaces (GaussianMixin + DeterministicMixin)
- SharedAttentionDiscrete: For discrete action spaces (CategoricalMixin + DeterministicMixin)

Both models use the same shared feature extraction network and compute different outputs
based on the 'role' parameter ('policy' or 'value').

Usage pattern:
    shared_model = SharedAttentionContinuous(...) or SharedAttentionDiscrete(...)
    models = {
        "policy": shared_model,
        "value": shared_model  # Same instance - shared computation
    }
    
This ensures that:
1. Feature extraction happens only once per forward pass
2. Memory usage is optimized 
3. Gradient updates affect both policy and value networks
4. Follows the original SB3 training paradigm with shared features
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, Mapping, Any

from skrl.models.torch import Model, CategoricalMixin, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

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


class AttentionFeaturesNetwork(Model):
    """
    SKRL-compatible attention features network
    Replaces SB3's AttentionFeaturesExtractor
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        
        self.features_dim = features_dim
        self.attention_size = 64
        
        # Calculate input dimensions dynamically
        robot_node_dim = observation_space['robot_node'].shape[-1]
        temporal_edges_dim = observation_space['temporal_edges'].shape[-1]
        self.robot_node_input_size = robot_node_dim + temporal_edges_dim
        self.spatial_edge_input_size = observation_space['spatial_edges'].shape[-1]
        
        # Get human_num from observation space
        self.human_num = observation_space['spatial_edges'].shape[0]
        
        # Initialize attention modules
        internal_features_dim = 256
        spatial_attn_features_dim = 512
        
        self.spatial_attn = SpatialEdgeSelfAttn(
            input_size=self.spatial_edge_input_size,
            attn_size=spatial_attn_features_dim,
            num_attn_heads=8
        )
        
        self.attn = EdgeAttention_M(
            input_feature_size=internal_features_dim,
            attention_size=self.attention_size
        )
        
        # Linear layers
        self.robot_linear = nn.Sequential(
            nn.Linear(self.robot_node_input_size, internal_features_dim), 
            nn.ReLU()
        )
        
        self.spatial_linear = nn.Sequential(
            nn.Linear(spatial_attn_features_dim, internal_features_dim), 
            nn.ReLU()
        )
        
        half_features_dim = features_dim // 2
        
        self.final_robot_linear = nn.Sequential(
            nn.Linear(internal_features_dim, half_features_dim), 
            nn.ReLU()
        )
        
        self.final_spatial_linear = nn.Sequential(
            nn.Linear(internal_features_dim, half_features_dim), 
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
    
    def compute(self, inputs: dict, role: str = "") -> torch.Tensor:
        """
        Compute features from observations
        
        Args:
            inputs: Dictionary containing observations
            role: Role identifier (not used in this implementation)
            
        Returns:
            features: Extracted features tensor
        """
        robot_node = inputs['robot_node']
        temporal_edges = inputs['temporal_edges'] 
        spatial_edges = inputs['spatial_edges']
        detected_human_num = inputs['detected_human_num'].squeeze(-1).int()

        # For the case that no human is detected, set the detected_human_num to 1
        detected_human_num[detected_human_num == 0] = 1 
        
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
        
        # Combine features
        robot_states = self.final_robot_linear(robot_states)
        hidden_attn_weighted = self.final_spatial_linear(hidden_attn_weighted) 
        features = torch.cat((robot_states, hidden_attn_weighted), dim=-1)  # [batch_size, features_dim]
        # TODO, lack a linear layer if there is no RNN
        return features


class CityFeaturesNetwork(Model):
    """A minimal feature network for City env.

    - robot_node: [N, 1, 5]
    - temporal_edges: [N, 1, 2]
    - lidar: [N, 1, 36, 4]
    
    It flattens lidar to [N, 144] after simple conv/pool or linear, concatenates
    robot features, then outputs features_dim.
    """
    def __init__(self, observation_space, action_space, device=None, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, action_space, device, **kwargs)
        self.features_dim = features_dim

        # Infer dims from observation_space
        # there is no temporal_edges in city env
        robot_dim = observation_space['robot_node'].shape[-1]
        lidar_shape = observation_space['lidar'].shape  # (1, 36, 4)
        lidar_h, lidar_w = lidar_shape[-2], lidar_shape[-1]

        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_dim, 64), nn.ReLU(),
            nn.Linear(64, features_dim // 2), nn.ReLU(),
        )

        # Simple lidar encoder: flatten then mlp
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_h * lidar_w, 128), nn.ReLU(),
            nn.Linear(128, features_dim // 2), nn.ReLU(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

    def compute(self, inputs: dict, role: str = "") -> torch.Tensor:
        robot_feat = inputs['robot_node']  # [N, 1, 10]

        lidar = inputs['lidar']  # [N, 1, 36, 4]

        if robot_feat.dim() == 3:
            robot_feat = robot_feat.squeeze(1)

        # concat robot side
        robot_feat = self.robot_mlp(robot_feat)

        # lidar branch: flatten 36x4 -> 144
        lidar_feat = lidar.view(lidar.shape[0], -1)
        lidar_feat = self.lidar_mlp(lidar_feat)

        out = torch.cat([robot_feat, lidar_feat], dim=-1)
        return out


