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
        
        return features


# Shared model for continuous actions (following official SKRL pattern)
class SharedAttentionContinuous(GaussianMixin, DeterministicMixin, Model):
    """
    Shared model for continuous actions using attention mechanisms
    Follows the official SKRL pattern from torch_ant_ppo.py
    
    Architecture:
    - Shared: AttentionFeaturesNetwork (outputs 128-dim features)
    - Separate Actor Network: [256, 256] -> action_dim  
    - Separate Critic Network: [256, 256] -> 1
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 log_std_init: float = 0.0,
                 clip_actions=False,
                 clip_log_std=True, 
                 min_log_std=-20, 
                 max_log_std=2, 
                 reduction="sum",
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        # Shared attention features extractor (like SB3 implementation)
        self.features_extractor = AttentionFeaturesNetwork(
            observation_space, action_space, device, features_dim, **kwargs
        )
        
        # Actor network (separate from critic)
        actor_layers = []
        input_dim = features_dim
        for hidden_dim in net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy head for continuous actions
        self.mean_layer = nn.Linear(input_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * log_std_init)
        
        # Critic network (separate from actor)
        critic_layers = []
        input_dim = features_dim
        for hidden_dim in net_arch:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        
        # Value head
        self.value_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # For shared computation optimization
        self._shared_features = None
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)
        self.mean_layer.apply(init_weights)
        self.value_layer.apply(init_weights)
        
    def act(self, inputs, role):
        """Act method following official SKRL pattern"""
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role: str = "") -> Union[Tuple[torch.Tensor, torch.Tensor, dict], Tuple[torch.Tensor, dict]]:
        """
        Compute method following official SKRL pattern
        
        Only the features extraction is shared, actor and critic have separate networks
        """
        flat_states = inputs["states"]
        unflattened_states = unflatten_tensorized_space(self.observation_space, flat_states)
        if role == "policy":
            # Extract shared features and pass through actor network
            features = self.features_extractor.compute(unflattened_states)
            self._shared_features = features  # Cache for potential value computation
            actor_output = self.actor_net(features)
            mean_actions = self.mean_layer(actor_output)
            return mean_actions, self.log_std_parameter, {}
        elif role == "value":
            # Reuse shared features if available, otherwise compute them
            if self._shared_features is None:
                features = self.features_extractor.compute(unflattened_states)
            else:
                features = self._shared_features
                self._shared_features = None  # Reset for next iteration
            
            critic_output = self.critic_net(features)
            value_output = self.value_layer(critic_output)
            return value_output, {}


# Shared model for discrete actions (following official SKRL pattern) 
class SharedAttentionDiscrete(CategoricalMixin, DeterministicMixin, Model):
    """
    Shared model for discrete actions using attention mechanisms
    Follows the official SKRL pattern from torch_ant_ppo.py
    
    Architecture:
    - Shared: AttentionFeaturesNetwork (outputs 128-dim features)
    - Separate Actor Network: [256, 256] -> action_dim  
    - Separate Critic Network: [256, 256] -> 1
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 unnormalized_log_prob: bool = True,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self)
        
        # Shared attention features extractor (like SB3 implementation)
        self.features_extractor = AttentionFeaturesNetwork(
            observation_space, action_space, device, features_dim, **kwargs
        )
        
        # Actor network (separate from critic)
        actor_layers = []
        input_dim = features_dim
        for hidden_dim in net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy head for discrete actions
        self.logits_layer = nn.Linear(input_dim, self.num_actions)
        
        # Critic network (separate from actor)
        critic_layers = []
        input_dim = features_dim
        for hidden_dim in net_arch:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        
        # Value head
        self.value_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # For shared computation optimization
        self._shared_features = None
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)
        self.logits_layer.apply(init_weights)
        self.value_layer.apply(init_weights)
        
    def act(self, inputs, role):
        """Act method following official SKRL pattern"""
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role: str = "") -> Union[Tuple[torch.Tensor, dict], Tuple[torch.Tensor, dict]]:
        """
        Compute method following official SKRL pattern
        
        Only the features extraction is shared, actor and critic have separate networks
        """
        flat_states = inputs["states"]
        unflattened_states = unflatten_tensorized_space(self.observation_space, flat_states)
        if role == "policy":
            # Extract shared features and pass through actor network
            features = self.features_extractor.compute(unflattened_states)
            self._shared_features = features  # Cache for potential value computation
            actor_output = self.actor_net(features)
            logits = self.logits_layer(actor_output)
            return logits, {}
        elif role == "value":
            # Reuse shared features if available, otherwise compute them
            if self._shared_features is None:
                features = self.features_extractor.compute(unflattened_states)
            else:
                features = self._shared_features
                self._shared_features = None  # Reset for next iteration
            
            critic_output = self.critic_net(features)
            value_output = self.value_layer(critic_output)
            return value_output, {}


# GRU-based Shared models for RNN support (following official SKRL GRU pattern)
class SharedAttentionGRUContinuous(GaussianMixin, DeterministicMixin, Model):
    """
    GRU-based shared model for continuous actions using attention mechanisms
    
    Architecture:
    - Shared: AttentionFeaturesNetwork -> GRU (outputs hidden_size-dim features)
    - Separate Actor Network: [256, 256] -> action_dim  
    - Separate Critic Network: [256, 256] -> 1
    
    Following official SKRL RNN pattern with proper sequence handling
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 log_std_init: float = 0.0,
                 clip_actions=False,
                 clip_log_std=True, 
                 min_log_std=-20, 
                 max_log_std=2, 
                 reduction="sum",
                 num_envs=1,
                 num_layers=1, 
                 hidden_size=64, 
                 sequence_length=16,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Shared attention features extractor
        self.features_extractor = AttentionFeaturesNetwork(
            observation_space, action_space, device, features_dim, **kwargs
        )
        
        # GRU layer (applied after features extraction)
        self.gru = nn.GRU(
            input_size=features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Actor network (takes GRU output)
        actor_layers = []
        input_dim = self.hidden_size
        for hidden_dim in net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy head for continuous actions
        self.mean_layer = nn.Linear(input_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * log_std_init)
        
        # Critic network (takes GRU output)
        critic_layers = []
        input_dim = self.hidden_size
        for hidden_dim in net_arch:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        
        # Value head
        self.value_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # For shared computation optimization
        self._shared_gru_features = None
        self._shared_hidden_states = None
    
    def get_specification(self):
        """返回RNN规格配置，遵循SKRL RNN模式"""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]
            }
        }
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)
        self.mean_layer.apply(init_weights)
        self.value_layer.apply(init_weights)
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
                
    def _process_gru_sequence(self, features, hidden_states, terminated):
        """
        Process sequence through GRU following official SKRL pattern
        
        Args:
            features: Input features from attention extractor [batch_size * sequence_length, features_dim]
            hidden_states: GRU hidden states [num_layers, num_envs, hidden_size]
            terminated: Termination flags [batch_size * sequence_length]
            
        Returns:
            gru_output: GRU output [batch_size * sequence_length, hidden_size]
            new_hidden_states: Updated hidden states [num_layers, num_envs, hidden_size]
        """
        # Training mode: process sequences
        if self.training:
            # Reshape for sequence processing
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])  # (N, L, Hin)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
            # Get initial hidden states for each sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            
            # Handle episode termination in middle of sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
                
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(rnn_input[:,i0:i1,:], hidden_states)
                    # Reset hidden states for terminated episodes
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)
                
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                # No termination in sequence
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            # Rollout mode: process single steps
            rnn_input = features.view(-1, 1, features.shape[-1])  # (N, L=1, Hin)
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        
        # Flatten output for downstream networks
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        return rnn_output, hidden_states
        
    def act(self, inputs, role):
        """Act method following official SKRL pattern"""
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role: str = "") -> Union[Tuple[torch.Tensor, torch.Tensor, dict], Tuple[torch.Tensor, dict]]:
        """
        Compute method following official SKRL RNN pattern
        
        Architecture: AttentionFeatures -> GRU -> Actor/Critic networks
        """
        # Extract inputs
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]
        
        # Handle state flattening for discrete action spaces
        flat_states = inputs["states"]
        unflattened_states = unflatten_tensorized_space(self.observation_space, flat_states)
        
        if role == "policy":
            # Extract attention features
            attention_features = self.features_extractor.compute(unflattened_states)
            
            # Process through GRU
            gru_features, new_hidden_states = self._process_gru_sequence(
                attention_features, hidden_states, terminated
            )
            
            # Cache for potential value computation
            self._shared_gru_features = gru_features
            self._shared_hidden_states = new_hidden_states
            
            # Pass through actor network
            actor_output = self.actor_net(gru_features)
            mean_actions = self.mean_layer(actor_output)
            
            return mean_actions, self.log_std_parameter, {"rnn": [new_hidden_states]}
            
        elif role == "value":
            # Reuse shared GRU features if available
            if self._shared_gru_features is not None:
                gru_features = self._shared_gru_features
                new_hidden_states = self._shared_hidden_states
                # Reset cache
                self._shared_gru_features = None 
                self._shared_hidden_states = None
            else:
                # Compute fresh if not cached
                attention_features = self.features_extractor.compute(unflattened_states)
                gru_features, new_hidden_states = self._process_gru_sequence(
                    attention_features, hidden_states, terminated
                )
            
            # Pass through critic network
            critic_output = self.critic_net(gru_features)
            value_output = self.value_layer(critic_output)
            
            return value_output, {"rnn": [new_hidden_states]}


class SharedAttentionGRUDiscrete(CategoricalMixin, DeterministicMixin, Model):
    """
    GRU-based shared model for discrete actions using attention mechanisms
    
    Architecture:
    - Shared: AttentionFeaturesNetwork -> GRU (outputs hidden_size-dim features)
    - Separate Actor Network: [256, 256] -> action_dim  
    - Separate Critic Network: [256, 256] -> 1
    
    Following official SKRL RNN pattern with proper sequence handling
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 unnormalized_log_prob: bool = True,
                 num_envs=1,
                 num_layers=1, 
                 hidden_size=64, 
                 sequence_length=16,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self)
        
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Shared attention features extractor
        self.features_extractor = AttentionFeaturesNetwork(
            observation_space, action_space, device, features_dim, **kwargs
        )
        
        # GRU layer (applied after features extraction)
        self.gru = nn.GRU(
            input_size=features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Actor network (takes GRU output)
        actor_layers = []
        input_dim = self.hidden_size
        for hidden_dim in net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy head for discrete actions
        self.logits_layer = nn.Linear(input_dim, self.num_actions)
        
        # Critic network (takes GRU output)
        critic_layers = []
        input_dim = self.hidden_size
        for hidden_dim in net_arch:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        
        # Value head
        self.value_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # For shared computation optimization
        self._shared_gru_features = None
        self._shared_hidden_states = None
    
    def get_specification(self):
        """返回RNN规格配置，遵循SKRL RNN模式"""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]
            }
        }
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)
        self.logits_layer.apply(init_weights)
        self.value_layer.apply(init_weights)
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
                
    def _process_gru_sequence(self, features, hidden_states, terminated):
        """
        Process sequence through GRU following official SKRL pattern
        
        Args:
            features: Input features from attention extractor [batch_size * sequence_length, features_dim]
            hidden_states: GRU hidden states [num_layers, num_envs, hidden_size]
            terminated: Termination flags [batch_size * sequence_length]
            
        Returns:
            gru_output: GRU output [batch_size * sequence_length, hidden_size]
            new_hidden_states: Updated hidden states [num_layers, num_envs, hidden_size]
        """
        # Training mode: process sequences
        if self.training:
            # Reshape for sequence processing
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])  # (N, L, Hin)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
            # Get initial hidden states for each sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            
            # Handle episode termination in middle of sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
                
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(rnn_input[:,i0:i1,:], hidden_states)
                    # Reset hidden states for terminated episodes
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)
                
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                # No termination in sequence
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            # Rollout mode: process single steps
            rnn_input = features.view(-1, 1, features.shape[-1])  # (N, L=1, Hin)
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        
        # Flatten output for downstream networks
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        return rnn_output, hidden_states
        
    def act(self, inputs, role):
        """Act method following official SKRL pattern"""
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role: str = "") -> Union[Tuple[torch.Tensor, dict], Tuple[torch.Tensor, dict]]:
        """
        Compute method following official SKRL RNN pattern
        
        Architecture: AttentionFeatures -> GRU -> Actor/Critic networks
        """
        # Extract inputs
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]
        
        # Handle state flattening for discrete action spaces
        flat_states = inputs["states"]
        unflattened_states = unflatten_tensorized_space(self.observation_space, flat_states)
        
        if role == "policy":
            # Extract attention features
            attention_features = self.features_extractor.compute(unflattened_states)
            
            # Process through GRU
            gru_features, new_hidden_states = self._process_gru_sequence(
                attention_features, hidden_states, terminated
            )
            
            # Cache for potential value computation
            self._shared_gru_features = gru_features
            self._shared_hidden_states = new_hidden_states
            
            # Pass through actor network
            actor_output = self.actor_net(gru_features)
            logits = self.logits_layer(actor_output)
            
            return logits, {"rnn": [new_hidden_states]}
            
        elif role == "value":
            # Reuse shared GRU features if available
            if self._shared_gru_features is not None:
                gru_features = self._shared_gru_features
                new_hidden_states = self._shared_hidden_states
                # Reset cache
                self._shared_gru_features = None 
                self._shared_hidden_states = None
            else:
                # Compute fresh if not cached
                attention_features = self.features_extractor.compute(unflattened_states)
                gru_features, new_hidden_states = self._process_gru_sequence(
                    attention_features, hidden_states, terminated
                )
            
            # Pass through critic network
            critic_output = self.critic_net(gru_features)
            value_output = self.value_layer(critic_output)
            
            return value_output, {"rnn": [new_hidden_states]}
