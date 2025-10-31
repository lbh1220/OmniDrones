import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, Mapping, Any

from skrl.models.torch import Model, CategoricalMixin, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from networks.beta import BetaMixin
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
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        # Shared attention features extractor (like SB3 implementation)
        if features_extractor_cls is None:
            features_extractor_cls = nn.Module
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": features_dim}
        else:
            if features_extractor_kwargs.get("features_dim") is None:
                features_extractor_kwargs["features_dim"] = features_dim
        self.features_extractor = features_extractor_cls(
            observation_space, **features_extractor_kwargs
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
        self.features_extractor.apply(init_weights)
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
            features = self.features_extractor(unflattened_states)
            self._shared_features = features  # Cache for potential value computation
            actor_output = self.actor_net(features)
            mean_actions = self.mean_layer(actor_output)
            return mean_actions, self.log_std_parameter, {}
        elif role == "value":
            # Reuse shared features if available, otherwise compute them
            if self._shared_features is None:
                features = self.features_extractor(unflattened_states)
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
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self)
        
        # Shared attention features extractor (like SB3 implementation)
        if features_extractor_cls is None:
            features_extractor_cls = nn.Module
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": features_dim}
        else:
            if features_extractor_kwargs.get("features_dim") is None:
                features_extractor_kwargs["features_dim"] = features_dim
        self.features_extractor = features_extractor_cls(
            observation_space, action_space, device, **features_extractor_kwargs
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
        self.features_extractor.apply(init_weights)
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
            features = self.features_extractor(unflattened_states)
            self._shared_features = features  # Cache for potential value computation
            actor_output = self.actor_net(features)
            logits = self.logits_layer(actor_output)
            return logits, {}
        elif role == "value":
            # Reuse shared features if available, otherwise compute them
            if self._shared_features is None:
                features = self.features_extractor(unflattened_states)
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
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Shared attention features extractor
        # Shared attention features extractor (like SB3 implementation)
        if features_extractor_cls is None:
            features_extractor_cls = nn.Module
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": features_dim}
        else:
            if features_extractor_kwargs.get("features_dim") is None:
                features_extractor_kwargs["features_dim"] = features_dim
        self.features_extractor = features_extractor_cls(
            observation_space, action_space, device, **features_extractor_kwargs
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
        self.features_extractor.apply(init_weights)
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
            attention_features = self.features_extractor(unflattened_states)
            
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
                attention_features = self.features_extractor(unflattened_states)
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
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self)
        
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Shared attention features extractor
        # Shared attention features extractor (like SB3 implementation)
        if features_extractor_cls is None:
            features_extractor_cls = nn.Module
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": features_dim}
        else:
            if features_extractor_kwargs.get("features_dim") is None:
                features_extractor_kwargs["features_dim"] = features_dim
        self.features_extractor = features_extractor_cls(
            observation_space, action_space, device, **features_extractor_kwargs
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

        self.features_extractor.apply(init_weights)
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
            attention_features = self.features_extractor(unflattened_states)
            
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
                attention_features = self.features_extractor(unflattened_states)
                gru_features, new_hidden_states = self._process_gru_sequence(
                    attention_features, hidden_states, terminated
                )
            
            # Pass through critic network
            critic_output = self.critic_net(gru_features)
            value_output = self.value_layer(critic_output)
            
            return value_output, {"rnn": [new_hidden_states]}

# --- New Beta-based shared continuous model ---
class SharedBetaContinuous(BetaMixin, DeterministicMixin, Model):
    """
    Shared model for continuous actions using attention mechanisms and Beta policy

    Architecture:
    - Shared: AttentionFeaturesNetwork (outputs features_dim)
    - Separate Actor Network: [256, 256] -> action_dim (alpha, beta heads)
    - Separate Critic Network: [256, 256] -> 1
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 clip_actions=False,
                 reduction="sum",
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        BetaMixin.__init__(self, clip_actions, True, -20, 2, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        
        # Shared attention features extractor (like SB3 implementation)
        if features_extractor_cls is None:
            features_extractor_cls = nn.Module
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": features_dim}
        else:
            if features_extractor_kwargs.get("features_dim") is None:
                features_extractor_kwargs["features_dim"] = features_dim
        self.features_extractor = features_extractor_cls(
            observation_space, action_space, device, **features_extractor_kwargs
        )
        
        # Actor network (separate from critic)
        actor_layers = []
        input_dim = features_dim
        for hidden_dim in net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy heads for Beta (alpha and beta)
        self.alpha_layer = nn.Linear(input_dim, self.num_actions)
        self.beta_layer = nn.Linear(input_dim, self.num_actions)
        
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
        self.features_extractor.apply(init_weights)
        self.actor_net.apply(init_weights)
        self.critic_net.apply(init_weights)
        self.alpha_layer.apply(init_weights)
        self.beta_layer.apply(init_weights)
        self.value_layer.apply(init_weights)
        
    def act(self, inputs, role):
        """Act method following official SKRL pattern"""
        if role == "policy":
            return BetaMixin.act(self, inputs, role)
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
            features = self.features_extractor(unflattened_states)
            self._shared_features = features  # Cache for potential value computation
            actor_output = self.actor_net(features)
            alpha_logits = self.alpha_layer(actor_output)
            beta_logits = self.beta_layer(actor_output)
            return alpha_logits, beta_logits, {}
        elif role == "value":
            # Reuse shared features if available, otherwise compute them
            if self._shared_features is None:
                features = self.features_extractor(unflattened_states)
            else:
                features = self._shared_features
                self._shared_features = None  # Reset for next iteration
            
            critic_output = self.critic_net(features)
            value_output = self.value_layer(critic_output)
            return value_output, {}
