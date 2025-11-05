

from typing import Union, Tuple, Mapping, Any

from torch.distributions import Beta
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gymnasium
from skrl.models.torch import Model, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

# class BetaMixin(GaussianMixin):
#     def __init__(
#         self,
#         clip_actions: bool = False,
#         clip_log_std: bool = True,
#         min_log_std: float = -20,
#         max_log_std: float = 2,
#         reduction: str = "sum",
#         role: str = "",
#     ) -> None:
# 经过检查，Mixin不需要继承GaussianMixin，skrl不会做type check
class BetaMixin:
    def __init__(
        self,
        clip_actions: bool = False, #这个最好永远是False， 因为我的env中有动作放缩的操作
        min_concentration: float = 1e-6,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Beta mixin model (stochastic model)

        This mixin mirrors the public interface of ``GaussianMixin`` so that
        it can be used transparently by SKRL algorithms expecting a stochastic
        policy. Internally it parameterizes a Beta distribution per action dim.

        The policy's ``compute`` method must return two tensors for policy role:
        ``alpha_logits`` and ``beta_logits`` (pre-activation). They will be
        transformed via ``softplus`` and shifted by a small epsilon to obtain
        positive concentrations.

        Action scaling:
        - Samples ``y`` are drawn in (0, 1) from Beta(alpha, beta)
        - If ``clip_actions`` is enabled and an action space is present,
          actions are mapped as ``x = y * (high - low) + low`` and clipped.
        - Log-probabilities account for the affine transform with the
          appropriate log-Jacobian correction ``-sum(log(high - low))``.

        Parameters are kept compatible with ``GaussianMixin`` even if not used
        (e.g., log std bounds), to preserve the expected constructor signature.
        """
        # keep the same flags/names as GaussianMixin for compatibility
        self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)
        if self._g_clip_actions:
            raise NotImplementedError("clip_actions is not supported for BetaMixin for now, because the action space is not bounded in [0, 1]")
            self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        # Reduction behavior compatible with GaussianMixin
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

        # Beta-specific state
        self._b_alpha = None
        self._b_beta = None
        self._b_distribution = None
        self._b_num_samples = None
        self._b_min_concentration = abs(min_concentration)

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        The model must provide in ``compute`` for role=="policy":
        ``alpha_logits``, ``beta_logits``, and ``outputs``.
        """
        # map from states/observations to alpha/beta logits
        alpha_logits, beta_logits, outputs = self.compute(inputs, role)

        # convert logits to positive concentrations
        alpha = torch.nn.functional.softplus(alpha_logits) + self._b_min_concentration
        beta = torch.nn.functional.softplus(beta_logits) + self._b_min_concentration

        self._b_alpha = alpha
        self._b_beta = beta
        self._b_num_samples = alpha.shape[0]

        # distribution over (0, 1)
        self._b_distribution = Beta(alpha, beta)

        # sample using reparameterization
        y = self._b_distribution.rsample()

        # map to action space if requested
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            actions = y * scale + self._g_clip_actions_min
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)
        else:
            actions = y

        # compute log-probability (with transform correction if needed)
        taken_actions = inputs.get("taken_actions", actions)
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            # invert affine transform to (0, 1)
            y_taken = (taken_actions - self._g_clip_actions_min) / (scale + 1e-12)
            y_taken = torch.clamp(y_taken, 1e-8, 1.0 - 1e-8)
            log_prob = self._b_distribution.log_prob(y_taken)
            # log |det J| = sum(log(scale)) per action dim
            log_det_jacobian = torch.log(scale + 1e-12)
            log_prob = log_prob - log_det_jacobian
        else:
            y_taken = torch.clamp(taken_actions, 1e-8, 1.0 - 1e-8)
            log_prob = self._b_distribution.log_prob(y_taken)

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # mean actions for logging (scaled if needed)
        mean_y = self._b_alpha / (self._b_alpha + self._b_beta)
        if self._g_clip_actions:
            scale = self._g_clip_actions_max - self._g_clip_actions_min
            mean_actions = mean_y * scale + self._g_clip_actions_min
        else:
            mean_actions = mean_y
        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model"""
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._b_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return a log-standard-deviation-like tensor for compatibility

        For Beta distributions there is no native log std parameter. This method
        returns the log of the current distribution's standard deviation.
        """
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        var = self._b_distribution.variance + 1e-12
        log_std = 0.5 * torch.log(var)
        return log_std

    def distribution(self, role: str = "") -> torch.distributions.Beta:
        """Get the current distribution of the model"""
        return self._b_distribution

class SharedBetaMixin(BetaMixin, DeterministicMixin, Model):
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
                 min_concentration=1.0,
                 reduction="sum",
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        BetaMixin.__init__(self, clip_actions, min_concentration, reduction)
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
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Actor network (separate from critic)
        actor_layers = []
        input_dim = features_dim
        for hidden_dim in pi_layers_dims:
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
        for hidden_dim in vf_layers_dims:
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


class SharedBetaMixinGRU(BetaMixin, DeterministicMixin, Model):
    """
    GRU-based shared model for continuous actions using attention mechanisms and Beta policy
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device: Union[str, torch.device] = None,
                 features_dim: int = 128,
                 net_arch: list = [256, 256],
                 clip_actions=False,
                 min_concentration=1.0,
                 reduction="sum",
                 num_envs=1,
                 num_layers=1, 
                 hidden_size=64, 
                 sequence_length=16,
                 features_extractor_cls=None,
                 features_extractor_kwargs: dict = None,
                 **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        BetaMixin.__init__(self, clip_actions, min_concentration, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

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

        # GRU layer (applied after features extraction)
        self.gru = nn.GRU(
            input_size=features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Actor network (separate from critic)
        actor_layers = []
        input_dim = self.hidden_size
        for hidden_dim in pi_layers_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        
        # Policy heads for Beta (alpha and beta)
        self.alpha_layer = nn.Linear(input_dim, self.num_actions)
        self.beta_layer = nn.Linear(input_dim, self.num_actions)
        
        # Critic network (separate from actor)
        critic_layers = []
        input_dim = self.hidden_size
        for hidden_dim in vf_layers_dims:
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
        self.alpha_layer.apply(init_weights)
        self.beta_layer.apply(init_weights)
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
            return BetaMixin.act(self, inputs, role)
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
            features = self.features_extractor(unflattened_states)
            
            # Process through GRU
            gru_features, new_hidden_states = self._process_gru_sequence(
                features, hidden_states, terminated
            )
            
            # Cache for potential value computation
            self._shared_gru_features = gru_features
            self._shared_hidden_states = new_hidden_states
            
            # Pass through actor network
            actor_output = self.actor_net(gru_features)
            alpha_logits = self.alpha_layer(actor_output)
            beta_logits = self.beta_layer(actor_output)
            
            return alpha_logits, beta_logits, {"rnn": [new_hidden_states]}
            
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
                features = self.features_extractor(unflattened_states)
                gru_features, new_hidden_states = self._process_gru_sequence(
                    features, hidden_states, terminated
                )
            
            # Pass through critic network
            critic_output = self.critic_net(gru_features)
            value_output = self.value_layer(critic_output)
            
            return value_output, {"rnn": [new_hidden_states]}