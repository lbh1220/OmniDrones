from .shared_gaussian import SharedGaussianMixin, SharedGaussianMixinGRU
from .shared_discrete import SharedCategoricalMixin, SharedAttentionGRUDiscrete
from .shared_beta import SharedBetaMixin

def select_skrl_model(action_space_type: str, use_rnn: bool = False):
    if action_space_type == "discrete":
        if use_rnn:
            return SharedAttentionGRUDiscrete
        else:
            return SharedCategoricalMixin
    elif action_space_type == "gaussian":
        if use_rnn:
            return SharedGaussianMixinGRU
        else:
            return SharedGaussianMixin
    elif action_space_type == "beta":   
        return SharedBetaMixin
    else:
        raise ValueError(f"Invalid action space type: {action_space_type}")