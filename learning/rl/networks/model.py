import torch
import torch.nn as nn


from rl.networks.distributions import Bernoulli, Categorical, DiagGaussian, DiagGaussian_tanh
from .srnn_model import SRNN
from .selfAttn_srnn_temp_node import selfAttn_merge_SRNN
from .double_selfAttn_srnn import double_selfAttn_merge_SRNN
from .drl_vo_cnn import DRLVO_CNN
from .gat_network import GAT_singlelayer, GAT_single, HeterogeneousGAT_single
from .heter_selfAttn_srnn import heter_selfAttn_merge_SRNN
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'srnn':
            base=SRNN
            self.srnn = True
        elif base == 'selfAttn_merge_srnn':
            base = selfAttn_merge_SRNN
            self.srnn = True
        elif base == 'double_selfAttn_merge_srnn':
            base = double_selfAttn_merge_SRNN
            self.srnn = True
        elif base == 'heter_selfAttn_merge_srnn':
            base = heter_selfAttn_merge_SRNN
            self.srnn = True
        elif base == 'drl_vo':
            base = DRLVO_CNN
            self.srnn = False
        elif base == 'single_layer_gat':
            base = GAT_singlelayer
            self.srnn = False
        elif base == 'single_gat':
            base = GAT_single
            self.srnn = False
        elif base == 'heterogeneous_gat':
            base = HeterogeneousGAT_single
            self.srnn = False
        else:
            raise NotImplementedError

        self.base = base(obs_shape, base_kwargs)
        
        # 这个类是作者自己定义的, 注意其定义并不是一个常规的分布, 其中是包含nn的, 这样能让输出和动作空间维度对齐
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
            # self.dist = DiagGaussian_tanh(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)

        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features) # 这句话输出的才是一个真正的概率分布, self.dist不是一个概率分布

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        else:
            value, _, _ = self.base(inputs, rnn_hxs, masks)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def get_trainable_params(self, freeze_base=False):
        """获取需要训练的参数
        
        Args:
            freeze_base: 如果为True，只训练actor、critic、critic_linear和dist部分
        
        Returns:
            需要训练的参数列表
        """
        if not freeze_base:
            # 训练所有参数
            return self.parameters()
        
        # 只训练actor、critic、critic_linear和dist部分
        trainable_params = []
        
        # 添加dist网络的参数
        trainable_params.extend(self.dist.parameters())
        
        # 添加base中的actor、critic和critic_linear参数
        if hasattr(self.base, 'actor'):
            trainable_params.extend(self.base.actor.parameters())
        if hasattr(self.base, 'critic'):
            trainable_params.extend(self.base.critic.parameters())
        if hasattr(self.base, 'critic_linear'):
            trainable_params.extend(self.base.critic_linear.parameters())
            
        return trainable_params



