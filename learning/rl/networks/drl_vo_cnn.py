import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DRLVO_CNN(nn.Module):
    """
    Implementation of the DRLVO CNN architecture as a standalone policy
    """
    def __init__(self, obs_space_dict: Dict, args):
        super(DRLVO_CNN, self).__init__()
        self.args = args
        self.is_recurrent = False
        
        # Network parameters
        block = Bottleneck
        layers = [2, 1, 1]
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.features_dim = 256  # Output feature dimension
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Main network layers
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # Additional conv layers
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1, stride=4, padding=0),
            nn.BatchNorm2d(512)
        )

        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final layers
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * block.expansion + 2, self.features_dim),
            nn.ReLU()
        )

        # Actor and Critic heads
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(self.features_dim, 256)), 
            nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.features_dim, 128)), 
            nn.Tanh())

        self.critic_linear = init_(nn.Linear(128, 1))

        # Initialize weights
        self._initialize_weights()
        self.output_size = 256

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _extract_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get inputs from observation dictionary
        # ped_pos = observations['ped_pos'].reshape(-1, 2, 80, 80)  # [B, 2, 80, 80]
        ped_pos = observations['ped_pos']# [B, 2, 80, 80]
        # scan = observations['scan'].reshape(-1, 1, 80, 80)      # [B, 1, 80, 80]
        goal = observations['goal']                             # [B, 2]

        # Fusion net
        # fusion_in = torch.cat((scan, ped_pos), dim=1)  # [B, 3, 80, 80]
        
        x = self.conv1(ped_pos)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)
        identity2 = self.downsample2(x)
        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        # Combine with goal
        fc_in = torch.cat((fusion_out, goal), dim=1)
        features = self.linear_fc(fc_in)

        return features

    def forward(self, observations, rnn_hxs, masks):
        features = self._extract_features(observations)
        
        hidden_critic = self.critic(features)
        hidden_actor = self.actor(features)
        
        # 这里原来的代码中区分了infer和train，应该不需要区分，没有循环神经网络的行为
        # 但是维度是不是需要调整暂时不太确定
        # mask也不是很必要，mask在原来的模型中是为了调整RNN的行为，避免下一个episode的起点用上个终点的hiden state
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
