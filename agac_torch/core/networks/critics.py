from typing import List, Tuple, Union

import torch.nn as nn

from core.networks.cnn import CNN
from core.networks.mlp import MLP
from core.networks.rl_model import RLModel

Critic = Union["ContinuousVNetwork", "CNNContinuousVNetwork"]


class ContinuousVNetwork(RLModel):
    """Simple value network with MLP."""

    def __init__(self, observation_dim: int, layers_dim: List[int]):
        super(ContinuousVNetwork, self).__init__()

        # V network architecture
        self._v_mlp = MLP(observation_dim, 1, layers_dim)

    def forward(self, x):
        x = self._v_mlp(x)
        return x


class CNNContinuousVNetwork(RLModel):
    """Value network with CNN extractor."""

    def __init__(
        self,
        observation_dim: Tuple[int],
        layers_dim: List[int],
        layers_num_channels: List[int],
    ):
        super(CNNContinuousVNetwork, self).__init__()

        flattened_dim = (
            observation_dim[1] * observation_dim[2] * layers_num_channels[-1]
        )
        v_mlp = MLP(flattened_dim, 1, layers_dim)
        v_cnn = CNN(observation_dim, layers_num_channels, stride=2, kernel_size=3)
        self._v = nn.Sequential(v_cnn, nn.Flatten(), v_mlp)

    def forward(self, x):
        x = self._v(x)
        return x
