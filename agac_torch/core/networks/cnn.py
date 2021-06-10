import math
from typing import List, Tuple, Union

import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        input_dim: Union[List[int], Tuple[int]],
        layers_num_channels: List[int],
        stride: int = 2,
        kernel_size: int = 3,
    ):
        nn.Module.__init__(self)
        # Same padding
        input_num_channels, input_h, input_w = input_dim
        padding = math.ceil((input_h * (stride - 1) + kernel_size - stride) / 2)
        layers_num_channels = [input_num_channels] + layers_num_channels
        num_channels = zip(layers_num_channels[:-1], layers_num_channels[1:])
        layers = []
        self._layers_without_activations = []
        for in_channels, out_channels in num_channels:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self._layers_without_activations.append(layer)
            layers.append(layer)
            layers.append(nn.ELU())
        self._cnn = nn.Sequential(*layers)

    @property
    def layers(self):
        return self._layers_without_activations

    def forward(self, x):
        return self._cnn(x)
