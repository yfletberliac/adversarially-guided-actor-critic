from typing import List, Union

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim: List[int], output_dim: int, layers_dim: Union[List[int], int]
    ):
        nn.Module.__init__(self)
        num_neurons = [input_dim] + list(layers_dim) + [output_dim]
        num_neurons = zip(num_neurons[:-1], num_neurons[1:])
        layers = []
        self._layers_without_activations = []
        for in_dim, out_dim in num_neurons:
            layer = nn.Linear(in_dim, out_dim)
            self._layers_without_activations.append(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
        layers.pop()  # remove last activation
        self._mlp = nn.Sequential(*layers)

    @property
    def layers(self):
        return self._layers_without_activations

    def forward(self, x):
        return self._mlp(x)
