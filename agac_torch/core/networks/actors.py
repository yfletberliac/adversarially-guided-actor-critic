from typing import List, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical, Normal

from core.networks.cnn import CNN
from core.networks.mlp import MLP
from core.networks.rl_model import RLModel
from core.utils.inits import initialize_hidden_layer, initialize_last_layer
from core.utils.types import Action, LogProb, Observation, Prob


class GaussianActor(RLModel):
    def __init__(self, observation_dim, action_dim, max_action, layers_dim):
        super(GaussianActor, self).__init__()

        assert (
            len(layers_dim) > 1
        ), "can not define continuous stochastic actor without hidden layers"

        # define shared layers
        self._mean_net = MLP(observation_dim, action_dim, layers_dim)

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self._log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self._max_action = max_action

    def compute_distribution(self, obs) -> Normal:
        mu = self._mean_net(obs)
        std = torch.exp(self._log_std)
        return Normal(mu, std)

    def forward(
        self, x, deterministic: bool = False, return_log_prob: bool = False
    ) -> Tuple[Action, LogProb]:
        # forward pass
        distribution = self.compute_distribution(x)

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        if return_log_prob:
            log_prob = distribution.log_prob(action).sum(axis=-1)
        else:
            log_prob = None

        return action, log_prob, distribution.mean, distribution.scale

    def compute_log_prob(self, observations: Observation, actions: Action) -> LogProb:
        # forward pass
        distribution = self.compute_distribution(observations)
        return distribution.log_prob(actions).sum(axis=-1)

    def select_action(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        action = self.forward(observation, deterministic)[0]
        action = action.detach().cpu().numpy().flatten()
        return action


class DiscreteActor(RLModel):
    def __init__(
        self,
        observation_dim: Tuple[int],
        action_dim: int,
        layers_dim: List[int],
        cnn_extractor: bool = False,
        layers_num_channels: List[int] = None,
    ):
        """
        Simple Discrete Actor with or without CNN extractor.
        observation_dim, Tuple: shape of environment's observation
        """
        super(DiscreteActor, self).__init__()

        if cnn_extractor:
            core_cnn = CNN(
                observation_dim, layers_num_channels, stride=2, kernel_size=3
            )
            flattened_dim = (
                observation_dim[1] * observation_dim[2] * layers_num_channels[-1]
            )
            core_mlp = MLP(flattened_dim, action_dim, layers_dim)
            layers = core_cnn.layers + core_mlp.layers
            self._core = nn.Sequential(core_cnn, nn.Flatten(), core_mlp)

        else:
            self._core = MLP(observation_dim[0], action_dim, layers_dim)
            layers = self._core.layers

        for layer in layers[:-1]:
            initialize_hidden_layer(layer)
        initialize_last_layer(layers[-1])

    def forward(
        self, x, deterministic: bool = False, return_log_prob: bool = False
    ) -> Tuple[Action, LogProb, Prob, LogProb]:
        m = self.compute_distribution(x)
        if deterministic:
            action = torch.argmax(m.probs, dim=-1)
        else:
            action = m.sample()

        if return_log_prob:
            log_prob = m.log_prob(action)
            probs = m.probs
            log_probs = m.logits
        else:
            log_probs, log_prob = None, None
        return action, log_prob, probs, log_probs

    def compute_distribution(self, observations: Observation) -> Categorical:
        logits = self._core(observations)
        probs = nn.Softmax(dim=-1)(logits)
        return Categorical(probs=probs)

    def compute_log_prob(self, observations: Observation, actions: Action) -> LogProb:
        # forward pass
        distribution = self.compute_distribution(observations)
        return distribution.log_prob(actions)

    def select_action(
        self, observation: Observation, deterministic: bool = True
    ) -> Action:
        action = (
            self(observation, deterministic=deterministic)[0]
            .cpu()
            .data.numpy()
            .flatten()
        )
        return int(action)
