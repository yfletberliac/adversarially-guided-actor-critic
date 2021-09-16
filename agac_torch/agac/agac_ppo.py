from typing import List

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Space
from mpi4py import MPI
from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence

from agac.configs import ExperimentConfig
from agac.logger import LogData
from agac.memory import Batch
from core.networks.actors import DiscreteActor, GaussianActor
from core.networks.critics import CNNContinuousVNetwork, ContinuousVNetwork
from core.utils.mpi_utils import sync_grads, sync_networks
from core.utils.types import Action, Distribution, Observation

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


class PPO:
    def __init__(self, obs_space: Box, action_space: Space, config: ExperimentConfig):
        rl_config = config.reinforcement_learning
        layers_dim = rl_config.layers_dim
        adv_layers_dim = rl_config.adv_layers_dim
        actor_lr = rl_config.actor_learning_rate
        critic_lr = rl_config.critic_learning_rate

        self._discrete = config.algorithm.discrete
        self._clipping_epsilon = rl_config.clipping_epsilon
        self._value_loss_clip = rl_config.value_loss_clip
        self._value_loss_coeff = rl_config.value_loss_coeff
        self._adv_loss_coeff = rl_config.adversary_loss_coeff
        self._entropy_coeff = rl_config.entropy_coefficient
        self._clip_grad_norm = rl_config.clip_grad_norm
        if config.algorithm.gpu != -1:
            self._device = torch.device("cuda:" + str(config.algorithm.gpu))
        else:
            self._device = "cpu"
        observation_dim = obs_space.shape

        if self._discrete:
            action_dim = action_space.n
            self._actor = DiscreteActor(
                observation_dim,
                action_dim,
                layers_dim,
                cnn_extractor=rl_config.cnn_extractor,
                layers_num_channels=rl_config.layers_num_channels,
            )
            self._adversary = DiscreteActor(
                observation_dim,
                action_dim,
                adv_layers_dim,
                cnn_extractor=rl_config.cnn_extractor,
                layers_num_channels=rl_config.layers_num_channels,
            )
        else:
            # Continuous action space
            observation_dim = observation_dim[0]
            action_dim = action_space.shape[0]
            max_action = action_space.high[0]
            self._actor = GaussianActor(
                observation_dim, action_dim, max_action, layers_dim
            )
            self._adversary = GaussianActor(
                observation_dim, action_dim, max_action, adv_layers_dim
            )

        if rl_config.cnn_extractor:
            self._critic = CNNContinuousVNetwork(
                observation_dim, layers_dim, rl_config.layers_num_channels
            )
        else:
            self._critic = ContinuousVNetwork(observation_dim, layers_dim)

        # To device
        self._actor.to(self._device)
        self._critic.to(self._device)
        self._adversary.to(self._device)

        sync_networks(self._actor, comm)
        sync_networks(self._critic, comm)
        sync_networks(self._adversary, comm)

        self._actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self._adversary_optimizer = torch.optim.Adam(
            self._adversary.parameters(), lr=actor_lr, eps=1e-5
        )
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), lr=critic_lr, eps=1e-5
        )

        # Monitoring
        self._monitor = dict()

    def get_actor_weights(self):
        """
        Returns actor weights.
        """
        self._actor.get_params()

    def set_actor_weights(self, params):
        """
        Set actor weights.
        """
        self._actor.set_params(params)

    def select_action(
        self, observation: Observation, deterministic: bool = False
    ) -> Action:
        """
        Select an action given an observation. Returns the selected action
        and the corresponding log probability.
        """
        observation = (
            torch.from_numpy(observation).float().to(self._device).unsqueeze(0)
        )
        outs = self._actor.forward(observation, deterministic, return_log_prob=True)
        action, log_pi = outs[0], outs[1]
        adv_log_pi = self._adversary.compute_log_prob(observation, action)
        action = action.detach().cpu().numpy().flatten()
        log_pi = log_pi.detach().cpu().numpy().flatten()
        adv_log_pi = adv_log_pi.detach().cpu().numpy().flatten()

        # compute additional distribution information
        actor_distrib = self._actor.compute_distribution(observation)
        adversary_distrib = self._adversary.compute_distribution(observation)
        if self._discrete:
            params_pi = outs[-1]
            params_adv = adversary_distrib.logits
            params_pi = params_pi.detach().cpu().numpy().flatten()
            params_adv = params_adv.detach().cpu().numpy().flatten()
        else:
            mean = actor_distrib.mean.detach().cpu().numpy().flatten()
            scale = actor_distrib.scale.detach().cpu().numpy().flatten()
            params_pi = np.concatenate([mean, scale], -1)
            mean = adversary_distrib.mean.detach().cpu().numpy().flatten()
            scale = adversary_distrib.scale.detach().cpu().numpy().flatten()
            params_adv = np.concatenate([mean, scale], -1)
        return action, log_pi, adv_log_pi, params_pi, params_adv

    def compute_values(self, observations: Observation) -> np.ndarray:
        """
        Compute value estimations for a batch of observations.
        """
        observations = torch.from_numpy(observations).float().to(self._device)
        return (
            self._critic(observations)
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .astype(np.float32)
        )

    def compute_distributions(self, observations: Observation) -> Distribution:
        """
        Compute action distributions conditional on observations
        """
        observations = torch.from_numpy(observations).float().to(self._device)
        return self._actor.compute_distribution(observations)

    def train_on_batch(self, batch: Batch, intrinsic_coef: float):
        """
        Update actor and critic networks on batch of transitions.
        """

        # read batch
        actions = torch.tensor(
            batch.actions, device=self._device, dtype=torch.float32
        ).squeeze()
        observations = torch.tensor(
            batch.observations, device=self._device, dtype=torch.float32
        )
        log_pi_old = torch.tensor(
            batch.log_pis, device=self._device, dtype=torch.float32
        )
        adv_log_pi_old = torch.tensor(
            batch.adv_log_pis, device=self._device, dtype=torch.float32
        )
        advantages = torch.tensor(
            batch.advantages, device=self._device, dtype=torch.float32
        )
        agac_advantages = torch.tensor(
            batch.agac_advantages, device=self._device, dtype=torch.float32
        )
        values_old = torch.tensor(
            batch.values, device=self._device, dtype=torch.float32
        )
        params_pi = torch.tensor(
            batch.logits_pi, device=self._device, dtype=torch.float32
        )
        adv_params_pi = torch.tensor(
            batch.adv_logits_pi, device=self._device, dtype=torch.float32
        )

        returns = values_old + advantages
        self._monitor["estimated_returns"] = float(returns.mean())
        self._monitor["log_pis"] = float(log_pi_old.mean())
        self._monitor["adv_log_pis"] = float(adv_log_pi_old.mean())
        self._monitor["advantages"] = float(advantages.mean())
        self._monitor["agac_advantages"] = float(agac_advantages.mean())

        # get distributions
        pi_distribution = self._actor.compute_distribution(observations)
        log_pi = pi_distribution.log_prob(actions)
        if not self._discrete:
            # Summing independent univariate normal logits for Gaussian Actor
            log_pi = log_pi.sum(axis=-1).flatten()

        # compute surrogate loss
        probability_ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = torch.clamp(
            probability_ratio,
            min=1.0 - self._clipping_epsilon,
            max=1.0 + self._clipping_epsilon,
        )

        policy_loss = -torch.min(
            probability_ratio * agac_advantages,
            clipped_ratio * agac_advantages,
        ).mean()
        self._monitor["policy_loss"] = float(policy_loss)

        # add entropy bonus
        entropy = pi_distribution.entropy()
        if not self._discrete:
            # Summing independent univariate normal entropies for Gaussian Actor
            entropy = entropy.sum(axis=-1).flatten()
        entropy = entropy.mean()
        self._monitor["entropy"] = float(entropy)
        policy_loss -= self._entropy_coeff * entropy

        # compute kl
        with torch.no_grad():
            if self._discrete:
                probs_pi = nn.Softmax(-1)(params_pi)
                adv_probs_pi = nn.Softmax(-1)(adv_params_pi)
                old_pi_distribution = Categorical(probs=probs_pi.squeeze())
                adv_old_pi_distribution = Categorical(
                    probs=adv_probs_pi.squeeze() + 1e-8
                )
            else:
                dim = pi_distribution.loc.shape[-1]
                old_pi_mean = params_pi.squeeze()[:, :dim]
                old_pi_std = adv_params_pi.squeeze()[:, dim:]
                adv_old_pi_mean = adv_params_pi.squeeze()[:, :dim]
                adv_old_pi_std = adv_params_pi.squeeze()[:, dim:]
                old_pi_distribution = Normal(loc=old_pi_mean, scale=old_pi_std)
                adv_old_pi_distribution = Normal(
                    loc=adv_old_pi_mean, scale=adv_old_pi_std + 1e-8
                )

            # compute kl
            pi_adv_kl = kl_divergence(old_pi_distribution, adv_old_pi_distribution)
            pi_adv_kl = pi_adv_kl[:, None]
            if isinstance(pi_distribution, Normal):
                pi_adv_kl = pi_adv_kl.sum(axis=-1)
            pi_adv_kl *= intrinsic_coef

        # compute value losses
        pred_values = self._critic(observations).flatten()
        # clipped Values
        pred_values_clipped = values_old + torch.clamp(
            pred_values - values_old,
            min=-self._value_loss_clip,
            max=self._value_loss_clip,
        )
        vloss1 = torch.pow(pred_values - returns - pi_adv_kl, 2)
        vloss2 = torch.pow(pred_values_clipped - returns - pi_adv_kl, 2)
        value_loss = self._value_loss_coeff * torch.max(vloss1, vloss2).mean()
        self._monitor["value_loss"] = float(value_loss)

        # adversary loss
        adv_pi_distribution = self._adversary.compute_distribution(observations)
        if self._discrete:
            adv_pi_distribution.probs = adv_pi_distribution.probs + 1e-8
            adv_loss = kl_divergence(old_pi_distribution, adv_pi_distribution)
        else:
            adv_pi_distribution.scale = adv_pi_distribution.scale + 1e-8
            adv_loss = kl_divergence(old_pi_distribution, adv_pi_distribution).sum(
                axis=-1
            )
        adv_loss = adv_loss.mean()
        self._monitor["adversary_loss"] = float(adv_loss)
        adv_loss *= self._adv_loss_coeff

        # backpropagation
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._adversary.zero_grad()
        adv_loss.backward()
        self._critic_optimizer.zero_grad()
        value_loss.backward()

        if self._clip_grad_norm != -1:
            nn.utils.clip_grad_norm_(self._actor.parameters(), self._clip_grad_norm)
            nn.utils.clip_grad_norm_(self._critic.parameters(), self._clip_grad_norm)
            nn.utils.clip_grad_norm_(self._adversary.parameters(), self._clip_grad_norm)

        sync_grads(self._actor, comm)
        self._actor_optimizer.step()
        sync_grads(self._adversary, comm)
        self._adversary_optimizer.step()
        sync_grads(self._critic, comm)
        self._critic_optimizer.step()

    @property
    def logs(self) -> List[LogData]:
        logs = []
        logs.append(
            LogData(
                name="ppo/actor_loss", value=self._monitor["policy_loss"], type="scalar"
            )
        )
        logs.append(
            LogData(
                name="ppo/value_loss", value=self._monitor["value_loss"], type="scalar"
            )
        )
        logs.append(
            LogData(
                name="ppo/actor_entropy", value=self._monitor["entropy"], type="scalar"
            )
        )
        logs.append(
            LogData(name="ppo/log_pis", value=self._monitor["log_pis"], type="scalar")
        )
        logs.append(
            LogData(
                name="ppo/adv_log_pis",
                value=self._monitor["adv_log_pis"],
                type="scalar",
            )
        )
        logs.append(
            LogData(
                name="model/estimated_returns",
                value=self._monitor["estimated_returns"],
                type="scalar",
            )
        )
        logs.append(
            LogData(
                name="model/agac_advantages",
                value=self._monitor["agac_advantages"],
                type="scalar",
            )
        )
        logs.append(
            LogData(
                name="model/advantages",
                value=self._monitor["advantages"],
                type="scalar",
            )
        )
        logs.append(
            LogData(
                name="adversary/loss",
                value=self._monitor["adversary_loss"],
                type="scalar",
            )
        )
        return logs
