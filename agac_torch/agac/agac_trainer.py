import re
from copy import deepcopy
from typing import List, Tuple

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete
from mpi4py import MPI
from welford import Welford

from agac.agac_ppo import PPO
from agac.configs import ExperimentConfig
from agac.logger import LogData, Logger
from agac.memory import Memory, Transition
from agac.utils import DiscreteGrid, compute_advantages_and_returns

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


class AGAC:
    def __init__(self, config: ExperimentConfig, env: gym.Env):
        self._config = config
        self._env = env
        self._evaluation_env = deepcopy(env)
        self._discrete = config.algorithm.discrete
        if self._discrete and (not isinstance(env.action_space, Discrete)):
            raise TypeError("`env.action_space` must be of `Discrete` type")
        if (not self._discrete) and (not isinstance(env.action_space, Box)):
            raise TypeError("`env.action_space` must be of `Box` type")

        self._memory = Memory()
        self._ppo = PPO(env.observation_space, env.action_space, config)

        if rank == 0:
            self._logger = Logger(config)
        self._intrinsic_returns_stats = Welford()
        self._extrinsic_returns_stats = Welford()

        if config.algorithm.seed != -1:
            self._set_seed(config.algorithm.seed + rank)

        self._train_freq = config.algorithm.train_freq // num_workers
        self._eval_freq = config.algorithm.eval_freq // num_workers
        self._max_steps = config.algorithm.max_steps // num_workers
        self._num_epochs = config.algorithm.num_epochs
        if not self._discrete:
            self._max_action = env.action_space.high[0]
        self._max_updates = self._max_steps // self._train_freq

        rl_config = config.reinforcement_learning
        self._discount = rl_config.discount
        self._lambda_gae = rl_config.lambda_gae
        self._batch_size = rl_config.batch_size
        self._intrinsic_reward_coefficient = rl_config.intrinsic_reward_coefficient
        self._current_intrinsic_reward_coefficient = (
            rl_config.intrinsic_reward_coefficient
        )
        self._episodic_count = rl_config.episodic_count_coefficient > 0
        self._episodic_count_coefficient = rl_config.episodic_count_coefficient
        if self._episodic_count:
            if not hasattr(env, "state_key_extraction"):
                raise AttributeError(
                    f"To use episodic counts, the environment must"
                    "have a `state_key_extraction` method"
                )
            # TODO: pass function in config
            self._state_key_extraction = env.state_key_extraction

        # logging grid
        is_minigrid = bool(re.search("minigrid", env.spec.id.lower()))
        self._display_grid = (
            DiscreteGrid(self._evaluation_env)
            if (config.logging.log_grid and is_minigrid)
            else None
        )

        self._total_timesteps = 0
        self._total_updates = 0
        self._episode_num = 0

    def _set_seed(self, seed: int):
        self._env.seed(seed)
        self._env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self):
        """
        Train the AGAC agent.
        """

        # Start training loop
        done = True

        actions = []
        observations = []
        next_observations = []
        rewards = []
        log_pis = []
        dones = []
        state_counts = []
        state_counter = {}

        while self._total_timesteps < self._max_steps:

            if done:
                # Reset environment
                observation = self._env.reset()
                self._episode_num += 1

                # Reset lists
                actions = []
                observations = []
                next_observations = []
                rewards = []
                log_pis = []
                adv_log_pis = []
                logits_pis = []
                adv_logits_pis = []
                dones = []
                state_counts = []
                state_counter = {}

            # Evaluation
            if self._total_timesteps % self._eval_freq == 0:
                # Run evaluation
                evaluation_returns, last_rewards = self.evaluate()
                recvbuf = None
                if rank == 0:
                    num_episodes_eval = self._config.algorithm.num_episodes_eval
                    recvbuf = np.empty(
                        [num_workers, num_episodes_eval], dtype=np.float32
                    )
                comm.Gather(evaluation_returns, recvbuf, root=0)
                last_rewards = comm.gather(last_rewards, root=0)

                if rank == 0:
                    evaluation_returns = recvbuf.flatten()
                    last_rewards = np.array(last_rewards).flatten()

                    # Log stats
                    mean_evaluation_return = np.mean(evaluation_returns)
                    mean_last_reward = np.mean(last_rewards)

                    mean_return_log = LogData(
                        name="evaluation_mean_return",
                        type="scalar",
                        value=mean_evaluation_return,
                    )
                    mean_last_reward_log = LogData(
                        name="evaluation_mean_last_reward",
                        type="scalar",
                        value=mean_last_reward,
                    )
                    steps_log = LogData(
                        name="total_steps",
                        type="scalar",
                        value=self._total_timesteps * num_workers,
                    )

                    actor_weights_log = LogData(
                        name="actor_weights",
                        type="array",
                        value=self._ppo.get_actor_weights(),
                    )
                    current_intrinsic_reward_coefficient_log = LogData(
                        name="training_intrinsic_reward_coefficient",
                        type="scalar",
                        value=float(self._current_intrinsic_reward_coefficient),
                    )

                    logs = [
                        mean_return_log,
                        mean_last_reward_log,
                        steps_log,
                        actor_weights_log,
                        current_intrinsic_reward_coefficient_log,
                    ]

                    if self._extrinsic_returns_stats.mean is not None:
                        mean_extrinsic_returns_log = LogData(
                            name="training_mean_return",
                            type="scalar",
                            value=float(self._extrinsic_returns_stats.mean.copy()),
                        )
                        mean_intrinsic_returns_log = LogData(
                            name="training_mean_intrinsic_return",
                            type="scalar",
                            value=float(self._intrinsic_returns_stats.mean.copy()),
                        )

                        logs += [mean_extrinsic_returns_log, mean_intrinsic_returns_log]
                        if self._display_grid:
                            logs += self._display_grid.logs

                    if self._total_updates > 1:
                        logs += self._ppo.logs

                    self._logger.log(logs)

            # Select action randomly or according to policy
            action, log_pi, adv_log_pi, logits_pi, logits_adv = self._ppo.select_action(
                observation, deterministic=False
            )
            log_pi = float(log_pi)
            adv_log_pi = float(adv_log_pi)

            # Retrieve state count if necessary
            if self._episodic_count:
                state_key = self._state_key_extraction(self._env)
                if state_key in state_counter:
                    state_counter[state_key] += 1
                else:
                    state_counter[state_key] = 1
                state_count = state_counter.get(state_key)
                state_counts.append(state_count)

            # Perform action
            if not self._discrete:
                action = np.clip(action, -self._max_action, self._max_action)
            new_observation, reward, done, _ = self._env.step(action)

            # Save transition
            observations.append(observation.copy())
            next_observations.append(new_observation.copy())
            actions.append(action.copy())
            rewards.append(reward)
            log_pis.append(log_pi)
            adv_log_pis.append(adv_log_pi)
            logits_pis.append(logits_pi)
            adv_logits_pis.append(logits_adv)
            dones.append(float(done))

            # next observation becomes current observation
            observation = deepcopy(new_observation)

            # when episode done or T timesteps reached, process transitions
            # and fill memory
            if self._total_timesteps > 0:
                if done or self._total_timesteps % self._train_freq == 0:
                    self._process_trajectory(
                        observations,
                        next_observations,
                        actions,
                        rewards,
                        log_pis,
                        adv_log_pis,
                        logits_pis,
                        adv_logits_pis,
                        dones,
                        state_counts,
                    )

                # Update
                if self._total_timesteps % self._config.algorithm.train_freq == 0:
                    for _ in range(self._num_epochs):
                        batches = self._memory.get_epoch_batches(self._batch_size)
                        for batch in batches:
                            self._ppo.train_on_batch(
                                batch, self._current_intrinsic_reward_coefficient
                            )

                    # Flush memory
                    self._memory.reset()

                    # Increment updates variables
                    self._total_updates += 1

            # Increment steps variables
            self._total_timesteps += 1

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Play episodes in deterministic modes and return mean return.
        """
        episodes_returns = []
        last_rewards = []
        for _ in range(self._config.algorithm.num_episodes_eval):
            observation = self._evaluation_env.reset()
            # Update grid if display grid
            episode_return = 0.0
            done = False
            while not done:
                action = self._ppo.select_action(observation, deterministic=True)[0]
                if not self._discrete:
                    action = np.clip(action, -self._max_action, self._max_action)
                if self._display_grid:
                    self._display_grid.add()
                new_observation, reward, done, _ = self._evaluation_env.step(action)
                episode_return += reward
                observation = new_observation
                if done and self._display_grid:
                    self._display_grid.reset()
            last_rewards.append(reward)
            episodes_returns.append(episode_return)
        return np.asarray(episodes_returns).astype(np.float32), np.asarray(last_rewards)

    def _process_trajectory(
        self,
        observations: List[np.ndarray],
        next_observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        log_pis: List[np.ndarray],
        adv_log_pis: List[np.ndarray],
        logits_pis: List[np.ndarray],
        adv_logits_pis: List[np.ndarray],
        dones: List[np.ndarray],
        state_counts: List[float],
    ):
        """
        Compute values, returns and advantages when a trajectory is completed
        and add the transitions in the memory.
        """
        observations = np.asarray(observations).astype(np.float32)
        dones = np.asarray(dones).astype(np.float32).flatten()
        extrinsic_rewards = np.asarray(rewards).astype(np.float32).flatten()
        log_pis = np.asarray(log_pis).astype(np.float32).flatten()
        adv_log_pis = np.asarray(adv_log_pis).astype(np.float32).flatten()
        logits_pis = np.asarray(logits_pis).astype(np.float32)
        adv_logits_pis = np.asarray(adv_logits_pis).astype(np.float32)
        if self._episodic_count:
            state_counts = np.asarray(state_counts).astype(np.float32).flatten()
            state_count_rewards = np.floor(1 / np.sqrt(state_counts))

        values = self._ppo.compute_values(observations)

        # Anneal intrinsic rewards coef (linearly annealed to 0):
        progress_fraction = 1.0 - self._total_updates / self._max_updates
        self._current_intrinsic_reward_coefficient = (
            self._intrinsic_reward_coefficient * progress_fraction
        )

        # compute intrinsic rewards
        intrinsic_rewards = log_pis - adv_log_pis

        # update returns stats
        self._intrinsic_returns_stats.add(np.sum(intrinsic_rewards))
        self._extrinsic_returns_stats.add(np.sum(extrinsic_rewards))

        # scale intrinsic rewards
        intrinsic_rewards *= self._current_intrinsic_reward_coefficient

        # compute advantages
        if dones[-1] == 1:
            last_r = 0.0
        else:
            last_next_obs = np.asarray([next_observations[-1]]).astype(np.float32)
            last_r = self._ppo.compute_values(last_next_obs)
            last_r = float(last_r)

        advantages, returns = compute_advantages_and_returns(
            extrinsic_rewards,
            values,
            last_r,
            self._discount,
            self._lambda_gae,
        )

        if self._episodic_count:
            agac_advantages, _ = compute_advantages_and_returns(
                (
                    extrinsic_rewards
                    + self._episodic_count_coefficient * state_count_rewards
                ),
                values,
                last_r,
                self._discount,
                self._lambda_gae,
            )
        else:
            agac_advantages = advantages

        # Add Agac discrepancy
        agac_advantages = agac_advantages + intrinsic_rewards

        # create transitions and add them in memory
        for i in range(len(rewards)):
            transition = Transition(
                observation=observations[i],
                action=actions[i],
                extrinsic_return=returns[i],
                advantage=advantages[i],
                agac_advantage=agac_advantages[i],
                value=values[i],
                log_pi=log_pis[i],
                adv_log_pi=adv_log_pis[i],
                logits_pi=logits_pis[i],
                adv_logits_pi=adv_logits_pis[i],
                done=dones[i],
            )
            self._memory.add(transition)
