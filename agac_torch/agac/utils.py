from typing import List, Tuple

import gym_minigrid
import numpy as np
import scipy.signal

from agac.logger import LogData


def discount(x: np.ndarray, gamma: float):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_advantages_and_returns(rewards, values, last_r, gamma, lambda_gae):
    values = np.append(values, last_r)
    rewards = np.append(rewards, last_r)
    delta_t = rewards[:-1] + gamma * values[1:] - values[:-1]
    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    advantages = discount(delta_t, gamma * lambda_gae)
    returns = discount(rewards, gamma)[:-1]
    return advantages.copy().astype(np.float32), returns.copy().astype(np.float32)


class DiscreteGrid:
    def __init__(self, env: gym_minigrid.minigrid.MiniGridEnv):
        self._env = env
        self._tile_size = 32
        self._visit_weight = 20
        self._agent_positions = []

        self._display_grids = []
        self._grid = None

    def add(self):
        if self._grid is None:
            self._grid = self._env.grid.render(self._tile_size)
        self._agent_positions.append(self._env.agent_pos)

    def reset(self):
        self._display_grids.append((self._grid, self._agent_positions))
        self._grid = None
        self._agent_positions = []

    def _grid_coverage(
        self, grid: np.ndarray, positions: List[Tuple[int]]
    ) -> np.ndarray:
        """
        Returns an image representing the grid coverage over 1 episode.
        """
        img = grid
        for pos in positions:
            img[
                pos[1] * self._tile_size : (pos[1] + 1) * self._tile_size,
                pos[0] * self._tile_size : (pos[0] + 1) * self._tile_size,
                2,
            ] += self._visit_weight

        img[img > 255] = 255
        return img.transpose(-1, 0, 1)

    @property
    def logs(self) -> List[LogData]:
        logs = [
            LogData(
                name="grid_coverage",
                value=self._grid_coverage(grid, positions),
                type="image",
            )
            for grid, positions in self._display_grids
        ]
        self._display_grids = []
        return logs
