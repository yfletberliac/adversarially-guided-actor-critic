from typing import List

import gym_minigrid
import numpy as np
import scipy.signal

from agac.configs import ExperimentConfig
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


class Grid:
    def __init__(self, config: ExperimentConfig):
        logging_config = config.logging
        self._dim_bc = logging_config.num_descriptors
        self._num_cells_per_dimension = logging_config.num_cells_per_dimension
        self._bd_min_values = logging_config.min_values
        self._bd_max_values = logging_config.max_values

        # array containing all cell ids
        self.cell_ids = np.arange(self._num_cells_per_dimension ** self._dim_bc)
        grid_shape = [self._num_cells_per_dimension] * self._dim_bc
        self.cell_ids = self.cell_ids.reshape(grid_shape)

        total_num_cells = self._num_cells_per_dimension ** self._dim_bc
        self.cells = [None for _ in range(total_num_cells)]

        # Define boundaries
        self.boundaries = []  # boundaries for each cell
        self.cell_sizes = []  # compute cell size
        for i in range(self._dim_bc):
            bc_min = self._bd_min_values[i]
            bc_max = self._bd_max_values[i]
            boundaries = np.arange(
                bc_min, bc_max + 1e-5, (bc_max - bc_min) / self._num_cells_per_dimension
            )
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf
            self.boundaries.append(boundaries)
            self.cell_sizes.append((bc_max - bc_min) / self._num_cells_per_dimension)

    def add(self, behavior):
        cell_id = self.find_cell_id(behavior)
        if self.cells[cell_id] is None:
            self.cells[cell_id] = 1

    def find_cell_id(self, bc):
        """
        Find cell identifier of the BC map corresponding to bc.
        """
        coords = []
        for j in range(self._dim_bc):
            inds = np.atleast_1d(np.argwhere(self.boundaries[j] < bc[j]).squeeze())
            coords.append(inds[-1])
        coords = tuple(coords)
        cell_id = self.cell_ids[coords]
        return cell_id

    def _grid_coverage(self):
        """
        Returns an image representing the grid coverage.
        """
        num_cells_per_dimension = self._num_cells_per_dimension
        img = np.zeros((3, num_cells_per_dimension, num_cells_per_dimension))
        for i in range(num_cells_per_dimension):
            for j in range(num_cells_per_dimension):
                if self.cells[i * num_cells_per_dimension + j] is not None:
                    img[0][i][j] = 1
        return img

    def _grid_percentage_filled(self):
        """
        Returns the percentage of the grid that has been visited
        """
        return sum(x is not None for x in self.cells) / len(self.cells)

    @property
    def logs(self) -> List[LogData]:
        logs = []
        logs.append(
            LogData(name="grid_coverage", value=self._grid_coverage(), type="image")
        )
        logs.append(
            LogData(
                name="grid_percentage_filled",
                value=self._grid_percentage_filled(),
                type="scalar",
            )
        )
        return logs


class DiscreteGrid:
    def __init__(self, env: gym_minigrid.minigrid.MiniGridEnv):
        self._env = env
        self._tile_size = 32
        self._visit_weight = 20
        self._agent_positions = []

        self.display_grids = []
        self._grid = None

    def add(self):
        self._agent_positions.append(self._env.agent_pos)

    def find_cell_id(self, bc):
        """
        Find cell identifier of the BC map corresponding to bc.
        """
        pass

    def add_grid(self):
        self._grid = self._env.grid.render(self._tile_size)

    def _grid_coverage(self) -> np.ndarray:
        """
        Returns an image representing the grid coverage over 1 episode.
        """
        img = self._grid
        for pos in self._agent_positions:
            img[
                pos[1] * self._tile_size : (pos[1] + 1) * self._tile_size,
                pos[0] * self._tile_size : (pos[0] + 1) * self._tile_size,
                2,
            ] += self._visit_weight

        img[img > 255] = 255
        self._agent_positions = []
        self._grid = None
        return img.transpose(-1, 0, 1)

    @property
    def logs(self) -> List[LogData]:
        logs = []
        logs.append(
            LogData(name="grid_coverage", value=self._grid_coverage(), type="image")
        )
        return logs
