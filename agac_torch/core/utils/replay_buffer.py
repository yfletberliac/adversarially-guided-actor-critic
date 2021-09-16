import operator
from collections import namedtuple

import numpy as np

Transition = namedtuple(
    "Transition",
    ["observation", "next_observation", "action", "reward", "done", "descriptors"],
)

Batch = namedtuple(
    "Batch",
    ["observations", "next_observations", "actions", "rewards", "dones", "descriptors"],
)


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self._storage = []
        self._max_size = max_size
        self._pointer_position = 0

    @property
    def num_elements(self):
        return len(self._storage)

    def add(self, transition: Transition):
        if len(self._storage) == self._max_size:
            self._storage[int(self._pointer_position)] = transition
            self._pointer_position = (self._pointer_position + 1) % self._max_size
        else:
            self._storage.append(transition)

    def reset(self):
        self._storage = []
        self._pointer_position = 0

    def sample(self, batch_size: int) -> Batch:

        if len(self._storage) < batch_size:
            raise ValueError("Not enough data in replay buffer to sample a batch.")

        else:
            ind = np.random.randint(0, len(self._storage), size=batch_size)
            op = operator.itemgetter(*ind)
            x, y, u, r, d, o = list(zip(*op(self._storage)))

            batch = Batch(
                observations=np.array(x).copy(),
                next_observations=np.array(y).copy(),
                actions=np.array(u).copy(),
                rewards=np.array(r).copy(),
                dones=np.array(d).copy(),
                descriptors=np.array(o).copy(),
            )

            return batch

    def save(self, outfile):
        np.save(outfile, self._storage)
        print(f"* {outfile} succesfully saved..")
