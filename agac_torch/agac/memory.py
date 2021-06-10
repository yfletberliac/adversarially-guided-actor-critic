import operator
from collections import namedtuple
from typing import List

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


Transition = namedtuple(
    "Transition",
    [
        "observation",
        "action",
        "extrinsic_return",
        "advantage",
        "agac_advantage",
        "value",
        "log_pi",
        "adv_log_pi",
        "logits_pi",
        "adv_logits_pi",
        "done",
    ],
)

Batch = namedtuple(
    "Batch",
    [
        "observations",
        "actions",
        "extrinsic_returns",
        "advantages",
        "agac_advantages",
        "values",
        "log_pis",
        "adv_log_pis",
        "logits_pi",
        "adv_logits_pi",
        "dones",
    ],
)


class Memory(object):
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
            (
                observation,
                action,
                extrinsic_return,
                advantage,
                agac_advantage,
                value,
                log_pi,
                adv_log_pi,
                logits_pi,
                adv_logits_pi,
                done,
            ) = list(zip(*op(self._storage)))

            batch = Batch(
                observations=np.array(observation).copy(),
                actions=np.array(action).copy(),
                extrinsic_returns=np.array(extrinsic_return).copy(),
                advantages=np.array(advantage).copy(),
                agac_advantages=np.array(agac_advantage).copy(),
                values=np.array(value).copy(),
                log_pis=np.array(log_pi).copy(),
                adv_log_pis=np.array(adv_log_pi).copy(),
                logits_pi=np.array(logits_pi).copy(),
                adv_logits_pi=np.array(adv_logits_pi).copy(),
                dones=np.array(done).copy(),
            )

            return batch

    def get_advantages_stats(self):
        _, _, _, advantages, agac_advantages, _, _, _, _, _, _ = list(
            zip(*self._storage)
        )
        advantages = np.array(advantages).copy()
        agac_advantages = np.array(agac_advantages).copy()

        # Share advantages between processes to get more accurate stats
        advantages = comm.allgather(advantages)
        advantages = np.concatenate(tuple(advantages))

        agac_advantages = comm.allgather(agac_advantages)
        agac_advantages = np.concatenate(tuple(agac_advantages))

        return (
            advantages.mean(),
            advantages.std(),
            agac_advantages.mean(),
            agac_advantages.std(),
        )

    def get_epoch_batches(self, batch_size: int) -> List[Batch]:

        if len(self._storage) < batch_size:
            raise ValueError("Not enough data in replay buffer to sample a batch.")

        else:
            idxes = np.arange(len(self._storage))
            np.random.shuffle(idxes)
            num_batches = len(self._storage) // batch_size
            if len(self._storage) > num_batches * batch_size:
                num_batches += 1
            num_batches = np.min(comm.allgather(num_batches))
            idxes = idxes[: (num_batches * batch_size)]

            adv_mean, adv_std, agac_adv_mean, agac_adv_std = self.get_advantages_stats()

            batches = []
            for i in range(num_batches):
                ind = idxes[i::num_batches]
                op = operator.itemgetter(*ind)
                (
                    observation,
                    action,
                    extrinsic_return,
                    advantage,
                    agac_advantage,
                    value,
                    log_pi,
                    adv_log_pi,
                    logits_pi,
                    adv_logits_pi,
                    done,
                ) = list(zip(*op(self._storage)))

                batch = Batch(
                    observations=np.array(observation).copy(),
                    actions=np.array(action).copy(),
                    extrinsic_returns=np.array(extrinsic_return).copy(),
                    advantages=np.array(advantage).copy(),  # unnormalized
                    agac_advantages=(np.array(agac_advantage).copy() - agac_adv_mean)
                    / agac_adv_std,
                    values=np.array(value).copy(),
                    log_pis=np.array(log_pi).copy(),
                    adv_log_pis=np.array(adv_log_pi).copy(),
                    logits_pi=np.array(logits_pi).copy(),
                    adv_logits_pi=np.array(adv_logits_pi).copy(),
                    dones=np.array(done).copy(),
                )

                batches.append(batch)

            return batches

    def save(self, outfile):
        np.save(outfile, self._storage)
        print(f"* {outfile} succesfully saved..")
