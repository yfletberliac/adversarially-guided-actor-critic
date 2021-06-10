import numpy as np
import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


def gather_global_data(comm, *args):
    """Takes an arbitrary number (say N) of variables (*args), aggregates each variable
    among all workers and outputs N numpy arrays being aggregations of these variables
    Ex:
    Worker 1 has x = 20, y = 40
    Worker 2 has x = 21, y = 41
    global_x, global_y = gather_global_data(comm, x, y)
    All workers have global_x = [[20],[21]], global_y = [[40],[41]]
    """
    locals = [np.array([arg]) if type(arg) is not np.ndarray else arg for arg in args]
    globals = [
        np.empty((comm.Get_size(), *locals[i].shape), dtype=locals[i].dtype)
        for i in range(len(locals))
    ]
    for i in range(len(locals)):
        comm.Allgather(locals[i], globals[i])
    if len(globals) == 1:
        return globals[0]
    else:
        return tuple(globals)


def sync_tensor_grads(tensor, comm=None):
    """Sync for scalar tensors of shape (1,)"""
    assert tensor.shape == (1,), "non supported tensor shape"
    if comm is None:
        comm = MPI.COMM_WORLD
    flat_grads = tensor.grad.numpy().flatten()
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    global_grads /= comm.Get_size()
    tensor.grad[0] = torch.tensor(global_grads[0])


def sync_networks(network, comm=None):
    """Synchronize neural nets between threads"""
    if comm is None:
        comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode="params")
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode="params")


def sync_grads(network, comm=None):
    """Synchronize gradients between threads"""
    if comm is None:
        comm = MPI.COMM_WORLD
    flat_grads = _get_flat_params_or_grads(network, mode="grads")
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    global_grads /= comm.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode="grads")


def print_from_master_node(s, end=None):
    if MPI.COMM_WORLD.Get_rank() == 0:
        if end is not None:
            print(s, end=end)
        else:
            print(s)


def _get_flat_params_or_grads(network, mode="params"):
    """include two kinds: grads and params"""
    attr = "data" if mode == "params" else "grad"
    params_or_grads_list = [getattr(param, attr) for param in network.parameters()]
    params_or_grads_list = [
        p.cpu().numpy().flatten() for p in params_or_grads_list if p is not None
    ]
    return np.concatenate(params_or_grads_list)


def _set_flat_params_or_grads(network, flat_params, mode="params"):
    """include two kinds: grads and params"""
    attr = "data" if mode == "params" else "grad"
    # the pointer
    pointer = 0
    for param in network.parameters():
        if getattr(param, attr) is not None:
            getattr(param, attr).copy_(
                torch.tensor(
                    flat_params[pointer : pointer + param.data.numel()]
                ).view_as(param.data)
            )
            pointer += param.data.numel()


def sync_buffers(
    replay_buffer_per_worker, replay_buffer, store_in_32=True, between_threads=True
):
    """Synchronize buffers between threads"""
    if between_threads:
        comm = MPI.COMM_WORLD
        buffers = comm.allgather(replay_buffer_per_worker._storage)
        for buffer in buffers:
            for i in range(len(buffer)):
                if store_in_32:
                    obs, n_obs, action, reward, done_bool, behavior = buffer[i]
                    replay_buffer.add(
                        (
                            obs.astype("f"),
                            n_obs.astype("f"),
                            action,
                            reward,
                            done_bool,
                            behavior.astype("f"),
                        )
                    )
                else:
                    replay_buffer.add(buffer[i])
    else:
        for i in range(len(replay_buffer_per_worker)):
            if store_in_32:
                (
                    obs,
                    n_obs,
                    action,
                    reward,
                    done_bool,
                    behavior,
                ) = replay_buffer_per_worker[i]
                replay_buffer.add(
                    (obs.astype("f"), n_obs.astype("f"), action, reward, done_bool),
                    behavior.astype("f"),
                )
            else:
                replay_buffer.add(replay_buffer_per_worker[i])

    replay_buffer_per_worker.reset()


def gather_buffers(buffers, replay_buffer, store_in_32=False):
    for buffer in buffers:
        for i in range(len(buffer)):
            if store_in_32:
                obs, n_obs, action, reward, done_bool, behavior = buffer[i]
                replay_buffer.add(
                    (
                        obs.astype("f"),
                        n_obs.astype("f"),
                        action,
                        reward,
                        done_bool,
                        behavior.astype("f"),
                    )
                )
            else:
                replay_buffer.add(buffer[i])


def sync_archive(
    archive,
    params_per_worker,
    behavior_per_worker,
    fitness_per_worker,
    from_novelty_per_worker,
):
    """Synchronize archives between threads"""
    comm = MPI.COMM_WORLD
    all_params = comm.allgather(params_per_worker)
    all_behaviors = comm.allgather(behavior_per_worker)
    all_fitnesses = comm.allgather(fitness_per_worker)
    all_from_novelties = comm.allgather(from_novelty_per_worker)
    for ind in range(len(all_params)):
        archive.add(
            all_params[ind],
            all_behaviors[ind],
            all_fitnesses[ind],
            all_from_novelties[ind],
        )
