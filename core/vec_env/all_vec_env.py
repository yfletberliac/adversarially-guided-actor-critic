import multiprocessing
from collections import OrderedDict
from typing import Sequence
import pickle

import gym
from gym import spaces
import numpy as np

from core.math_util import RunningMeanStd
from core.vec_env.base_vec_env import VecEnv, CloudpickleWrapper, VecEnvWrapper
from core.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

import warnings


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.

    It is pickleable which will save moving averages and configuration parameters.
    The wrapped environment `venv` is not saved, and must be restored manually with
    `set_venv` after being unpickled.

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = None
        self.old_rews = None

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state['venv']
        del state['class_attributes']
        # these attributes depend on the above and so we would prefer not to pickle
        del state['ret']
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state: (dict)"""
        self.__dict__.update(state)
        assert 'venv' not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv: (VecEnv)
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)
        if self.obs_rms.mean.shape != self.observation_space.shape:
            raise ValueError("venv is incompatible with current statistics.")
        self.ret = np.zeros(self.num_envs)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_rews = rews

        if self.training:
            self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rews)
        rews = self.normalize_reward(rews)

        self.ret[news] = 0
        return obs, rews, news, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs,
                          self.clip_obs)
        return obs

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon),
                             -self.clip_reward, self.clip_reward)
        return reward

    def get_original_obs(self) -> np.ndarray:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return self.old_obs.copy()

    def get_original_reward(self) -> np.ndarray:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_rews.copy()

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        if self.training:
            self._update_reward(self.ret)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path, venv):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return: (VecNormalize)
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn("Usage of `save_running_average` is deprecated. Please "
                      "use `save` or pickle instead.", DeprecationWarning)
        for rms, name in zip([self.obs_rms, self.ret_rms], ['obs_rms', 'ret_rms']):
            with open("{}/{}.pkl".format(path, name), 'wb') as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path):
        """
        :param path: (str) path to log dir

        .. deprecated:: 2.9.0
            This function will be removed in a future version
        """
        warnings.warn("Usage of `load_running_average` is deprecated. Please "
                      "use `load` or pickle instead.", DeprecationWarning)
        for name in ['obs_rms', 'ret_rms']:
            with open("{}/{}.pkl".format(path, name), 'rb') as file_handler:
                setattr(self, name, pickle.load(file_handler))


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment

    :param venv: (VecEnv) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """

    def __init__(self, venv, n_stack):
        self.venv = venv
        self.n_stack = n_stack
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        last_ax_size = observations.shape[-1]
        self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
        for i, done in enumerate(dones):
            if done:
                if 'terminal_observation' in infos[i]:
                    old_terminal = infos[i]['terminal_observation']
                    new_terminal = np.concatenate(
                        (self.stackedobs[i, ..., :-last_ax_size], old_terminal), axis=-1)
                    infos[i]['terminal_observation'] = new_terminal
                else:
                    warnings.warn(
                        "VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        self.stackedobs[..., -observations.shape[-1]:] = observations
        return self.stackedobs, rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multithread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = \
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        return [env.render(*args, mode='rgb_array', **kwargs) for env in self.envs]

    def render(self, *args, **kwargs):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via `BaseVecEnv.render()`.
        Otherwise (if `self.num_envs == 1`), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as `mode` will have values that are valid
        only when `num_envs == 1`.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
