import argparse

import gym
from gym_minigrid.minigrid import MiniGridEnv

from agac.agac_trainer import AGAC
from agac.configs import get_config_from_yaml
from core.utils.envs import EpisodicCountWrapper, MinigridWrapper


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="agac_torch/configs/minigrid.yaml",
        help="config",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="MiniGrid-MultiRoom-N10-S10-v0",
        help="gym env name",
    )
    parser.add_argument("--seed", type=int, default=123, help="seed number")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser_args()

    # read and merge configs
    config = get_config_from_yaml(args.config_path)
    config.algorithm.seed = args.seed
    config.algorithm.env_name = args.env_name

    env = gym.make(args.env_name)
    if isinstance(env, MiniGridEnv):
        env = MinigridWrapper(env, num_stack=4)

        def state_key_extraction(env):
            return tuple(env.agent_pos)

    else:

        def state_key_extraction(env):
            return env.get_grid_pos()

    if config.reinforcement_learning.episodic_count_coefficient > 0:
        env = EpisodicCountWrapper(env=env, state_key_extraction=state_key_extraction)

    # create trainer and train
    agac = AGAC(config, env)
    agac.train()
