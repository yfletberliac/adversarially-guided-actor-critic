import dataclasses
import os
import time
from collections import namedtuple
from datetime import datetime
from typing import List

import neptune.new as neptune
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
from neptune.new.types import File
from torch.utils.tensorboard import SummaryWriter

from agac.configs import ExperimentConfig

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


LogData = namedtuple("LogData", ["name", "type", "value"])


class Logger:
    """
    Class used to log training evolution.
    """

    def __init__(self, config: ExperimentConfig):
        self.start_time = time.time()
        self._config = config

        # Get params
        logging_config = config.logging
        self.dir = "."
        self._use_neptune = logging_config.use_neptune

        # Initialize neptune
        if self._use_neptune:
            user_name = logging_config.neptune_user_name
            project_name = logging_config.neptune_project_name
            self._neptune_run = neptune.init(user_name + "/" + project_name)
            self._neptune_run["parameters"] = dataclasses.asdict(config)
            tags = [logging_config.experiment_name] + [config.algorithm.env_name]
            self._neptune_run["sys/tags"].add(tags)

        # Create directories
        self._create_dirs(
            config.algorithm.env_name,
            config.algorithm.seed,
            config.logging.save_models,
        )

        # Dump config file
        with open(self._save_dir + "/" + "config.yaml", "w") as cfg:
            yaml.dump(dataclasses.asdict(config), cfg, default_flow_style=True)

        # Dump config
        config_str = yaml.dump(dataclasses.asdict(config), default_flow_style=True)
        self._summary_writer.add_text("config", config_str, 0)

        self._logs_df = None

        self.print_experiment_info()

    def print_experiment_info(self):
        print("---------------------------------")
        env_name = self._config.algorithm.env_name
        print("Env: {}".format(env_name), flush=True)
        print("Agent: AGAC PPO", flush=True)
        print("Seed: {}".format(self._config.algorithm.seed), flush=True)
        print("---------------------------------")

        rl_config = self._config.reinforcement_learning
        learning_rate = rl_config.actor_learning_rate
        print(" actor lr: {}".format(learning_rate), flush=True)
        learning_rate = rl_config.critic_learning_rate
        print(" critic lr: {}".format(learning_rate), flush=True)
        learning_rate = rl_config.adversary_learning_rate
        print(" adversary lr: {}".format(learning_rate), flush=True)
        layers_dim = rl_config.layers_dim
        print("layers_dim: {}".format(layers_dim), flush=True)
        print("---------------------------------")

    def _create_dirs(self, env_name, seed, save_models):
        # create directories
        save_dir = "./runs/" + "%s_%s_" % ("AGAC", env_name)
        save_dir += datetime.now().strftime("%b%d_%H-%M-%S")
        save_dir += "_%s" % seed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._save_dir = save_dir
        self._summary_writer = SummaryWriter(self._save_dir)

        if save_models and not os.path.exists(self._save_dir + "/models"):
            os.makedirs(self._save_dir + "/models")

    def log(self, logs: List[LogData]):
        """
        Log data to different channels and save model if required.
        """
        # get number of steps
        steps = [log.value for log in logs if log.name == "total_steps"][0]

        # log
        self._log_in_tensorboard(steps, logs)
        self._log_in_file(logs)
        if self._use_neptune:
            self._log_in_neptune(logs)

        # save actors if needed
        if self._config.logging.save_models:
            weights = [log.value for log in logs if log.name == "actor_weights"][0]
            self._save_actor_weights(weights, steps)

    def _log_in_tensorboard(self, steps: int, logs: List[LogData]):
        """
        Log data in tensorboard.
        """

        for log in logs:
            if log.type == "scalar":
                self._summary_writer.add_scalar(log.name, log.value, steps)
            if log.type == "image":
                self._summary_writer.add_image(log.name, log.value, steps)

    def _log_in_neptune(self, logs: List[LogData]):
        """
        Log data in neptune.
        """
        # log scalar metrics
        try:
            for log in logs:
                if log.type == "scalar":
                    self._neptune_run[log.name].log(log.value)
                if log.type == "image":
                    if (log.value.ndim == 3) and (log.value.shape[-1] != 3):
                        self._neptune_run[log.name].log(
                            File.as_image(log.value.transpose(1, 2, 0))
                        )
                    else:
                        self._neptune_run[log.name].log(File.as_image(log.value))

            # log also csv and pkl files
            # self._neptune_run["logs.csv"].upload(self._save_dir + "/logs.csv")
            # self._neptune_run["logs.pkl"].upload(self._save_dir + "/logs.pkl")
            # neptune.log_artifact(self._save_dir + "/logs.csv")
            # neptune.log_artifact(self._save_dir + "/logs.pkl")
        except RuntimeError:
            print("WARNING: failed to log in Neptune")

    def _log_in_file(self, logs: List[LogData]):
        """
        Log data in tensorboard.
        """
        df_dict = {}
        for log in logs:
            if log.type == "scalar":
                df_dict[log.name] = log.value
            elif log.type == "array":
                if "weights" not in log.name:
                    df_dict[log.name] = log.value

        if self._logs_df is None:
            # if not already done, create dataframe
            for key, value in df_dict.items():
                df_dict[key] = [value]
            self._logs_df = pd.DataFrame(data=df_dict)

        else:
            # otherwise, add data in dataframe
            self._logs_df = self._logs_df.append(df_dict, ignore_index=True)

        # save in csv format
        self._logs_df.to_csv(self._save_dir + "/logs.csv")
        # save in pickle format
        self._logs_df.to_pickle(self._save_dir + "/logs.pkl")

    def _save_actor_weights(self, actor_weights, steps):
        """
        Save the weights of the population actors.
        """
        path = self._save_dir + "/models/agent_steps_{}".format(steps)
        np.save(path, actor_weights)
