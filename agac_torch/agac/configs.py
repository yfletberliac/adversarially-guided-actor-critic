from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


def get_config_from_yaml(config_path: Path) -> "ExperimentConfig":
    """Read yaml file and returns corresponding config object."""
    with open(config_path, "r") as file:
        parameters = yaml.safe_load(file)

    # instantiate config object
    algo_config = AlgorithmConfig(**parameters["algorithm"])
    rl_config = ReinforcementLearningConfig(**parameters["reinforcement_learning"])
    logging_config = LoggingConfig(**parameters["logging"])

    experiment_config = ExperimentConfig(
        algorithm=algo_config,
        reinforcement_learning=rl_config,
        logging=logging_config,
    )

    # checks if the obtained config is valid
    if not experiment_config.is_valid():
        raise ValueError("Tries to instantiate invalid experiment config.")

    return experiment_config


@dataclass
class ExperimentConfig:
    """Class that defines an experiment config."""

    algorithm: "AlgorithmConfig"
    logging: "LoggingConfig"
    reinforcement_learning: "ReinforcementLearningConfig"

    def is_valid(self) -> bool:
        cond = self.algorithm.is_valid()
        cond &= self.reinforcement_learning.is_valid()
        cond &= self.logging.is_valid()
        return cond


@dataclass
class LoggingConfig:
    """Class that defines the general algorithm arguments."""

    save_models: bool
    save_models_freq: int
    use_neptune: bool
    experiment_name: str
    neptune_user_name: str
    neptune_project_name: str
    log_grid: bool

    def is_valid(self) -> bool:
        return True


@dataclass
class AlgorithmConfig:
    """Class that defines the general algorithm arguments."""

    max_steps: int
    eval_freq: int
    num_episodes_eval: int
    train_freq: int
    num_epochs: int
    seed: int = 123
    env_name: str = None
    discrete: bool = False
    gpu: int = -1

    def is_valid(self) -> bool:
        return True


@dataclass
class ReinforcementLearningConfig:
    """Class that defines the config for the RL part of the algorithm."""

    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    adversary_learning_rate: float
    discount: float
    lambda_gae: float
    layers_dim: List[int]
    intrinsic_reward_coefficient: float
    entropy_coefficient: float
    clipping_epsilon: float
    value_loss_clip: float
    value_loss_coeff: float
    adversary_loss_coeff: float
    layers_num_channels: List[int]
    adv_layers_dim: List[int]
    clip_grad_norm: float
    cnn_extractor: bool = False
    nb_stack: int = 1
    episodic_count_coefficient: float = 0.0125

    def is_valid(self) -> bool:
        return True
