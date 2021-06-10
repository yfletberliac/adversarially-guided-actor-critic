from typing import Union

import torch
from torch.distributions import Categorical, Normal

Action = Union[int, torch.Tensor]
LogProb = torch.Tensor
Prob = torch.Tensor
Observation = torch.Tensor
Distribution = Union[Categorical, Normal]
