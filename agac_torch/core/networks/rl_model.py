from typing import Any, List, NewType, Optional

import numpy as np
import torch
import torch.nn as nn

Parameters = NewType("Parameters", np.ndarray)
Gradients = List[Optional[Any]]


class RLModel(nn.Module):
    def __init__(self):
        super(RLModel, self).__init__()

    def set_params(self, params: Parameters):
        """Set the params of the network to the given parameters"""
        self.load_state_dict(params)

    def get_params(self) -> Parameters:
        """Return parameters of the actor"""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def get_size(self):
        """Return the number of parameters of the network"""
        return self.get_params().shape[0]

    def get_grads(self) -> Gradients:
        """
        Get gradients.
        """
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_grads(self, gradients: Gradients):
        """Set gradients"""
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

    def load_model(self, filename, net_name):
        """Load the model"""
        if filename is None:
            return

        self.load_state_dict(
            torch.load(
                "{}/{}.pkl".format(filename, net_name),
                map_location=lambda storage, loc: storage,
            )
        )

    def save_model(self, output, net_name):
        """Saves the model"""
        torch.save(self.state_dict(), "{}/{}.pkl".format(output, net_name))
