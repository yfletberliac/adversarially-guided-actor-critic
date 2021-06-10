import numpy as np


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_hidden_layer(layer, b_init_value=0.1):
    fanin_init(layer.weight)
    layer.bias.data.fill_(b_init_value)


def initialize_last_layer(layer, init_w=1e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
