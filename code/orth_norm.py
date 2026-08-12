import torch

def ortho_weights(layer, gain):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)
    return layer