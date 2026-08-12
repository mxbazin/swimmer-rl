import torch
import orth_norm
import numpy as np

class Critic(torch.nn.Module):
    def __init__(self, obs_dim_critic):
        super().__init__()
        self.model = torch.nn.Sequential(
            orth_norm.ortho_weights(torch.nn.Linear(obs_dim_critic, 64), np.sqrt(2)),
            torch.nn.Tanh(),
            orth_norm.ortho_weights(torch.nn.Linear(64, 64), np.sqrt(2)),
            torch.nn.Tanh(),
            orth_norm.ortho_weights(torch.nn.Linear(64, 1), 1))

    def forward(self, obs):
        value = self.model(obs).squeeze(-1)
        return value

