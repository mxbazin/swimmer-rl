import torch

class Critic(torch.nn.Module):
    def __init__(self, obs_dim_critic):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim_critic,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1))

    def forward(self, obs):
        value = self.model(obs).squeeze(-1)
        return value