import torch

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1))

    def forward(self, obs):
        value = self.model(obs).squeeze(-1)
        return value