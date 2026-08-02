# le nn.Module gaussien (session 1)
import torch

class PolicyRL(torch.nn.Module):
    def __init__(self, log_std_init, delta_max=1, squash=True):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1))
        self.delta_max=delta_max
        self.squash=squash
        log_std = torch.tensor(log_std_init, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, obs):
        mu = self.model(obs).squeeze(-1)
        if self.squash:
            mu = torch.tanh(mu)*self.delta_max
        sigma = torch.exp(self.log_std)
        normal = torch.distributions.Normal(mu, sigma)
        return normal 

    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float)
        distrib = self(obs)
        if deterministic:
            action = distrib.mean
        else:
            action = distrib.sample()
        return action.item()

if __name__ == "__main__":
    print("Starting the script..")
    obs = torch.randn(8, 6)
    policy = PolicyRL()
    distrib = policy(obs)
    action = distrib.sample()
    print("parameters:", sum(p.numel() for p in policy.parameters()))
    print("action", action.shape)
    log_density = distrib.log_prob(action)
    print("log_density", log_density.shape)