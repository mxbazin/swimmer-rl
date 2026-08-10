import torch
import numpy as np
from ppo import train
from evaluate import evaluate
import gymnasium as gym 
from viz import plot_returns

class PendulumRL():
    def __init__(self,  env, delta_max):
        super().__init__()
        self.delta_max= delta_max
        self.env = env

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        action = np.reshape(action, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
## ─── Parameters ───────────────────────
seed=104

delta_max=2
obs_dim=3
truncated =True
env_raw = gym.make("Pendulum-v1", g=9.81)
env_pendulum = PendulumRL(env=env_raw, delta_max=delta_max)

list_mode = 'gae'
lr_loss = 1e-3
lr_critic = 1e-2

packet=8
n_updates=300

shaping = True
log_std_init = -1

lam = 0.95
gamma=0.9
squash=True
normalize ='steps'

K = 4
eps = 0.2

if __name__ == "__main__":

    retour_stats, policy = train(env=env_pendulum, seed=seed, obs_dim=obs_dim, log_std_init=log_std_init, squash=squash, lr_loss=lr_loss, lr_critic=lr_critic, lam=lam, gamma=gamma, packet=packet, n_updates=n_updates, K = K, eps = eps)

    np.save(f'runs/returns_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_K_{K}_epsilon_{eps}.npy', retour_stats)
    torch.save(policy.state_dict(), f'runs/policy_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_epsilon_{eps}.pt')
    plot_returns(retour_stats, 20, f"mean_vPENDULUM_periodic_ppo__nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_epsilon_{eps}")

    retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env_pendulum, policy, 20, deterministic=False)
    print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

    retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env_pendulum, policy, 1, deterministic=True)
    print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 1, "|", "length", length_stats_mean_deterministic)
