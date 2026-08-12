import torch
import numpy as np
from ppo import train
from evaluate import evaluate
import gymnasium as gym 
from viz import plot_returns
import statistics as stats

class PendulumRL():
    def __init__(self,  env, delta_max):
        super().__init__()
        self.delta_max= delta_max
        self.env = env

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs

    def step(self, action):
        action = np.reshape(action, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
## ─── Parameters ───────────────────────
list_seed=[97, 104, 171]

delta_max=2
obs_dim=3
truncated =True

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

K = 10
eps = 0.2
max_grad_norm = 0.5 
entropy_coef=0.05
minibatch_size = 64
N_warmup=1000


if __name__ == "__main__":

    list_synthesis=[]


    for seed in list_seed:

        env_raw = gym.make("Pendulum-v1", g=9.81)
        env_pendulum = PendulumRL(env=env_raw, delta_max=delta_max)
        env_pendulum.reset(seed=seed)

        retour_stats, policy, obs_t_mean, obs_t_std = train(env=env_pendulum, seed=seed, obs_dim=obs_dim, log_std_init=log_std_init, 
                                    squash=squash, lr_loss=lr_loss, lr_critic=lr_critic, 
                                    lam=lam, gamma=gamma, 
                                    packet=packet, n_updates=n_updates, 
                                    K = K, eps = eps, max_grad_norm=max_grad_norm, entropy_coef=entropy_coef, minibatch_size=minibatch_size, N_warmup=N_warmup)

        np.save(f'runs/PENDULUM_returns_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_K_{K}_epsilon_{eps}.npy', retour_stats)
        torch.save(policy.state_dict(), f'runs/PENDULUM_policy_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_epsilon_{eps}.pt')
        plot_returns(retour_stats, 20, f"mean_PENDULUM_periodic_ppo__nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}_epsilon_{eps}")

        retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env_pendulum, policy, 20, deterministic=False, obs_mean=obs_t_mean, obs_std=obs_t_std)
        print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

        retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env_pendulum, policy, 20, deterministic=True, obs_mean=obs_t_mean, obs_std=obs_t_std)
        print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 20, "|", "length", length_stats_mean_deterministic)

        list_synthesis.append( (seed, retour_stats_mean_deterministic, length_stats_mean_deterministic, retour_stats_mean_stoch, length_stats_mean_stoch))

    # Synthesis of the runs
    retour_synthesis_stoch =[]
    retour_synthesis_deterministic =[]

    for element in list_synthesis:
        retour_synthesis_stoch.append(element[3])
        retour_synthesis_deterministic.append(element[1])

    print("median of the return (stoch): ", stats.median(retour_synthesis_stoch), 
            "std of the return (stoch): ", stats.stdev(retour_synthesis_stoch))

    print("median of the return (det): ", stats.median(retour_synthesis_deterministic), 
            "std of the return (det): ", stats.stdev(retour_synthesis_deterministic))