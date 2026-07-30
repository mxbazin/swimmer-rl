# boucle d'entraînement (session 2)
import torch
import os

from env import SwimmerEnv
from policy import PolicyRL
import statistics as stats
import numpy as np
from viz import plot_returns

from evaluate import evaluate
from greedy import GreedyPolicy


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(104)

print("Starting the script..")
N= 500
policy = PolicyRL()
env = SwimmerEnv(2, 0.1)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
length_stats = []
retour_stats =[]
flag_list=[]

def function_discounted_return(list_rewards, gamma =0.99):
    taille = len(list_rewards)
    Somme=0
    for t in range(0, taille):
        Somme += (gamma**t) * list_rewards[t]
    return Somme

if __name__ == "__main__":
    for episode in range (N):
        terminated = False
        truncated = False
        obs= env.reset()
        list_logp = [] ; list_reward = []
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float)
            dist = policy(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action)
            list_logp.append(logp)
            action = action.item()
            obs, reward, terminated, truncated, info = env.step(action)
            list_reward.append(reward)

        #Compute the discounted return
        retour_discount = function_discounted_return(list_reward, gamma =1)

        #Compute the loss 
        mean_log_prob = torch.stack(list_logp).mean()
        loss = -mean_log_prob*retour_discount
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        length = len(list_reward) ; retour = sum(list_reward)
        length_stats.append(length) ; retour_stats.append(retour)


        if terminated == True:
            flag_list.append(True)
        elif truncated == True:
            flag_list.append(False)

        ## Print the infromations of the episode every 20 episodes
        if episode % 20 == 0:

            mean_retour_stats = stats.mean(retour_stats[-20:])
            success_win = sum(flag_list[-20:]) 
            sigma = torch.exp(policy.log_std).item()
            print("ep:", episode, "|", "retour", mean_retour_stats, "|", "success", success_win, "/", 20, "|", "sigma", sigma)

    np.save('runs/returns_mean_periodic.npy', retour_stats)
    torch.save(policy.state_dict(), 'runs/policy_mean_periodic.pt')
    plot_returns(retour_stats, 20, f"mean_vswim{env.v_swim}_periodic")

    retour_stats_mean_greedy, success_win_greedy, length_stats_mean_greedy = evaluate(env, GreedyPolicy(), 20, deterministic=True)
    print("GREEDY:", "|", "retour", retour_stats_mean_greedy, "|", "success", success_win_greedy, "/", 20, "|", "length", length_stats_mean_greedy)

    retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env, policy, 20, deterministic=False)
    print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

    retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env, policy, 1, deterministic=True)
    print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 1, "|", "length", length_stats_mean_deterministic)



