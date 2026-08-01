# boucle d'entraînement (session 2)
import torch
import os

from env import SwimmerEnv
from greedy import GreedyPolicy
from policy import PolicyRL
import statistics as stats
import numpy as np
from viz import plot_returns
from evaluate import evaluate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def function_discounted_return(list_rewards, gamma =0.99):
    taille = len(list_rewards)
    Somme=0
    for t in range(0, taille):
        Somme += (gamma**t) * list_rewards[t]
    return Somme

def function_reward_to_go_return(list_rewards, gamma=1):
    n = len(list_rewards)
    rtg_sum = np.zeros_like(list_rewards, dtype=np.float64)
    for t in reversed(range(n)):
        rtg_sum[t] = list_rewards[t] + gamma*(rtg_sum[t+1] if t+1 < n else 0)
    return rtg_sum

# if __name__ == "__main__":
#     test = [1, 1, 1]
#     rtg_sum_test = function_reward_to_go_return(test, gamma = 0.5)
#     print(rtg_sum_test)

if __name__ == "__main__":

    list_seed=[97, 104, 171]
    list_lr=[1e-2, 1e-3]
    #list_mode = ['rtg', 'discounted']
    list_mode = ['discounted']


    for mode in (list_mode):
        for lr in (list_lr):
            for seed in (list_seed):

                torch.manual_seed(seed)
                print("SEED:", seed, "MODE:", mode, "lr:", lr)
                length_stats = []
                retour_stats =[]

                print("Starting the script..")
                N= 500
                policy = PolicyRL()
                env = SwimmerEnv(2, 0.1)
                optimizer = torch.optim.Adam(policy.parameters(),lr=lr)
                flag_list=[]

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

                    if mode == 'discounted': 
                        retour_discount = function_discounted_return(list_reward, gamma =1)
                        loss = -torch.stack(list_logp).mean() * retour_discount

                    elif mode == 'rtg': 
                        return_rtg = function_reward_to_go_return(list_reward, gamma =1)
                        return_rtg= torch.as_tensor(return_rtg)

                        #print("shape list_logp", torch.stack(list_logp).shape, "shape return_rtg", np.shape(return_rtg))

                        weighted_logp = torch.stack(list_logp)*return_rtg
                        loss = -weighted_logp.mean()

                    else: 
                        print("Error")

                    #Compute the loss 
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

                np.save(f'runs/returns_mean_periodic_{mode}_lr{lr}_seed{seed}.npy', retour_stats)
                torch.save(policy.state_dict(), f'runs/policy_mean_periodic_{mode}_lr{lr}_seed{seed}.pt')
                plot_returns(retour_stats, 20, f"mean_vswim{env.v_swim}_periodic_{mode}_lr{lr}_seed{seed}")

                retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env, policy, 20, deterministic=False)
                print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

                retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env, policy, 1, deterministic=True)
                print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 1, "|", "length", length_stats_mean_deterministic)

            retour_stats_mean_greedy, success_win_greedy, length_stats_mean_greedy = evaluate(env, GreedyPolicy(), 20, deterministic=True)
            print("GREEDY:", "|", "retour", retour_stats_mean_greedy, "|", "success", success_win_greedy, "/", 20, "|", "length", length_stats_mean_greedy)



