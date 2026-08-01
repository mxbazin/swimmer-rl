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

# Testing the function reward to go
# if __name__ == "__main__":
#     test = [1, 1, 1]
#     rtg_sum_test = function_reward_to_go_return(test, gamma = 0.5)
#     print(rtg_sum_test)

def train(env, seed, lr, mode, packet=8, baseline=True, normalize='episode', n_updates=63):

    print("SEED:", seed, "MODE:", mode, "lr:", lr)
    print("Starting the script..")

    torch.manual_seed(seed)
    policy = PolicyRL()
    optimizer = torch.optim.Adam(policy.parameters(),lr=lr)

    length_stats = []
    retour_stats =[]
    flag_list=[]

    for update in range(n_updates):
        batch_logp=[]
        batch_weights=[]

        for episode in range (packet):
            terminated = False
            truncated = False
            obs= env.reset()
            episode_logp = [] ; episode_reward = []

            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float)
                dist = policy(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                episode_logp.append(logp)
                action = action.item()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward.append(reward)

            if mode == 'discounted': 
                retour_discount = function_discounted_return(episode_reward, gamma =1)
                batch_weights.extend([retour_discount] * len(episode_logp))

            elif mode == 'rtg': 
                return_rtg = function_reward_to_go_return(episode_reward, gamma =1)
                batch_weights.extend(return_rtg)

            else: 
                raise ValueError(f"mode inconnu : {mode}")


            batch_logp.extend(episode_logp)

            #Statistics
            length = len(episode_reward) ; retour = sum(episode_reward)
            length_stats.append(length) ; retour_stats.append(retour)

            if terminated == True:
                flag_list.append(True)
                
            elif truncated == True:
                flag_list.append(False)

        #Compute the loss 
        batch_logp = torch.stack(batch_logp)
        batch_weights = torch.as_tensor(batch_weights, dtype=torch.float)

        if baseline:
            batch_weights_meaned = batch_weights.mean()
        else: 
            batch_weights_meaned = 0 

        batch_weights_baselined = batch_weights - batch_weights_meaned

        if normalize == 'episode':
            loss = - ((batch_weights_baselined * batch_logp).sum())/packet

        elif normalize == 'steps':
            loss = - (batch_weights_baselined * batch_logp).mean()

        else: 
            raise ValueError(f"normalize inconnu : {normalize}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Print the informations every 5 episodes
        if update % 5 == 0:
            mean_retour_stats = stats.mean(retour_stats[-packet:])
            success_win = sum(flag_list[-packet:]) 
            sigma = torch.exp(policy.log_std).item()
            print("batch:", update, "|", "retour", mean_retour_stats, "|", "success", success_win, "/", packet, "|", "sigma", sigma)

    return retour_stats, policy 

if __name__ == "__main__":

    list_seed=[97, 104, 171]
    list_mode = ['rtg', 'discounted']
    lr = 1e-2
    packet=8
    n_updates=63
    baseline=True
    env = SwimmerEnv(2, 0.1)

    #normalize ='episode'
    normalize ='steps'

    for seed in list_seed:
        for mode in list_mode:

            retour_stats, policy = train(env, seed, lr, mode, packet=packet, baseline=baseline, normalize=normalize, n_updates=n_updates)

            np.save(f'runs/returns_mean_periodic_baseline{baseline}_{mode}_normalize{normalize}_lr{lr}_seed{seed}_packet{packet}.npy', retour_stats)
            torch.save(policy.state_dict(), f'runs/policy_mean_periodic_baseline{baseline}_{mode}_normalize{normalize}_lr{lr}_seed{seed}_packet{packet}.pt')
            #plot_returns(retour_stats, 20, f"mean_vswim{env.v_swim}_periodic_baseline{baseline}_{mode}_normalize{normalize}_lr{lr}_seed{seed}_packet{packet}")

            retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env, policy, 20, deterministic=False)
            print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

            retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env, policy, 1, deterministic=True)
            print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 1, "|", "length", length_stats_mean_deterministic)

    retour_stats_mean_greedy, success_win_greedy, length_stats_mean_greedy = evaluate(env, GreedyPolicy(), 20, deterministic=True)
    print("GREEDY:", "|", "retour", retour_stats_mean_greedy, "|", "success", success_win_greedy, "/", 20, "|", "length", length_stats_mean_greedy)



