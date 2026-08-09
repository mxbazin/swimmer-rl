## -------------------------------------------------------------------------------
## ------------------------------Libraries----------------------------------------
##--------------------------------------------------------------------------------

import torch
import os

from env import SwimmerEnv
from greedy import GreedyPolicy
from policy import PolicyRL
import statistics as stats
import numpy as np
from viz import plot_returns
from evaluate import evaluate
from value import Critic

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

## -------------------------------------------------------------------------------
## ------------------------------Functions----------------------------------------
##--------------------------------------------------------------------------------

def function_reward_to_go_return(list_rewards, gamma):
    n = len(list_rewards)
    rtg_sum = np.zeros_like(list_rewards, dtype=np.float64)
    for t in reversed(range(n)):
        rtg_sum[t] = list_rewards[t] + gamma*(rtg_sum[t+1] if t+1 < n else 0)
    return rtg_sum

def compute_gae(rewards, values, last_value, gamma, lambd):
    n = len(rewards)
    A = np.zeros(n)
    delta = np.zeros(n)

    for t in range(n):
        v_next = values[t+1] if t+1 < n else last_value
        delta[t] = rewards[t] + gamma*v_next - values[t]

    for t in reversed(range(n)):
        A[t] = delta[t] + gamma*lambd*(A[t+1] if t+1 < n else 0)
    return A 

def train(env, seed, log_std_init, squash, lr_loss, lr_critic, lam, gamma, packet, n_updates):

    print(r"""
        .----------------.  .----------------.  .----------------. 
    | .--------------. || .--------------. || .--------------. |
    | |   ______     | || |   ______     | || |     ____     | |
    | |  |_   __ \   | || |  |_   __ \   | || |   .'    `.   | |
    | |    | |__) |  | || |    | |__) |  | || |  /  .--.  \  | |
    | |    |  ___/   | || |    |  ___/   | || |  | |    | |  | |
    | |   _| |_      | || |   _| |_      | || |  \  `--'  /  | |
    | |  |_____|     | || |  |_____|     | || |   `.____.'   | |
    | |              | || |              | || |              | |
    | '--------------' || '--------------' || '--------------' |
    '----------------'  '----------------'  '----------------' 
    """)

    print("SEED:", seed, "|", "lr_loss:", lr_loss, "|", "lr_critic:", lr_critic, "|")
    print("Starting the script..")

    # ─── Setup: networks, optimizers, stats ───────────────────────
    torch.manual_seed(seed)

    policy = PolicyRL(log_std_init, delta_max=env.delta_max, squash=squash)
    optimizer = torch.optim.Adam(policy.parameters(),lr=lr_loss)

    critic = Critic()
    optimizer_critic = torch.optim.Adam(critic.parameters(),lr=lr_critic)

    length_stats = []
    retour_stats =[]
    flag_list=[]

    # ─── Main loop: 1 iteration = 1 update ───────────────────────
    for update in range(n_updates):
        batch_logp=[]
        batch_weights=[]
        batch_rtg=[]
        batch_obs=[]

        # ─── Collect the batch ───────────────────────
        for episode in range (packet):
            terminated = False
            truncated = False
            obs= env.reset()
            episode_logp = [] 
            episode_reward = []
            episode_obs=[]
            episode_values = []

            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float)
                episode_obs.append(obs_t)

                dist = policy(obs_t)
                value_critic = critic(obs_t).item()
                episode_values.append(value_critic)

                action = dist.sample()
                logp = dist.log_prob(action)
                episode_logp.append(logp)
                action = action.item()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward.append(reward)

            # ─── End of and episode, critic target, boostrap, advantage ───────────────────────
            return_rtg = function_reward_to_go_return(episode_reward, gamma=gamma)
            batch_rtg.extend(return_rtg)

            if terminated == True:
                last_value = 0.0
                
            elif truncated == True:
                last_value = critic(torch.as_tensor(obs, dtype=torch.float)).item()

            advantage = compute_gae(episode_reward, episode_values, last_value, gamma, lam)
            batch_weights.extend(advantage)

            batch_logp.extend(episode_logp)
            batch_obs.extend(episode_obs)

            # ─── Statistics ───────────────────────
            length = len(episode_reward) ; retour = sum(episode_reward)
            length_stats.append(length) ; retour_stats.append(retour)

            if terminated == True:
                flag_list.append(True)
                
            elif truncated == True:
                flag_list.append(False)

        # ─── Update the policy ───────────────────────
    
        batch_logp = torch.stack(batch_logp)
        batch_weights = torch.as_tensor(batch_weights, dtype=torch.float)
        batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float)
        batch_obs = torch.stack(batch_obs)  

        loss = - (batch_weights * batch_logp).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ─── Update the critic ───────────────────────
        for _ in range (0,10):
            values = critic(batch_obs)   
            loss_values = torch.square(values - batch_rtg).mean()
            optimizer_critic.zero_grad()
            loss_values.backward()
            optimizer_critic.step()

        ## ─── Logging and print the informations every 5 episodes ───────────────────────
        if update % 5 == 0:
            mean_retour_stats = stats.mean(retour_stats[-packet:])
            success_win = sum(flag_list[-packet:]) 
            sigma = torch.exp(policy.log_std).item()
            print("batch:", update, "|", "mean length", stats.mean(length_stats[-packet:]), "|", "retour", mean_retour_stats, "|", "success", success_win, "/", packet, "|", "sigma", sigma, "|")

    return retour_stats, policy 

## -------------------------------------------------------------------------------
## ------------------------------Code---------------------------------------------
##--------------------------------------------------------------------------------

if __name__ == "__main__":

    ## ─── Parameters ───────────────────────
    list_seed= [104, 171] #97
    list_mode = 'gae'
    lr_loss = 1e-3
    lr_critic = 1e-2
    packet=8
    n_updates=300
    shaping = True
    env = SwimmerEnv(0.5, 0.1, shaping=shaping)
    log_std_init = -1
    lam = 0.95
    gamma=1
    squash=True
    normalize ='steps'

    for seed in list_seed:

        retour_stats, policy = train(env, seed=seed, log_std_init=log_std_init, squash=squash, 
                                        lr_loss=lr_loss, lr_critic=lr_critic, lam=lam, gamma=gamma, packet=packet, n_updates=n_updates)

        np.save(f'runs/returns_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}.npy', retour_stats)
        torch.save(policy.state_dict(), f'runs/policy_mean_periodic_nupdates{n_updates}_log_std_init{log_std_init}_lrcritic{lr_critic}_shaping{shaping}_squash{squash}_lam{lam}_gamma{gamma}_normalize{normalize}_lrloss{lr_loss}_seed{seed}_packet{packet}.pt')
        #plot_returns(retour_stats, 20, f"mean_vswim{env.v_swim}_periodic_baseline{baseline}_{mode}_normalize{normalize}_lr{lr}_seed{seed}_packet{packet}")

        retour_stats_mean_stoch, success_win_stoch, length_stats_mean_stoch = evaluate(env, policy, 20, deterministic=False)
        print("STOCHASTIC:", "|", "retour", retour_stats_mean_stoch, "|", "success", success_win_stoch, "/", 20, "|", "length", length_stats_mean_stoch)

        retour_stats_mean_deterministic, success_win_deterministic, length_stats_mean_deterministic = evaluate(env, policy, 1, deterministic=True)
        print("DETERMINISTIC:", "|", "retour", retour_stats_mean_deterministic, "|", "success", success_win_deterministic, "/", 1, "|", "length", length_stats_mean_deterministic)

    retour_stats_mean_greedy, success_win_greedy, length_stats_mean_greedy = evaluate(env, GreedyPolicy(), 1, deterministic=True)
    print("GREEDY:", "|", "retour", retour_stats_mean_greedy, "|", "success", success_win_greedy, "/", 20, "|", "length", length_stats_mean_greedy)
