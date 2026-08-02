# rollout(env, policy, n_episodes, deterministic) -> retours, 

from env import SwimmerEnv
import statistics as stats
from greedy import GreedyPolicy 
from policy import PolicyRL
import numpy as np
import torch

def evaluate(env, policy, n_episodes=20, deterministic=True):
    length_stats = []
    retour_stats =[]
    flag_deterministic_list=[]

    for episode in range (n_episodes):
            terminated = False
            truncated = False
            obs= env.reset()
            list_reward = []
            while not (terminated or truncated):
                action = policy.act(obs, deterministic)                
                obs, reward, terminated, truncated, info = env.step(action)
                list_reward.append(reward)
                #print("action:", action)
                #print("px", env.px, "py", env.py)
    
            length = len(list_reward) ; retour = sum(list_reward)
            length_stats.append(length) ; retour_stats.append(retour)

            if terminated == True:
                flag_deterministic_list.append(True)
            elif truncated == True:
                flag_deterministic_list.append(False)

    retour_stats_mean = stats.mean(retour_stats[-n_episodes:])
    length_stats_mean = stats.mean(length_stats[-n_episodes:])
    success_win = sum(flag_deterministic_list[-n_episodes:]) 

    return retour_stats_mean, success_win, length_stats_mean

if __name__ == "__main__":
    # env = SwimmerEnv(0.5, 0.1, shaping=True)
    # policy = PolicyRL()
    # policy.load_state_dict(torch.load('runs/policy_naif.pt'))
    # policy = GreedyPolicy()
    # obs = env.reset(2, 1)
    # terminated = False
    # truncated = False
    # list_reward_test1 = []
    # while not (terminated or truncated):
    #     action = policy.act(obs, deterministic=True)                
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     list_reward_test1.append(reward)
    # print("env.step_count:", env.step_count, "list_reward.sum", sum(list_reward_test1))

    # obs = env.reset(2, 1 + 2*np.pi)
    # terminated = False
    # truncated = False
    # list_reward_test2 = []
    # while not (terminated or truncated):
    #     action = policy.act(obs, deterministic=True)                
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     list_reward_test2.append(reward)
    # print("env.step_count:", env.step_count, "list_reward.sum", sum(list_reward_test2))

    env = SwimmerEnv(0.5, 0.1, shaping=True)
    policy = GreedyPolicy()
    n_episodes=1
    retour_stats_mean, success_win, length_stats_mean = evaluate(env, policy, n_episodes, deterministic=True)
    print("ep:", n_episodes, "|", "retour", retour_stats_mean, "|", "success", success_win, "/", n_episodes, "|", "length_stats_mean", length_stats_mean)
