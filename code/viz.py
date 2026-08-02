 # plot_traj(), make_gif() — importés par les autres

import matplotlib.pyplot as plt 
import statistics as stats 
import torch
import numpy as np
from env import SwimmerEnv, taylor_green_field
from policy import PolicyRL
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plot_returns(returns, window, title):
   mean_returns=[]
   x_abscisses=[]
   for i in range(window, len(returns)):
      x_abscisses.append(i)
      mean_returns.append(stats.mean(returns[i-window:i]))
   plt.plot(returns, '.', color='red', alpha=0.3, label='Return per episode')
   plt.plot(x_abscisses, mean_returns, '-', color='blue', label=f'Mean return ({window} episodes)')
   plt.xlabel("Episode")
   plt.ylabel("Return")
   plt.title(f"REINFORCE {title}")
   plt.legend()
   plt.savefig(f'figs/mean_return_{title}.png')
   #plt.show()
   plt.close()

def plot_comparison(pattern, title, window=20, list_label=None, greedy_ref=None):
   list_paths= sorted(glob.glob(pattern))
   if list_label is None:
         list_label=[]
         for label in list_paths:
            path = os.path.basename(label) 
            path = path.replace('returns_mean_periodic_', '').replace('.npy', '')
            list_label.append(path)

   for element, label in zip(list_paths, list_label):      
      data= np.load(element)
      mean_returns=[]
      x_abscisses=[]      
      for i in range(window, len(data)):
         x_abscisses.append(i)
         mean_returns.append(stats.mean(data[i-window:i]))
      plt.plot(x_abscisses, mean_returns, '-', label=label)
   if greedy_ref is not None:
      plt.axhline(greedy_ref, linestyle='--', color='grey', label='greedy')
   plt.xlabel("Episode")
   plt.ylabel("Return")
   plt.title(f"REINFORCE {title}")
   plt.legend()
   plt.savefig(f'figs/mean_return_{title}.png')
   plt.show()


def plot_trajectory(list_pos, env, title, ax=None):
    
    #Plot TG field + trajectory of the swimmer
   if ax is None: 
      fig, ax = plt.subplots()
      standalone=True
   else: 
      standalone=False

   X = np.linspace(env.domain_min, env.domain_max, 50)
   Y = np.linspace(env.domain_min, env.domain_max, 50)
   x, y = np.meshgrid(X, Y, indexing='xy')
   U, V = taylor_green_field(x, y)

   traj = np.array(list_pos)
   q = ax.quiver(X, Y, U, V)
   ax.scatter(traj[0,0], traj[0,1], c='b', marker='o', s=100, label='start')
   ax.scatter(env.target_x, env.target_y, c='g', marker='*', s=200, label='target')
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_aspect('equal')
   ax.plot(traj[:,0], traj[:,1], 'r.',label='trajectory')
   ax.set_title(f'{title} — {len(list_pos)} pas')
   if standalone:
      ax.legend()
      ax.quiverkey(q, X=1, Y=1, U=5, label =f'Quiver, length =5', labelpos='E')
      fig.savefig(f'figs/traj_vswim_{title}.png')
      plt.show()
      plt.close()

if __name__ == "__main__":
   env = SwimmerEnv(0.5, 0.1, shaping=True)
   log_std_init = -1 ; squash=True
   policy = PolicyRL(log_std_init, delta_max=env.delta_max, squash=squash)
   paths = sorted(glob.glob('runs/reinforce/policy_*.pt'))
   fig, axes = plt.subplots(2, 3, figsize=(15,10))
   axes = axes.flatten()
   list_pos=[]
   
   for path, ax in zip(paths, axes):
         pts = torch.load(path)
         policy.load_state_dict(pts)
         obs = env.reset()
         terminated = truncated = False
         list_pos = []
         while not (terminated or truncated):
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            list_pos.append((env.px, env.py))  
         plot_trajectory(list_pos, env, title=os.path.basename(path).replace('policy_mean_periodic_', '').replace('.pt', '').replace('nupdates300_log_std_init-1_use_criticTrue_lrcritic0.01_shapingTrue_squashTrue_baselineTrue_', ''), ax=ax)
   fig.tight_layout()
   plt.savefig('figs/traj_vswim_traj_6runs.png')
   plt.show()

   # list_label = ["disc_104", "disc_171", "disc_97", "rtg_104", "rtg_171", "rtg_97"]
   # plot_comparison('runs/*nupdates300*.npy', 'vswim05_6runs', window=100, list_label=list_label, greedy_ref=-98.5)

