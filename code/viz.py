 # plot_traj(), make_gif() — importés par les autres

import matplotlib.pyplot as plt 
import statistics as stats 
import random
import numpy as np
import glob
import os

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

def plot_comparison(pattern, title, window=20, list_label=None):
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
   plt.axhline(97.8, linestyle='--', color='grey', label='greedy')
   plt.xlabel("Episode")
   plt.ylabel("Return")
   plt.title(f"REINFORCE {title}")
   plt.legend()
   plt.savefig(f'figs/mean_return_{title}.png')
   plt.show()


# if __name__ == "__main__":
   # plot_comparison('runs/*seed97*.npy', 'rtg_seed97',window=20, list_label=None)
   # plot_comparison('runs/*seed104*.npy', 'rtg_seed104',window=20, list_label=None)
   # plot_comparison('runs/*seed171*.npy', 'rtg_seed171',window=20, list_label=None)

   # plot_comparison('runs/*rtg*lr0.001*.npy', 'rtg_lr1e-3_3seeds' ,window=20, list_label=None)
   # plot_comparison('runs/*discounted*lr0.001*.npy', 'discounted_lr1e-3_3seeds',window=20, list_label=None)
   # plot_comparison('runs/*seed171*.npy','seed171_toutes_variantes',window=20, list_label=None)


    #Plot TG field + trajectory of the swimmer
    # traj = np.array(list_pos)
    # fig, ax = plt.subplots()
    # q = ax.quiver(X, Y, U, V)
    # ax.quiverkey(q, X=1, Y=1, U=5, label =f'Quiver, length =5', labelpos='E')
    # ax.scatter(traj[0,0], traj[0,1], c='b', marker='o', s=100, label='start')
    # ax.scatter(env.target_x, env.target_y, c='g', marker='*', s=200, label='target')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_aspect('equal')
    # ax.plot(traj[:,0], traj[:,1], 'r--',label='trajectory')
    # ax.set_title(f'v_swim= {env.v_swim}')
    # ax.legend()
    # fig.savefig(f'figs/traj_vswim_{env.v_swim}.png')
    # plt.show()
    # plt.close()

    # #Animation
    # fig2, ax2 = plt.subplots()
    # q2 = ax2.quiver(X, Y, U, V)
    # ax2.quiverkey(q2, X=1, Y=1, U=5, label =f'Quiver, length =5', labelpos='E')
    # ax2.scatter(traj[0,0], traj[0,1], c='b', marker='o', s=100, label='start')
    # ax2.scatter(env.target_x, env.target_y, c='g', marker='*', s=200, label='target')

    # point, = ax2.plot([], [], 'ro')   # nageur, vide au départ
    # trace, = ax2.plot([], [], 'r--')   # trace, vide au départ

    # def update(frame):
    #     point.set_data([traj[frame,0]], [traj[frame,1]])        
    #     trace.set_data(traj[:frame, 0], traj[:frame, 1])        
    #     return point, trace
    
    # anim = FuncAnimation(fig2, update, frames=range(0, len(traj), 5), interval=30)
    # anim.save(f'figs/traj_vswim_{env.v_swim}.gif', writer='pillow')
    # plt.show()