# taylor_green_field, euler_method, SwimmerEnv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def taylor_green_field(x,y):
    u = np.sin(x)*np.cos(y)
    v = -np.cos(x)*np.sin(y)
    return u, v

def RungeKutta2_method(p, velocity, dt):
    k1 = dt*velocity(p)
    k2 = dt*velocity(p + k1/2)
    p = p + k2 
    return p

def wrap_angle(angle): 
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return angle 

class SwimmerEnv: 
    def __init__(self, v_swim=0.75, dt=0.01):
        self.v_swim = v_swim 
        self.dt = dt
        self.target_x = 5
        self.target_y = 6
        self.threshold = 0.2
        self.max_steps = 2000
        self.domain_min = 0
        self.domain_max = 2*np.pi
        self.delta_max = 0.5 

    def _get_obs(self):
        obs = np.array([self.target_x - self.px, self.target_y - self.py, np.cos(self.theta), np.sin(self.theta)])
        return obs

    def reset(self):
        self.px = 2
        self.py = 1
        dy = self.target_y - self.py 
        dx = self.target_x - self.px 
        self.theta = np.arctan2(dy, dx)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):

        action = np.clip(action, -self.delta_max, self.delta_max)
        self.theta = wrap_angle(self.theta + action)

        def velocity(p):
            u, v= taylor_green_field(p[0],p[1])
            u = u + self.v_swim*np.cos(self.theta)
            v = v + self.v_swim*np.sin(self.theta)
            return np.array([u,v])
        
        p = np.array([self.px, self.py])
        p = RungeKutta2_method(p, velocity, self.dt)

        self.px = p[0]; self.py = p[1]
        self.px = np.clip(self.px, 0, 2*np.pi)
        self.py = np.clip(self.py, 0, 2*np.pi)

        self.step_count +=1
        self.distance_target = np.sqrt((self.px - self.target_x)**2 
                                       + (self.py - self.target_y)**2)
        
        self.terminated = self.distance_target <= self.threshold
        self.truncated = self.step_count >= self.max_steps 
        self.info={}
        self.reward= -self.dt

        if self.terminated: 
            self.reward +=100

        return self._get_obs(), self.reward, self.terminated, self.truncated, self.info 

if __name__ == "__main__":
    print("Starting the script..")
    terminated = False
    truncated = False
    env = SwimmerEnv(0.3, 0.1)
    obs= env.reset()

    list_pos = []

    X = np.linspace(env.domain_min, env.domain_max, 50)
    Y = np.linspace(env.domain_min, env.domain_max, 50)
    x, y = np.meshgrid(X, Y, indexing='xy')
    U, V = taylor_green_field(x, y)

    while not (terminated or truncated):
        theta_wanted = np.arctan2(obs[1], obs[0])
        theta_actual = np.arctan2(obs[3], obs[2])
        delta = theta_wanted - theta_actual
        theta = wrap_angle(delta)
        obs, reward, terminated, truncated, info = env.step(theta)
        list_pos.append((env.px, env.py))
         
    print("step_count:", env.step_count)

    if terminated == True: 
        print("Target reached!")

    if truncated == True: 
        print("Out of step_counts...")

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
