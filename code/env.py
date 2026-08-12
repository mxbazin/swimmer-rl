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

def wrap_displacement(d): 
    d = (np.mod(d + np.pi, 2*np.pi) - np.pi )
    return d

class SwimmerEnv: 
    def __init__(self, v_swim=0.75, dt=0.01, shaping=False):
        self.v_swim = v_swim 
        self.dt = dt
        self.target_x = 5
        self.target_y = 6
        self.threshold = 0.2
        self.max_steps = 1000
        self.domain_min = 0
        self.domain_max = 2*np.pi
        self.delta_max = 1 
        self.shaping = shaping

    def _displacement_to_target(self):
        dx = self.target_x - self.px
        dy = self.target_y - self.py
        return wrap_displacement(np.array([dx, dy]))

    def _get_obs(self):
        d = self._displacement_to_target()
        u, v= taylor_green_field(self.px,self.py)
        obs = np.array([d[0], d[1], np.cos(self.theta), np.sin(self.theta), u, v])
        return obs

    def reset(self, px=None, py=None):
        self.px = 2
        self.py = 1
        d = self._displacement_to_target()

        self.distance_target = np.sqrt(d[0]**2 + d[1]**2)
        self.old_distance = self.distance_target

        self.theta = np.arctan2(d[1], d[0])
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
        self.px = np.mod(self.px, 2*np.pi)
        self.py = np.mod(self.py, 2*np.pi)

        self.step_count +=1
        d = self._displacement_to_target()
        self.distance_target = np.sqrt(d[0]**2 + d[1]**2)
        self.terminated = self.distance_target <= self.threshold
        self.info={}

        if self.shaping:
            self.reward= -self.dt + (self.old_distance - self.distance_target)
        else: 
            self.reward= -self.dt

        if self.terminated: 
            self.reward +=100
        self.truncated = self.step_count >= self.max_steps   

        self.old_distance = self.distance_target

        return self._get_obs(), self.reward, self.terminated, self.truncated, self.info 

if __name__ == "__main__":
    print("Starting the script..")
    terminated = False
    truncated = False
    env = SwimmerEnv(2, 0.1, shaping=True)    
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
         
    print("step_count:", env.step_count, "reward", reward)

    if terminated == True: 
        print("Target reached!")

    if truncated == True: 
        print("Out of step_counts...")

