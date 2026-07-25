import numpy as np
import matplotlib.pyplot as plt

def taylor_green_field(x,y):
    u = np.sin(x)*np.cos(y)
    v = -np.cos(x)*np.sin(y)
    return u, v

def euler_method(px, py, u, v, dt):
    px_updated = px + dt*u 
    py_updated = py + dt*v
    return px_updated, py_updated

class SwimmerEnv: 
    def __init__(self):
        self.v_swim = 0.3
        self.dt = 0.1
        self.target_x = 5
        self.target_y = 6
        self.threshold = 0.2
        self.max_steps = 1000
        self.domain_min = 0
        self.domain_max = 2*np.pi

    def _get_obs(self):
        self.obs = np.array([self.target_x - self.px, self.target_y - self.py])
        return self.obs

    def reset(self):
        self.px = 2
        self.py = 1
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        self.u, self.v = taylor_green_field(self.px, self.py)
        self.u = self.u + self.v_swim*np.cos(action)
        self.v = self.v + self.v_swim*np.sin(action)

        self.px, self.py = euler_method(self.px, self.py, self.u, self.v, self.dt)
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
    env = SwimmerEnv()
    obs= env.reset()

    list_pos = []

    X = np.linspace(env.domain_min, env.domain_max, 50)
    Y = np.linspace(env.domain_min, env.domain_max, 50)
    x, y = np.meshgrid(X, Y, indexing='xy')
    U, V = taylor_green_field(x, y)

    while not (terminated or truncated):
        theta = np.arctan2(obs[1], obs[0])
        obs, reward, terminated, truncated, info = env.step(theta)
        list_pos.append((env.px, env.py))
         
    print("step_count:", env.step_count)

    if terminated == True: 
        print("Target reached!")

    if truncated == True: 
        print("Out of step_counts...")

    #Plot TG field + trajectory of the swimmer
    traj = np.array(list_pos)
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)
    ax.quiverkey(q,X=1, Y=1, U=5, label ='Quiver, length =5', labelpos='E')
    plt.plot(traj[:,0], traj[:,1], 'r--')
    plt.show()
