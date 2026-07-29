import numpy as np 
from env import wrap_angle

class GreedyPolicy():

    def act(self, obs, deterministic):
        theta_wanted = np.arctan2(obs[1], obs[0])
        theta_actual = np.arctan2(obs[3], obs[2])
        delta = wrap_angle(theta_wanted - theta_actual)
        return delta 
