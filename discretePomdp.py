from pomdp import POMDP
from scipy.stats import norm
import numpy as np

class DiscretePOMDP(POMDP):
    def __init__(self, k, all_cohs, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(DiscretePOMDP, self).__init__(k, std_z, step_time, num_samples)
        self.all_cohs = all_cohs

    def calcDirectionPositiveProb(self, c, t, Z):
        direction_prob = np.zeros(Z.size)
        n_direction_prob = np.zeros(Z.size)
        for state_c in self.all_cohs:
            #w = 1.0  # this is not necessary because of the prior : w * prior
            #if state_c < 1:
                #w = .5
            mu_s = self.k * state_c / 1000.0
            direction_prob = direction_prob + self.calcStateProb(t, Z, mu_s)                
            n_direction_prob =  n_direction_prob +  self.calcStateProb(t, Z, -mu_s)
        direction_prob = (direction_prob + self.EPSILON) / (n_direction_prob + direction_prob + self.EPSILON)
        return direction_prob

    def calcStateProb(self, t, Z, mu_s):
        var_z = float(self.std_z ** 2)
        var_t = var_z / float (t)
        mu_t = (Z / float(t))        
        return norm(mu_t, var_t**.5).pdf(mu_s)
