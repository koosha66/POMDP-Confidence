from pomdp import POMDP
from scipy.stats import norm
import numpy as np

class KnownCPOMDP(POMDP):
    def __init__(self, k, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(KnownCPOMDP, self).__init__(k, std_z, step_time, num_samples)

    def calcDirectionPositiveProb(self, c, t, Z): 
        direction_prob = np.zeros(Z.size)
        n_direction_prob = np.zeros(Z.size)
        mu_s = self.k * c
        direction_prob = self.calcStateProb(t, Z, mu_s)            
        n_direction_prob = self.calcStateProb(t, Z, -mu_s)
        direction_prob = (direction_prob + self.EPSILON) / (n_direction_prob + direction_prob + self.EPSILON)
        return direction_prob

    def calcStateProb(self, t, Z, mu_s):
        var_z = float(self.std_z ** 2)
        var_t = var_z / t
        mu_t = Z / t       
        return norm(mu_t, var_t**.5).pdf(mu_s)
