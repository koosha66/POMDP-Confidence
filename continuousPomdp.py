from pomdp import POMDP
from scipy.stats import norm

class ContinuousPOMDP(POMDP):
    def __init__(self, k, mu_i, std_i, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(ContinuousPOMDP, self).__init__(k, std_z, step_time, num_samples)
        self.std_i = std_i
        self.mu_i = mu_i

    def calcDirectionPositiveProb(self, c, t, Z):
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + var_z/var_i)
        prob = norm.cdf(((Z / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)
        return prob


