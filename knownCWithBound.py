from scipy.stats import norm, mvn
import numpy as np
from knownCPomdp import KnownCPOMDP


class KnownCWithBound(KnownCPOMDP):
    def __init__(self, k, c, bound, time_max = 900, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(KnownCWithBound, self).__init__(k, std_z, step_time, num_samples)
        self.coh = c
        self.bound = bound
        self.t_max = int (time_max / step_time) + 1
        self.mu_Z_t_matrix = np.zeros([num_samples, self.t_max])
        self.mu_Z_t_matrix[:,:] = self.generateObservations(self.k * c/ 1000.0)

    def generateObservations(self, mu):
        Z_t = np.zeros([self.num_samples, self.t_max])
        observe = np.ones(self.num_samples)
        for t in range(1, self.t_max):
            Z_t[:, t] = Z_t[:, t - 1] + observe * np.random.normal(mu, self.std_z, self.num_samples)
            reached = np.where(Z_t[:, t] > self.bound)[0]
            observe[reached] = 0
            num_done = reached.size
            reached = np.where(Z_t[:, t] < -self.bound)[0]
            observe[reached] = 0
            num_done += reached.size
            if num_done >= self.num_samples:
                print (mu, "finished after ", t)
                for k in range(t + 1, self.t_max):
                    Z_t[:, k] = Z_t[:, t]
                return Z_t
        print (mu, "num done:" , num_done)
        return Z_t

    def calcDirectionPositiveProb(self, c, t, Z):
        Z_cost = self.mu_Z_t_matrix[:, int(t)]
        t_array = t * np.ones(Z_cost.size)
        for i in range(Z_cost.size):
            j = t - 1
            while (self.mu_Z_t_matrix[i, int(t)] == self.mu_Z_t_matrix[i, int(j)]):
                j -= 1
            t_array[i] = j + 1  
        direction_prob = np.zeros(Z_cost.size)
        n_direction_prob = np.zeros(Z_cost.size)
        mu_s = self.k * c
        direction_prob = self.calcStateProb(t_array, Z_cost, mu_s)            
        n_direction_prob = self.calcStateProb(t_array, Z_cost, -mu_s)
        direction_prob = (direction_prob + self.EPSILON) / (n_direction_prob + direction_prob + self.EPSILON)
        return direction_prob
    
