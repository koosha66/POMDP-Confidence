from scipy.stats import norm, mvn
import numpy as np
from continuousPomdp import ContinuousPOMDP


class POMDPWithBound(ContinuousPOMDP):
    def __init__(self, k, mu_i, std_i, all_cohs, bound, time_max = 900, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(POMDPWithBound, self).__init__(k, mu_i, std_i, std_z, step_time, num_samples)
        self.all_cohs = all_cohs
        self.bound = bound
        self.t_max = int (time_max / step_time) + 1
        self.mu_Z_t_matrix = np.zeros([len(all_cohs), num_samples, self.t_max])
        for i in range(len(all_cohs)):
            self.mu_Z_t_matrix[i,:,:] = self.generateObservations(self.k * all_cohs[i]/ 1000.0)

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
        mu = self.k * c 
        est_mu_i = 0
        for i in range(1, len(self.all_cohs)):
            if abs(mu - self.all_cohs[i] * self.k / 1000.0) < abs(mu - self.all_cohs[est_mu_i] * self.k / 1000.0):
                est_mu_i = i
        Z_cost = self.mu_Z_t_matrix[est_mu_i, :, int(t)]
        t_array = t * np.ones(Z_cost.size)
        for i in range(Z_cost.size):
            j = t - 1
            while (self.mu_Z_t_matrix[est_mu_i, i, int(t)] == self.mu_Z_t_matrix[est_mu_i, i, int(j)]):
                j -= 1
            t_array[i] = j + 1      
        #if np.mean(Z) < 0: #left side
        #    Z_cost = -Z_cost 
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / (t_array + var_z/var_i)
        prob = norm.cdf((Z_cost + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf((-Z_cost + self.mu_i * var_z / var_i) * (var_t ** .5))
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)
        return prob
