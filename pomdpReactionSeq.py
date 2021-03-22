import numpy as np
from pomdpReaction import POMDPReaction
from scipy.stats import norm, mvn


class POMDPReactionSeq(POMDPReaction):
    def __init__(self, k, mu_i, std_i, all_cohs, cost, time_max = 3000, time_delay = 200, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(POMDPReactionSeq, self).__init__(k, mu_i, std_i, all_cohs, cost, time_max, std_z, step_time, num_samples)
        self.t_delay = int (time_delay / step_time)

            
    def calcDirectionPositiveProb(self, c, t, Z):
        mu = self.k * c 
        est_mu_i = 0
        for i in range(1, len(self.all_cohs)):
            if abs(mu - self.all_cohs[i] * self.k / 1000.0) < abs(mu - self.all_cohs[est_mu_i] * self.k / 1000.0):
                est_mu_i = i
        Z_cost = self.mu_Z_t_matrix[est_mu_i, :, int(t)]
        Z_post = self.mu_Z_t_matrix[est_mu_i, :, int(t + 1)]
        stopped = np.where (Z_post == 0)[0]
        Z_cost = Z_cost[stopped]
        Z_cost = Z_cost[np.where(Z_cost != 0)[0]]
        if Z_cost.size == 0:
            return Z_cost
        Z_cost2 = Z_cost + np.random.normal(mu * self.t_delay, 1.00 * (self.t_delay ** .5), Z_cost.size)
        Z_cost = (np.sign(Z_cost) == np.sign(Z_cost2)) * Z_cost2 + 0.0001 * np.sign(Z_cost)
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)  
        return prob

    def calcDirectionPositiveProbCorrect(self, c, t, Z):
        mu = self.k * c 
        est_mu_i = 0
        for i in range(1, len(self.all_cohs)):
            if abs(mu - self.all_cohs[i] * self.k / 1000.0) < abs(mu - self.all_cohs[est_mu_i] * self.k / 1000.0):
                est_mu_i = i
        Z_cost = self.mu_Z_t_matrix[est_mu_i, :, int(t)]
        Z_post = self.mu_Z_t_matrix[est_mu_i, :, int(t + 1)]
        stopped = np.where (Z_post == 0)[0]
        Z_cost = Z_cost[stopped]
        Z_cost = Z_cost[np.where(Z_cost > 0)[0]]
        if Z_cost.size == 0:
            return Z_cost
        Z_cost = Z_cost + np.random.normal(mu * self.t_delay, 1.00 * (self.t_delay ** .5), Z_cost.size)
        #Z_cost = (np.sign(Z_cost) == np.sign(Z_cost2)) * Z_cost2 + 0.0001 * np.sign(Z_cost)
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)    
        return prob
    
    
    
    def calcDirectionPositiveProbIncorrect(self, c, t, Z):
        mu = self.k * c 
        est_mu_i = 0
        for i in range(1, len(self.all_cohs)):
            if abs(mu - self.all_cohs[i] * self.k / 1000.0) < abs(mu - self.all_cohs[est_mu_i] * self.k / 1000.0):
                est_mu_i = i
        Z_cost = self.mu_Z_t_matrix[est_mu_i, :, int(t)]
        Z_post = self.mu_Z_t_matrix[est_mu_i, :, int(t + 1)]
        stopped = np.where (Z_post == 0)[0]
        Z_cost = Z_cost[stopped]
        Z_cost = Z_cost[np.where(Z_cost < 0)[0]]
        if Z_cost.size == 0:
            return Z_cost
        Z_cost = Z_cost + np.random.normal(mu * self.t_delay, 1.00 * (self.t_delay ** .5), Z_cost.size)
        #Z_cost = (np.sign(Z_cost) == np.sign(Z_cost2)) * Z_cost2 + 0.0001 * np.sign(Z_cost)
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + self.t_delay + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)  
        return prob