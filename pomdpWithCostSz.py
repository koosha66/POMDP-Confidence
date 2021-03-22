from scipy.stats import norm, mvn
import numpy as np
from continuousPomdp import ContinuousPOMDP


class POMDPWithCostSz(ContinuousPOMDP):
    def __init__(self, k, mu_i, sz, std_i, all_cohs, cost, time_max = 900, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(POMDPWithCostSz, self).__init__(k, mu_i, std_i, std_z, step_time, num_samples)
        self.all_cohs = all_cohs
        self.sz = sz
        self.cost = cost
        self.t_max = int (time_max / step_time) + 1
        self.mu_Z_t_matrix = np.zeros([len(all_cohs), num_samples, self.t_max])
        for i in range(len(all_cohs)):
            self.mu_Z_t_matrix[i,:,:] = self.generateObservations(self.k * all_cohs[i]/ 1000.0)

    def calcExpectedConf(self, Z, t):
        # same direction
        mu_z = (Z * self.std_z ** -2) / (t * self.std_z ** -2 + self.std_i ** (-2))
        mu_2 = mu_z
        var_2 = self.std_z ** 2 + (1 / (t * self.std_z ** -2 + self.std_i ** (-2)))
        var_1 =  (t + 1) * self.std_z ** 2 +  self.std_i ** (-2) * self.std_z ** 4 + var_2
        mu_1 =  Z + mu_z
        cor = (var_2 / var_1) ** .5
        low = np.array([-100000, -100000])
        upp = np.array([(mu_1) / ((var_1)**.5), (Z + mu_2)/(var_2**.5)])
        mu_2d = np.zeros(2)
        cov = np.array([[1, cor], [cor, 1]])
        p, i = mvn.mvnun(low, upp, mu_2d, cov)
        # opposite direction
        p2, i = mvn.mvnun(low, -upp, -mu_2d, cov)
        return  p + p2

    def generateObservations(self, mu):
        file_name = 'observations/fixed_' + str(mu) + '_' + str(self.cost*1000) +  '_' + str(self.std_z) +  '_' + str(self.std_i) +  '_' + str(self.sz) + '_' + str(self.num_samples) + '.npy'
        try: 
            Z_t = np.load(file_name)
        except: 
            Z_t = np.zeros([self.num_samples, self.t_max])
            observe = np.ones(self.num_samples)
            num_stopped = 0 
            for t in range(1, self.t_max):
                Z_t[:, t] = Z_t[:, t - 1] + observe * np.random.normal(mu, self.sz, self.num_samples)
                for i in range(self.num_samples):
                    if observe[i] == 0:
                        continue
                    cur_conf = norm.cdf(abs(Z_t[i,t])* self.std_z ** -2 / ((t * self.std_z ** -2 + self.std_i ** (-2)) ** .5))
                    if self.calcExpectedConf(abs(Z_t[i,t]), t) + 5 * 10 **-16 - cur_conf < self.cost:
                        observe[i] = 0
                num_stopped +=1 
                if num_stopped == self.num_samples:
                    for k in range(t + 1, self.t_max):
                        Z_t[:, k] = Z_t[:, t]
                    np.save(file_name, Z_t)
                    return Z_t
            np.save(file_name, Z_t)
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
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / (t_array + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON) 
        return prob
