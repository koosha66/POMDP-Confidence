from scipy.stats import norm
import numpy as np
from pomdpReaction import POMDPReaction


class POMDPReactionIncCost(POMDPReaction):
    def __init__(self, k, mu_i, std_i, all_cohs, cost, gamma = 1, time_max = 3000, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        self.gamma = gamma
        super(POMDPReactionIncCost, self).__init__(k, mu_i, std_i, all_cohs, cost, time_max, std_z, step_time, num_samples)

    def generateObservations(self, mu):
        cost = self.cost
        file_name = 'observations/' + str(mu)  + '_' + str(self.cost*1000) + '_' + str(self.gamma*100) + '_' + str(self.t_max) + str(self.num_samples) + '.npy'
        try: 
            Z_t = np.load(file_name)
        except: 
            Z_t = np.zeros([self.num_samples, self.t_max])
            observe = np.ones(self.num_samples)
            num_done = 0
            for t in range(1, self.t_max):
                cost *= self.gamma
                Z_t[:, t] = observe * (Z_t[:, t - 1] + np.random.normal(mu, self.std_z, self.num_samples))
                for i in range(self.num_samples):
                    if observe[i] == 0:
                        continue
                    cur_conf = norm.cdf(abs(Z_t[i,t]) / ((t + self.std_i ** (-2)) ** .5))
                    if self.calcExpectedConf(abs(Z_t[i,t]), t) - cur_conf < cost:
                        observe[i] = 0
                        num_done +=1
                        if num_done >= self.num_samples:
                            print (mu, "finished after ", t)
                            np.save(file_name, Z_t)
                            return Z_t                   
        np.save(file_name, Z_t)
        return Z_t 

  