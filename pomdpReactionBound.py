import numpy as np
from pomdpReaction import POMDPReaction


class POMDPReactionBound(POMDPReaction):
    def __init__(self, k, mu_i, std_i, all_cohs, bound, time_max = 900, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        self.bound = bound
        super(POMDPReactionBound, self).__init__(k, mu_i, std_i, all_cohs, bound, time_max, std_z, step_time, num_samples)

    def generateObservations(self, mu):
        Z_t = np.zeros([self.num_samples, self.t_max])
        observe = np.ones(self.num_samples)
        num_done = 0
        for t in range(1, self.t_max):
            Z_t[:, t] = observe * (Z_t[:, t - 1] + np.random.normal(mu, self.std_z, self.num_samples))
            reached = np.where(Z_t[:, t] > self.bound)[0]
            observe[reached] = 0
            num_done += reached.size
            reached = np.where(Z_t[:, t] < -self.bound)[0]
            observe[reached] = 0
            num_done += reached.size
            if num_done >= self.num_samples:
                print (mu, "finished after ", t)
                return Z_t
        print (mu, "num done:" , num_done)
        return Z_t