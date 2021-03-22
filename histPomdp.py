from __future__ import division, print_function
from pomdp import POMDP
from scipy.stats import norm
import numpy as np

class HistPOMDP(POMDP):
    def __init__(self, k, trials, num_bins, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(HistPOMDP, self).__init__(k, std_z, step_time, num_samples)
        self.num_bins = num_bins
        self.bins = -k  + (2 * k  / num_bins) * np.array(range(0, num_bins+1))
        obs = np.zeros(len(trials))
        i = 0
        for trial in trials:
            t = trial.getDuration() / float(self.step_time)
            t = 1
            sigma = self.std_z/ float(t ** .5)
            c = trial.getCoherence()
            mu = k * c / 1000.0
            if trial.getCorrectTarget() > 1: #left
                mu = -mu
            obs[i] = np.random.normal(mu, sigma)
            i += 1
        self.prior, self.bins = np.histogram(obs, bins = self.bins, normed = True)
        self.prior = self.prior / (2 * k  / num_bins)
        self.points = self.bins[:-1] + np.diff(self.bins)
    def calcDirectionPositiveProb(self, c, t, Z):
        var_z = float(self.std_z ** 2)
        var_t = var_z / float (t)
        prob = np.zeros(Z.size)
        n_prob = np.zeros(Z.size)
        for i in range(int(self.prior.size / 2)):
            n_prob += self.prior[i] * norm.pdf((self.points[i] - (Z/t)) / (var_t ** .5))
            
        for i in range(int(self.prior.size / 2), self.prior.size):    
            prob += self.prior[i] * norm.pdf((self.points[i] - (Z/t)) / (var_t ** .5)) 
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)
        return prob