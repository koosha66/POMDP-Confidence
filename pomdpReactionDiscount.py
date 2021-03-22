#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:59:28 2019

@author: koosha
"""

from scipy.stats import norm
import numpy as np
from pomdpReaction import POMDPReaction

class POMDPReactionDiscount(POMDPReaction):
    def __init__(self, k, mu_i, std_i, all_cohs, discount, cost = 0, time_max = 3000, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        self.discount = discount
        super(POMDPReactionDiscount, self).__init__(k, mu_i, std_i, all_cohs, cost, time_max, std_z, step_time, num_samples)
        
    def generateObservations(self, mu):
        cost = self.cost
        file_name = 'observations/' + str(mu)  + '_' + str(int(self.std_i*100)) + '_' + str(int(self.std_z*100)) + '_' + str(int(self.discount*100000)) +  '_' +  str(self.cost*1000) + '_' + str(self.t_max) + str(self.num_samples) + '.npy'
        try: 
            Z_t = np.load(file_name)
        except: 
            Z_t = np.zeros([self.num_samples, self.t_max])
            observe = np.ones(self.num_samples)
            num_done = 0
            for t in range(1, self.t_max):
                Z_t[:, t] = observe * (Z_t[:, t - 1] + np.random.normal(mu, 1, self.num_samples))
                for i in range(self.num_samples):
                    if observe[i] == 0:
                        continue
                    cur_conf = norm.cdf(abs(Z_t[i,t]) * self.std_z ** -2 / ((t * self.std_z ** -2 + self.std_i ** (-2)) ** .5))
                    if self.discount * self.calcExpectedConf(abs(Z_t[i,t]), t) < cur_conf + cost:
                        observe[i] = 0
                        num_done +=1
                        if num_done >= self.num_samples:
                            print (mu, "finished after ", t)
                            np.save(file_name, Z_t)
                            return Z_t                   
        np.save(file_name, Z_t)
        return Z_t 