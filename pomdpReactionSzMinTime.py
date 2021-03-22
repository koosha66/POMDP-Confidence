from scipy.stats import norm
import numpy as np
from pomdpWithCostSz import POMDPWithCostSz
from trial import Trial
import scipy.io
import glob
from dataPoint import DataPoint
from analysis import Analysis
import math

class POMDPReactionSzMinTime(POMDPWithCostSz):
    def __init__(self, k, mu_i, sz, std_i, all_cohs, cost, min_t = 0, time_max = 3000, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        self.t_min = int (min_t / step_time)
        super(POMDPReactionSzMinTime, self).__init__(k, mu_i, sz, std_i, all_cohs, cost, time_max, std_z, step_time, num_samples)
    def generateObservations(self, mu):
        cost = self.cost
        file_name = 'observations/' + str(mu)  + '_' + str(int(self.sz*100)) + '_' + str(int(self.std_i*100)) + '_' + str(int(self.std_z*100))  + '_' +  str(self.cost*1000) + '_' + str(self.t_min)   + '_' + str(self.t_max) + str(self.num_samples) + '.npy'
        try: 
            Z_t = np.load(file_name)
        except: 
            Z_t = np.zeros([self.num_samples, self.t_max])
            observe = np.ones(self.num_samples)
            num_done = 0
            for t in range(1, self.t_min):
                Z_t[:, t] = np.random.normal(mu * t, self.sz * (t **.5), self.num_samples)
            
            for t in range(self.t_min, self.t_max):
                Z_t[:, t] = observe * (Z_t[:, t - 1] + np.random.normal(mu, self.sz, self.num_samples))
                for i in range(self.num_samples):
                    if observe[i] == 0:
                        continue
                    cur_conf = norm.cdf(abs(Z_t[i,t]) * self.std_z ** -2 / ((t * self.std_z ** -2 + self.std_i ** (-2)) ** .5))
                    if self.calcExpectedConf(abs(Z_t[i,t]), t) - cur_conf < cost:
                        observe[i] = 0
                        num_done +=1
                        if num_done >= self.num_samples:
                            print (mu, "finished after ", t)
                            np.save(file_name, Z_t)
                            return Z_t                   
        np.save(file_name, Z_t)
        return Z_t 

    def generateTrials(self):   #only for confidence and accuracy, no sure option 
        trials = []
        for t in range(1, self.t_max - 1):
            num_z = 0
            for i in range(len(self.all_cohs)):
                Z_cur = self.mu_Z_t_matrix[i, :, t]
                Z_post = self.mu_Z_t_matrix[i, :, t + 1]
                stopped = np.where (Z_post == 0)[0]
                Z_cur = Z_cur[stopped]
                Z_cur = Z_cur[np.where(Z_cur != 0)[0]]
                num_z += Z_cur.size
                for z in Z_cur:
                    cor_tar = 1 # true choice = right 
                    ch_tar = 2 - int(z > 0)
                    dur = t * self.step_time
                    coh = self.all_cohs[i]
                    trials.append(Trial(coh, dur, cor_tar, ch_tar, False))
                    trials.append(Trial(coh, dur, cor_tar, ch_tar, True))
        return trials

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
        var_z = float(self.std_z ** 2)   #####
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)   
        return prob
    
    def logLike(self, c, t, is_correct = True):
        mu = self.k * c 
        est_mu_i = 0
        for i in range(1, len(self.all_cohs)):
            if abs(mu - self.all_cohs[i] * self.k / 1000.0) < abs(mu - self.all_cohs[est_mu_i] * self.k / 1000.0):
                est_mu_i = i
        #print (est_mu_i, end = " ")
        Z_cost = self.mu_Z_t_matrix[est_mu_i, :, int(t)]
        Z_post = self.mu_Z_t_matrix[est_mu_i, :, int(t + 1)]
        stopped = np.where (Z_post == 0)[0]
        Z_cost = Z_cost[stopped]
        Z_cost = Z_cost[np.where(Z_cost != 0)[0]]
        if is_correct:
            Z_cost = Z_cost[np.where(Z_cost > 0)[0]]
        else: 
            Z_cost = Z_cost[np.where(Z_cost < 0)[0]]
        if Z_cost.size == 0:
            return math.log (1 / (2 * self.num_samples))
        return math.log (Z_cost.size / float(self.num_samples))  
        
    
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
        var_z = float(self.std_z ** 2)
        var_i = float(self.std_i ** 2)
        var_t = var_z / float (t + var_z/var_i)
        prob = norm.cdf(((Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))
        n_prob = norm.cdf(((-Z_cost / var_z) + self.mu_i * var_z / var_i) * (var_t ** .5))  
        prob = (prob + self.EPSILON) / (n_prob + prob + self.EPSILON)  
        return prob
 
        
