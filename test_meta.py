#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:24:20 2018

@author: koosha
"""
import numpy as np
from scipy.stats import norm
from pomdpWithCostSz import POMDPWithCostSz
from continuousPomdp import ContinuousPOMDP
#import matplotlib.pyplot as plt

def calcDirectionPositiveProb(mu_s, std_zs, t, Z):
    var_z = float(std_zs ** 2)
    var_t = var_z * t     
    return norm(mu_s* t, var_t**.5).pdf(Z)/( norm(mu_s*t, var_t**.5).pdf(Z) + norm(-mu_s * t, var_t**.5).pdf(Z))

EPSILON = 10 ** -40
def calcDirectionPositiveProbPOMDP(std_zs, t, Z):
    std_i = .32
    var_z = std_zs ** 2
    var_i = std_i ** 2
    var_t = var_z / (t + var_z/var_i)
    prob = norm.cdf((Z / var_z)  * (var_t ** .5))
    n_prob = norm.cdf((-Z / var_z) * (var_t ** .5))  
    prob = (prob + EPSILON) / (n_prob + prob + EPSILON)
    return prob  



num_samples = 100000
k = 1.0
c = 128 / 1000.0
mu = k * c
t = 40
std_z = .9
std_zs = 1.6
th = .70
cpc = POMDPWithCostSz(k, 0, std_z, .46, [128], 10 **-4, 420, std_zs, 10, num_samples) 


Z = np.random.normal(mu * t, std_z * (t ** .5), num_samples)
direction_prob = cpc.calcDirectionPositiveProb(c, t, Z)

correct_trials = np.where(direction_prob>.5)[0]
incorrect_trials = np.where(direction_prob<.5)[0]

HR = np.where(direction_prob > th)[0].size/correct_trials.size
FR = np.where(direction_prob < 1 - th)[0].size/incorrect_trials.size
d = norm.ppf(correct_trials.size/num_samples)- norm.ppf(incorrect_trials.size/num_samples)
crit = -.5 * (norm.ppf(correct_trials.size/num_samples) + norm.ppf(incorrect_trials.size/num_samples))

print ("avg confidence=", (np.sum(direction_prob[correct_trials]) + np.sum(1-direction_prob[incorrect_trials])) / num_samples)
print ("accuracy = ", correct_trials.size/num_samples)
print ("P(high) =" , (np.where(direction_prob > th)[0].size + np.where(direction_prob < 1 - th)[0].size) / num_samples)
print ("d = ", d, "criteria = ", crit) 
print ("HR = ", [HR])
print ("FR = ", [FR])
#metad = norm.ppf(HR) - norm.ppf(FR)
#print (metad)
#print (metad/d)

#### check change of choice and rating:
num_ch_changed = 0
num_rate_changed = 0
Z_t = cpc.mu_Z_t_matrix[0,:,:]
stopped = np.zeros(num_samples)
for ti in range(1, t):
    stopped_index = np.where(Z_t[:, ti] == Z_t[:, ti - 1])[0]
    if stopped_index.size == 0:
        continue
    stopped_now = stopped_index[np.where(stopped[stopped_index] == 0)[0]]
    stopped[stopped_now] = 1
    cur_z = Z_t[stopped_now, t]
    all_z = cur_z + np.random.normal(mu * (t - ti), std_z * ((t - ti) ** .5), cur_z.size)
    choice_change = np.where(np.sign(cur_z) != np.sign(all_z))[0].size
    num_ch_changed += choice_change 
    cur_conf = norm.cdf(np.abs(cur_z) * cpc.std_z ** -2 / ((ti * cpc.std_z ** -2 + cpc.std_i ** (-2)) ** .5))
    all_conf = norm.cdf(np.abs(all_z) * cpc.std_z ** -2 / ((t * cpc.std_z ** -2 + cpc.std_i ** (-2)) ** .5))
    cur_rate = cur_conf > th
    all_rate = all_conf > th
    rate_change = np.where (cur_rate != all_rate)[0].size
    num_rate_changed += rate_change
print ("choice changed = " , num_ch_changed / num_samples)
print ("rate changed = ", num_rate_changed / num_samples)





