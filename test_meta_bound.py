#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:24:20 2018

@author: koosha
"""
import numpy as np
from scipy.stats import norm
from knownCWithBound import KnownCWithBound

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



num_samples = 200000
k = 1.108
c = 128 / 1000.0
mu = k * c
t = 40
std_z = 1
std_zs = 1
th = .734
bound = 6
cpc = KnownCWithBound(k, c * 1000, bound, 420, std_z, 10, num_samples)

Z = np.random.normal(mu * t, std_z * (t ** .5), num_samples)
direction_prob = cpc.calcDirectionPositiveProb(c, t, Z)

correct_trials = np.where(direction_prob>.5)[0]
incorrect_trials = np.where(direction_prob<.5)[0]

HR = np.where(direction_prob > th)[0].size/correct_trials.size
FR = np.where(direction_prob < 1 - th)[0].size/incorrect_trials.size
d = norm.ppf(correct_trials.size/num_samples)- norm.ppf(incorrect_trials.size/num_samples)
crit = -.5 * (norm.ppf(correct_trials.size/num_samples) + norm.ppf(incorrect_trials.size/num_samples))

print ("accuracy = ", correct_trials.size/num_samples)
print ("P(high) =" , (np.where(direction_prob > th)[0].size + np.where(direction_prob < 1 - th)[0].size) / num_samples)
print ("d = ", d, "criteria = ", crit) 
print ("HR = ", [HR])
print ("FR = ", [FR])
metad = norm.ppf(HR) - norm.ppf(FR)
print (metad)
print (metad/d)

