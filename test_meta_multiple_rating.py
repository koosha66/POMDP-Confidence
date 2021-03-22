#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:24:20 2018

@author: koosha
"""
import numpy as np
from scipy.stats import norm
from pomdpWithCostSz import POMDPWithCostSz

ths = [.6]
num_samples = 100000
k = 1.0
c = 128 / 1000.0
mu = k * c
t = 40
std_z = .9
std_zs = 1.6
cpc = POMDPWithCostSz(k, 0, std_z, .46, [128], 10 ** -2, 420, std_zs, 10, num_samples) 



Z = np.random.normal(mu * t, std_z * (t ** .5), num_samples)
direction_prob = cpc.calcDirectionPositiveProb(c, t, Z)

correct_trials = np.where(direction_prob>.5)[0]
incorrect_trials = np.where(direction_prob<.5)[0]
HR = []
FR = []
all_HR = 0
all_FR = 0
for i in range(len(ths)-1, -1, -1):
    th = ths[i]
    new_HR = np.where(direction_prob > th)[0].size - all_HR
    print (new_HR)
    all_HR += new_HR
    HR.append(all_HR/correct_trials.size)
    new_FR = np.where(direction_prob < 1 - th)[0].size - all_FR
    all_FR += new_FR
    FR.append(all_FR/incorrect_trials.size)

d = norm.ppf(correct_trials.size/num_samples)- norm.ppf(incorrect_trials.size/num_samples)
crit = -.5 * (norm.ppf(correct_trials.size/num_samples) + norm.ppf(incorrect_trials.size/num_samples))

print ("accuracy = ", correct_trials.size/num_samples)
print ("P(high) =" , (np.where(direction_prob > th)[0].size + np.where(direction_prob < 1 - th)[0].size) / num_samples)
print ("d = ", d, "criteria = ", crit) 
print ("HR = ", HR)
print ("FR = ", FR)

