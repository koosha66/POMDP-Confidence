#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:24:20 2018

@author: koosha
"""
import numpy as np
from scipy.stats import norm
from pomdpReactionSz import POMDPReactionSz
from analysis import Analysis
from trial import Trial
from dataPoint import DataPoint


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
#num_samples = 20000
cost =  5 * 10 ** -3
k = 1
std_i = 5
std_z  = .75
step_time = 10
sz = .4
cohs = [128]

th = .74
rcpc = POMDPReactionSz(k, 0, sz, std_i, cohs, cost, 2000, std_z, step_time, num_samples)
fake_trials = rcpc.generateTrials()
time_windows_f = Trial.sortAndDivideToQuantiles(fake_trials, 2) 
trials_sep_sure_f = Trial.seperateBySureShown(fake_trials)
trials_acc_f = Trial.seperateByCoherence(trials_sep_sure_f[False])
analysis = Analysis(time_windows_f)
acc_results_r = analysis.generateForMultipleCohs(rcpc, trials_acc_f, cohs, 'Accuracy')
dur, prob= DataPoint.pointsToPlotForm(acc_results_r[128])
confidence_results_r = analysis.generateForMultipleCohs(rcpc, trials_acc_f, cohs, 'Confidence')
confidence_results_r_correct = analysis.generateForMultipleCohs(rcpc, trials_acc_f, cohs, 'ConfidenceCorrect')
confidence_results_r_error = analysis.generateForMultipleCohs(rcpc, trials_acc_f, cohs, 'ConfidenceIncorrect')

HR = 0 
FR = 0
for t in range(1, 100):
    probs = rcpc.calcDirectionPositiveProbCorrect(.128, t, None)
    HR += np.where(probs>th)[0].size
    fprobs = 1- rcpc.calcDirectionPositiveProbIncorrect(.128, t, None)
    FR += np.where(fprobs>th)[0].size

print ("ASDASD", HR, FR)
print ("P(high) =" , (HR + FR) / num_samples)

HR /= prob[0] * num_samples
FR /= (1 - prob[0]) * num_samples    
d = norm.ppf(prob[0])- norm.ppf(1-prob[0])
crit = -.5 * (norm.ppf(prob[0]) + norm.ppf(1-prob[0]))

print ("accuracy = ", prob[0])
print ("d = ", d, "criteria = ", crit) 
print ("HR = ", [HR])
print ("FR = ", [FR])
#metad = norm.ppf(HR) - norm.ppf(FR)
#print (metad)
#print (metad/d)






