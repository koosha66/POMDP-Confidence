from __future__ import division, print_function
from scipy.stats import norm as normal
#from scipy.stats import chisqprob
import numpy as np
from trial import Trial
from math import log
from discretePomdp import DiscretePOMDP
from continuousPomdp import ContinuousPOMDP
from dataPoint import DataPoint
from pomdpWithCostSz import POMDPWithCostSz
from fit import Fit
from analysis import Analysis
from environment import Environment
import random 
import copy
from sklearn import metrics

EPSILON = 10 ** (-40)
class StatTest:    

    @staticmethod
    def randomizeTrials(trials_dict):
        random_trials = []
        for key, trials in trials_dict.items():
            for i in range(len(trials)):
                from_trial = random.randint(0, len(trials)-1)
                random_trials.append(trials[from_trial])
        return random_trials
                
    @staticmethod
    def logLikelihood (trial_points, pomdp, method_name, threshold = None):
        method = getattr(pomdp, method_name)
        if threshold == None:
            predictions = method(trial_points)
        else:
            predictions = method(trial_points, threshold)
        ll = 0
        for real_point, prediction in zip(trial_points, predictions):
            if prediction.performance > EPSILON:
                ll += real_point.performance * real_point.nrData * log(prediction.performance)
            if 1 - prediction.performance > EPSILON:
                ll += (1 - real_point.performance) * real_point.nrData * log(1 - prediction.performance)
        return ll
    
    @staticmethod
    def voungTest (trial_points, pomdp1, pomdp2, method_name, diff_param = 0, thresholds = None):
        method1 = getattr(pomdp1, method_name)
        method2 = getattr(pomdp2, method_name)
        if thresholds == None:
            predictions1 = method1(trial_points)
            predictions2 = method2(trial_points)
        else:
            predictions1 = method1(trial_points, thresholds[0])
            predictions2 = method2(trial_points, thresholds[1])
        Nw2 = 0  # N * w^2 in the test
        N = 0
        ll1 = 0
        ll2 = 0 
        for real_point, prediction1, prediction2 in zip(trial_points, predictions1, predictions2):
            # 1 trials
            ll1_i = log(max(prediction1.performance,EPSILON))
            ll2_i = log(max(prediction2.performance,EPSILON))
            Nw2 += real_point.performance * real_point.nrData * ((ll1_i - ll2_i)**2)
            N += real_point.nrData
            ll1 += real_point.performance * real_point.nrData * ll1_i
            ll2 += real_point.performance * real_point.nrData * ll2_i
            # 0 trials
            ll1_i = log(max(1 - prediction1.performance,EPSILON))
            ll2_i = log(max(1 - prediction2.performance,EPSILON))
            Nw2 += (1 - real_point.performance) * real_point.nrData * ((ll1_i - ll2_i)**2)
            ll1 += (1 - real_point.performance) * real_point.nrData * ll1_i
            ll2 += (1 - real_point.performance) * real_point.nrData * ll2_i    
        print (N, log(N))
        return ll1, ll2, (ll1 - (ll2 + .5 * diff_param * log(N))) / (Nw2 **.5), (ll1 - ll2) / (Nw2 **.5)  
 
    @staticmethod
    def squaredError (trial_points, pomdp, method_name, threshold = None):
        method = getattr(pomdp, method_name)
        if threshold == None:
            predictions = method(trial_points)
        else:
            predictions = method(trial_points, threshold)
        error = 0
        nrData = 0
        for real_point, prediction in zip(trial_points, predictions):
            error += real_point.nrData * ((prediction.performance-real_point.performance) ** 2)
            nrData += real_point.nrData
        return error / nrData 
    
    @staticmethod
    def RSquared (trial_points, pomdp, method_name, threshold = None):
        method = getattr(pomdp, method_name)
        if threshold == None:
            predictions = method(trial_points)
        else:
            predictions = method(trial_points, threshold)
        prediction_array = []
        real_array = []
        #nrData_array = []
        for real_point, prediction in zip(trial_points, predictions):
            for i in range(real_point.nrData):
                prediction_array.append(prediction.performance)
                real_array.append(real_point.performance)
        real_array = np.array(real_array)
        prediction_array = np.array(prediction_array)
        res = (real_array - prediction_array) ** 2
        tot = (real_array - np.mean(real_array)) ** 2
        return 1 - np.sum(res)/np.sum(tot)
  
    @staticmethod
    def bootstrapForK (trials, fit, num_for_bootstrap = 100):
        time_window = fit.step_time
        all_ks = np.zeros(num_for_bootstrap)
        for i in range (num_for_bootstrap):
            random_trials = StatTest.randomizeTrials(Trial.seperateByCoherenceAndTime(trials, time_window))    
            all_ks[i] = fit.findK(random_trials, k_start = 1, num_search = 30)
            print (all_ks[i])
        start = all_ks.size * 2.5/100.0   # 5% interval
        end = all_ks.size - start 
        all_ks = np.sort(all_ks)
        return all_ks[int(start)], all_ks[int(end)]
    
    @staticmethod
    def bootstrapForCost (trials, fit, num_for_bootstrap = 100, filename = "bootCost.txt"):
        time_window = fit.step_time
        all_ks = np.zeros(num_for_bootstrap)
        all_costs = np.zeros(num_for_bootstrap)
        for i in range (num_for_bootstrap):
            random_trials = StatTest.randomizeTrials(Trial.seperateByCoherenceAndTime(trials, time_window))
            k = fit.findK(random_trials, k_start = 1, num_search = 30)
            k = float(format(k,'.2f'))
            ks = k * np.ones(6) + .01 * np.array(range(6))
            c1 = np.power(.1, np.array(range(15,0,-1)))
            costs = np.concatenate([[0], c1,  c1 * 5])
            costs = np.sort(costs)
            random_trials_coh = Trial.seperateByCoherence(random_trials)
            cohs = sorted(random_trials_coh.keys())
            trial_points = []
            for c in cohs:
                points = DataPoint.pointsFromTrialsFixedWindow(random_trials_coh[c], fit.step_time, False)
                trial_points.extend(points)
            all_ks[i], all_costs[i], ll = fit.findCost(random_trials, trial_points, cohs, ks, costs, num_samples)
            f = open(filename, "a")
            f.write(str(all_ks[i])+ ", " + str(all_costs[i]) +"\n")
            f.close()
            print (all_costs[i])

    @staticmethod
    def bootstrapForContinuousTh (trials, k, sd_i, sd_z, fit, num_for_bootstrap = 100):
        time_window = fit.step_time
        all_ths = np.zeros(num_for_bootstrap)
        for i in range(num_for_bootstrap):
            random_trials = StatTest.randomizeTrials(Trial.seperateByCoherenceAndTime(trials, time_window))    
            all_ths[i] = fit.findContinuousTh(random_trials, k, sd_i, sd_z, th_start = .85, num_search = 30)
            print (all_ths[i])
        start = all_ths.size * 2.5/100.0
        end = all_ths.size - start 
        all_ths = np.sort(all_ths)
        return all_ths[int(start)], all_ths[int(end)]
    
    @staticmethod
    def bootstrapForPomdpTh (trials, pomdp, num_for_bootstrap = 100, filename = "bootTh.txt"):
        time_window = pomdp.step_time
        all_ths = np.zeros(num_for_bootstrap)
        for i in range(num_for_bootstrap):
            random_trials = StatTest.randomizeTrials(Trial.seperateByCoherenceAndTime(trials, time_window))
            random_trials_coh = Trial.seperateByCoherence(random_trials)
            cohs = sorted(random_trials_coh.keys())
            trial_points = []
            for c in cohs:
                points = DataPoint.pointsFromTrialsFixedWindow(random_trials_coh[c], pomdp.step_time, True)
                trial_points.extend(points)
            all_ths[i] = Fit.findThbyPOMDP(trial_points, pomdp, "generateWage", th_start = .6, num_search =30)
            f = open(filename, "a")
            f.write(str(all_ths[i])+"\n")
            f.close()
            print (all_ths[i])
        start = all_ths.size * 2.5/100.0
        end = all_ths.size - start 
        all_ths = np.sort(all_ths)
        return all_ths[int(start)], all_ths[int(end)]
    
num_samples = 2000
norm = 1
std_z = 1.6
num_quantiles = 10
step_time = 10.0
mu_i = 0.0
monkey = 1

all_trials = Trial.readFile('beh_data.monkey' + str(monkey)+ '.mat')
if monkey == 1:
    trials = all_trials
else:
    trials = []
    for trial in all_trials:
        if trial.coherence == 16:
            continue
        trials.append(trial)
time_windows = Trial.sortAndDivideToQuantiles(trials, num_quantiles) 
trials_sep_sure = Trial.seperateBySureShown(trials)
trials_acc = Trial.seperateByCoherence(trials_sep_sure[False])
trials_sure = Trial.seperateByCoherence(trials_sep_sure[True])
trials_sure_reject = Trial.seperateByCoherence(Trial.seperateBySureChosen(trials_sep_sure[True])[False])
cohs = sorted(trials_acc.keys())

### bootstrap for K
fit = Fit(std_z, step_time)
#print(StatTest.bootstrapForK(trials_sep_sure[False], fit, 100))
#print(StatTest.bootstrapForCost(trials_sep_sure[False], fit, 10, "bootCost1.txt"))


#### bootstrap for th

#print(StatTest.bootstrapForContinuousTh(trials_sep_sure[True], k, std_i, std_z, fit, 100))
#fit.findThTwoFree(trials_sep_sure[True], k, std_i, 1.00, num_samples, num_search = 30)



####find cost 
trial_points = []
for c in cohs:
    points = DataPoint.pointsFromTrialsFixedWindow(trials_acc[c], 10, False)
    trial_points.extend(points)
#
#ks = [1.108, 1.118]
#costs = []
#for e in [-5, -4, -6]:
#    for i in [1, 2, 5, 7]:
#        costs.append(i * (10**e))
#print (fit.findCost(trials, trial_points, cohs, ks, costs, num_samples = 20000))



### Voung's test for pomdp cost
#cost = 7 * (10 ** -6)
#k = .60
#std_i = fit.findSdI(trials, k, cohs)
#cpc = POMDPWithCost(k, 0.0, std_i, cohs, cost, 1000, 1.0, step_time, num_samples)
#print ("done")
#cp = ContinuousPOMDP(.591, 0.0, .253, 1.0, step_time, num_samples)
#print (StatTest.voungTest (trial_points, cpc, cp, 'generateAccuracy', 1.0, None))

### Voung's test 
k = 1
std_i = .46 #.87
std_z = 1.6 #3.59
w_z = .9 #.1.69 
cost = 10**-4
num_samples = 20000
monkey = 1
cp1 = ContinuousPOMDP(k, 0, std_i, std_z, step_time, num_samples) #adjusted
#cost m1: 0.87 2e-05 0.44824569478922 1.5950203212145724 -22212.52311004393
#cost m2: Best: 1.67 4e-05 0.88 3.6 -17600.572674803956
cp2 = POMDPWithCostSz(k, 0, .87, .45, cohs, 2 * (10 ** -5), 1000, 1.6, step_time, num_samples)
#cp2 = POMDPWithCostSz(k, 0, 1.67, .88, cohs, 4 * (10 ** -5), 1000, 3.6, step_time, num_samples)
env1 = Environment(cp1, k, w_z, step_time, num_samples)
ll1, ll2, z_score, z0_score = StatTest.voungTest (trial_points, cp2, env1, 'generateAccuracy', 1, None)
print (ll1,ll2, z_score, z0_score)
print(normal.sf(abs(z_score)), normal.sf(abs(z0_score)))


##### Bootstrap for th (general POMDP)
#k = 1.108
#dp = DiscretePOMDP(k, cohs, std_z, step_time, num_samples)
#print (StatTest.bootstrapForPomdpTh(trials_sep_sure[True],dp, 10, "bootThDiscrete.txt"))




#cp = ContinuousPOMDP(k, mu_i, .25, 1.570, step_time, num_samples)
#env = Environment(cp, k, std_z, step_time, num_samples)
#acc_results = analysis.generateForMultipleCohs(env, trials_acc, cohs[1:], 'Accuracy')
#sq_cp = StatTest.squaredError(trial_points, env, 'generateAccuracy')
#print (sq_cp)
#rs = StatTest.RSquared(trial_points, env, 'generateAccuracy')
#print (rs)
#confidence_results = analysis.generateForMultipleCohs(env, trials_acc, cohs, 'Confidence')
#wage_results = analysis.generateForMultipleCohs(env, trials_sure, cohs, 'Wage', threshold_s)
#reject_results = analysis.generateForMultipleCohs(env, trials_sure, cohs[1:], 'Reject', threshold_s)

#cp2 = ContinuousPOMDP(k, mu_i, 0.44, std_z, step_time, num_samples)
#print(statTest.logLikelihood(trial_points, cp1, 'generateAccuracy'))
#print(statTest.logLikelihood(trial_points, cp2, 'generateAccuracy'))
 


#for i in [1]:
 #   std_i = 1.05 #i * best_std_i
 #   for sz in [.85,.86, .87, .88, .89, .90, 91, .92, .93]: 
 #       th = fit.findWageParamsDirectZ(trials_sep_sure[True], k, best_std_i, sz, th_start = .8, num_search = 30)    
 #       print (std_i, sz, th, end = ' ')
 #       cp = ContinuousPOMDP(k, mu_i, std_i, std_z, step_time, num_samples)
 #       env = Environment(cp, k, std_z, step_time, num_samples)
 #       ll_cp = statTest.logLikelihood(trial_points, env, 'generateWage', th)
 #       print (ll_cp, end = ' ')
 #       sq_cp = statTest.squaredError(trial_points, cp, 'generateWage', th)
 #       print (sq_cp)
    
    
#cp1 = ContinuousPOMDP(1.108, mu_i, .33, std_z, step_time, num_samples)
#cp2 = ContinuousPOMDP(k, mu_i, 0.44, std_z, step_time, num_samples)

#print (statTest.voungTest (trial_points, cp1, cp2, 'generateWage', [.736, .751]))
#ths = [.581, .606, .5631, .661, .689, .715, .736, 0.751, .761, 0.767, 0.771,  0.773, .774, .775]
#for i in range(-3,4, 2):
#    std_i = (1.1532 ** i) * best_std_i
#    th = fit.findWageParamsDirect(trials_sep_sure[True], k, std_i, th_start = .85, num_search = 50)    
#    cp = ContinuousPOMDP(k, mu_i, std_i, std_z, step_time, num_samples)
#    print (std_i, th, end = ' ')
#    ll_cp = statTest.logLikelihood(trial_points, cp, 'generateWage', th)
#    print (ll_cp, end = ' ')
#    sq_cp = statTest.squaredError(trial_points, cp, 'generateWage', th)
#    print (sq_cp)
    #wage_results = analysis.generateForMultipleCohs(cp, trials_sure, cohs, 'Wage',th)
    #error = 0
    #nr_data = 0
    #for j in range(len(cohs)):
    #    c = cohs[j]
    #    points = DataPoint.pointsFromTrialsQuantiles(trials_sure[c], time_windows, True)
    #    err, nd = Analysis.calcError(points, wage_results[c], 2) 
    #    nr_data += nd
    #    error +=  err
    #print (error/ float(nr_data))


    

 
    
# print(statTest.bootstrapForK(trials_sep_sure[False], step_time, 1000))
#print(statTest.bootstrapForTh(trials_sep_sure[True], k, std_i, step_time, 60))

#dp = DiscretePOMDP(k, cohs, std_z, step_time, num_samples)
#cp = ContinuousPOMDP(k, mu_i, std_i, std_z, step_time, num_samples)



#ll_cp = statTest.logLikelihood(trial_points, cp, 'generateWage', th)
#print (ll_cp)
#for th_acc in [.64, .75, .68, .69, .70]:
#    print (th_acc, ":, ", statTest.logLikelihood(trial_points, dp, 'generateWage', th_acc))
#print (ll_dp)
#LR = 2 * (ll_cp - ll_dp) 
#p = chisqprob(LR, 1) 
#print ('p: %.30f' % p )

#trial_points = []
#for c in cohs:
#    points = DataPoint.pointsFromTrialsFixedWindow(trials_acc[c], 10.0, False)
#    trial_points.extend(points)


#ll_cp = statTest.logLikelihood(trial_points, cp, 'generateAccuracy')
#print (ll_cp)
#ll_dp = statTest.logLikelihood(trial_points, dp, 'generateAccuracy')
#print (ll_dp)
