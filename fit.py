from __future__ import division, print_function
from scipy.stats import norm as normal
import numpy as np
from dataPoint import DataPoint
from math import log, log2
#from discretePomdp import DiscretePOMDP
from continuousPomdp import ContinuousPOMDP
from environment import Environment
from pomdpWithCostSz import POMDPWithCostSz
#from pomdpWithBound import POMDPWithBound
from trial import Trial
import pickle 


EPSILON = 10 ** (-40)

class Fit: 
    def __init__(self, sd_z = 1.0, step_time = 10.0):
        self.sd_z = sd_z
        self.step_time = step_time
    
    # find sensiivity parameter k based on accuracy assuming w_z = 1
    def findK(self, trials, k_start, num_search = 50):
        k = k_start
        alpha = 10 ** -6
        for i in range(num_search):
            dll = 0
            for trial in trials:
                t = trial.getDuration() / float(self.step_time)
                c = trial.getCoherence()
                mu = k * c / 1000.0
                if trial.getChoiceTarget() == trial.getCorrectTarget():
                    dll += c * (t ** .5) *  normal.pdf(mu * (t ** .5)) / normal.cdf(mu * (t ** .5))
                else:
                    dll -= c * (t ** .5) * normal.pdf(mu * (t ** .5)) / normal.cdf(-mu * (t ** .5))
            print (k, dll)
            if dll * dll < 400: #** -3:
                break
            k = k + alpha * dll
            if k < 0:
                k = 0
            if k > 1.5:
                k = 1.5
            alpha = .9 * alpha
        return k

    # find real std (w_z) assuming k = 1    
    def findSd(self, trials, sd_start, num_search = 50):
        sd = sd_start
        alpha = 10 ** -3
        for i in range(num_search):
            dll = 0
            for trial in trials:
                t = trial.getDuration() / float(self.step_time)
                c = trial.getCoherence() / 1000
                if trial.getChoiceTarget() == trial.getCorrectTarget():
                    dll += c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(c * (t ** .5) / sd)
                else:
                    dll -= c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(-c * (t ** .5) / sd)
            #print (sd, dll)
            if dll * dll < .00001: #** -3:
                break
            sd = sd + alpha * dll
            if sd < 0.1:
                sd = 0.1
            if sd > 5:
                sd = 5
            alpha = .99 * alpha
        return sd
    
    # find sigma_z from subject's view when they adjust their confidence with accuracy 
    # assuming k_z is fixed (for example 1) and w_z as the free parameter (here given)    
    def findSdSubject(self, trials, k, w_z, sd_i, sd_start, num_search = 50):
        sd_z = sd_start
        alpha = .001
        for i in range(num_search):
            dll = 0
            for trial in trials:
                t = int(trial.getDuration() / float(self.step_time))
                c = trial.getCoherence()
                mu = k * c / 1000.0
                Z = np.abs(np.random.normal(mu * t, (t **.5) * w_z, 1))[0]
                var_t = t*(sd_z ** 2) + ((sd_i ** -2) *(sd_z ** 4))
                d_gaus = -Z * .5 * (var_t ** -1.5) * (2 * t * sd_z + 4 * (sd_i ** -2) * (sd_z**3))
                if trial.getChoiceTarget() == trial.getCorrectTarget():
                    dll += d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(Z /(var_t**.5)), EPSILON) 
                else:
                    dll -= d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(-Z /(var_t**.5)), EPSILON)
            #print (sd_z, dll)
            if dll * dll < 5 * 5:
                break
            sd_z = sd_z + alpha * dll
            if sd_z < 0.01:
                sd_z = 0.01
            if sd_z > 5:
                sd_z = 5
            alpha = .9 * alpha
        return sd_z
    
        
    # find sigma_0 for the Gaussian prior 
    def findSdI(self, trials, k, w_z):
        s = []
        for trial in trials:
            t = trial.getDuration() / float(self.step_time)
            sigma = w_z/ float(t ** .5)
            c = trial.getCoherence()
            mu = k * c / 1000.0
            if trial.getCorrectTarget() > 1: #left
                mu = -mu
            s.append(np.random.normal(mu, sigma))
        s = np.array(s)
        return np.std(s)

    def findSdIUncertain(self, trials, k, w_z, sd_z):
        s = []
        for trial in trials:
            t = trial.getDuration() / float(self.step_time)
            sigma = w_z/ float(t ** .5)
            c = trial.getCoherence()
            mu = k * c / 1000.0
            if trial.getCorrectTarget() > 1: #left
                mu = -mu
            s.append(np.random.normal(np.random.normal(mu, sigma), sd_z/ float(t ** .5)))
        s = np.array(s)
        return np.std(s)
    
    # find consistant sigma_z and sd_i together 
    def findSdSubjectConsistant(self, trials, k, w_z, sd_start, num_search = 50):
        sd_i = self.findSdI(trials, k, w_z)
        sd_z =  self.findSdSubject(trials, k, w_z, sd_i, sd_start, num_search)
        #print ("Subject's SDs:", sd_i, sd_z)
        sd_i_new = self.findSdIUncertain(trials, k, w_z, sd_z)
        i = 1
        while (abs(sd_i-sd_i_new) > .01 and i < 10):
            i += 1
            sd_i = sd_i_new
            sd_z =  self.findSdSubject(trials, k, w_z, sd_i, sd_start, num_search)
            #print ("Subject's SDs:", sd_i, sd_z)
            sd_i_new = self.findSdIUncertain(trials, k, w_z, sd_z)
        if i > 9:
            print ("NO CONVERGE", sd_i, sd_i_new)
        return sd_i, sd_z
        
    
    # find the threshold based on wagering in the continuous POMDP
    def findContinuousTh(self, trials, k, sd_i, sd_z, th_start, num_search = 50):
        th = th_start
        alpha = .05
        for i in range(num_search):
            dll = 0
            for trial in trials:
                t = int (trial.getDuration() / self.step_time)
                c = trial.getCoherence()
                mu = k * c / 1000.0
                inv_th = normal.ppf(th) * (sd_z ** 2) * ((t * (sd_z ** -2) + sd_i ** (-2)) ** .5)
                inv_not_th = normal.ppf(1 - th) * (sd_z ** 2) * ((t * (sd_z ** -2) + sd_i ** (-2)) ** .5)
                mu_Z = t * mu
                sigma_Z = t ** .5
                reject = normal.cdf((inv_th - mu_Z) / sigma_Z) - normal.cdf((inv_not_th - mu_Z) / sigma_Z)
                d_reject = normal.pdf((inv_th - mu_Z) / sigma_Z) * (1 / (EPSILON + normal.pdf(normal.ppf(th)))) + normal.pdf((inv_not_th - mu_Z) / sigma_Z) * (1 / (EPSILON + normal.pdf(normal.ppf(1-th))))
                if trial.getChoiceTarget() == 3:
                    dll += d_reject / (reject + EPSILON)
                else:
                    dll -= d_reject / (1-reject + EPSILON)
            print (th, dll)
            if (dll * dll) < (20 * 20):
                return th
            th = th + alpha * (dll > 0) - alpha * (dll < 0)
            if th < 0.5:
                th = 0.5
            if th > .9999:
                th = .9999
            alpha = .9 * alpha
        return th
    
    # log Likelihood of our fits: Accuracy, wagering, and accuracy upon rejection of sure bet
    @staticmethod
    def logLikelihood (trial_points, pomdp, method_name, threshold = None):
        method = getattr(pomdp, method_name)
        if threshold == None:
            predictions = method(trial_points)
        else:
            predictions = method(trial_points, threshold)
        ll = 0
        for real_point, prediction in zip(trial_points, predictions):
            ll += real_point.performance * real_point.nrData * log(max(prediction.performance, EPSILON))
            ll += (1 - real_point.performance) * real_point.nrData * log(max(1 - prediction.performance, EPSILON))
        return ll
    
    # Fitting to Wagering behavior given any POMDP model  
    @staticmethod
    def findThbyPOMDP(trial_points, pomdp, method_name = "generateWage", th_start = .75, num_search = 50):
        th = th_start
        alpha = .05
        for i in range(num_search):
            ll = Fit.logLikelihood(trial_points, pomdp, method_name, th)
            ll_eps = Fit.logLikelihood(trial_points, pomdp, method_name, th+.005)
            dll = ll_eps - ll
            print (th, dll)
            if (dll * dll) < (20 * 20):
                return th
            th = th + alpha * (dll > 0) - alpha * (dll < 0)
            if th < 0.5:
                th = 0.5
            if th > .9999:
                th = .9999
            alpha = .9 * alpha
        return th                                                                                                                                                                                                                                                       
        
    def fitWithCost(self, trials, trial_points, cohs, k, num_samples = 2000): 
        best_ll = -1000000
        best_cost = 0 
        best_wz = 0 
        try:
            stds = pickle.load(open( "stds.p", "rb" ))
        except: 
            stds = {}
        costs= (10 ** -5) * np.array(list(range(2, 10))) # + list(range(10, 50, 10)))#+ list(range(100, 1000, 100)))
        w_zs = [.9, .89, .88, .87, .86, .85, .84, .83, .82, .81, .8, .75, .7, .65]
        w_zs = [1.69, 1.68, 1.67, 1.66] # 1.65, 1.64, 1.63, 1.6, 1.55, 1.5]
        for w_z in w_zs:
            if w_z in stds.keys():
                sd_i, std_z = stds[w_z]
                sd_i = round(sd_i, 2)
                std_z = round (std_z, 2)
            else:
                sd_i, std_z = fit.findSdSubjectConsistant(trials, k , w_z, w_z, num_search = 30)
                sd_i = round(sd_i, 2)
                std_z = round (std_z, 2)
                stds[w_z] = (sd_i, std_z)
                pickle.dump(stds, open( "stds.p", "wb" ))
            for cost in costs:
                cp = POMDPWithCostSz(k, 0, w_z, sd_i, cohs, cost, 1000, std_z, self.step_time, num_samples)
                ll = Fit.logLikelihood(trial_points, cp, "generateAccuracy")
                print ( "Cost, w_z, and ll:", cost, w_z, ll)
                if ll > best_ll:
                    best_ll = ll
                    best_cost = cost
                    best_wz = w_z
                    best_std_i = sd_i
                    best_std_z = std_z
        print ("Best: ", best_wz, best_cost, best_std_i, best_std_z, best_ll)
                

    # find real std (w_z) assuming k = 1    
    def findSdIter(self, trials, sd_start, chunk_size = 100):
        sd = sd_start
        trials_len = len(trials)
        alpha = 10 ** -2 / chunk_size
        for i in range(1, int(trials_len/chunk_size)):
            dll = 0
            for trial in trials[int((i-1)* chunk_size):int(i * chunk_size)]:
                t = trial.getDuration() / float(self.step_time)
                c = trial.getCoherence() / 1000
                if trial.getChoiceTarget() == trial.getCorrectTarget():
                    dll += c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(c * (t ** .5) / sd)
                else:
                    dll -= c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(-c * (t ** .5) / sd)
            print (sd, dll)
            sd = sd + alpha * dll
            if sd < 0.1:
                sd = 0.1
            if sd > 5:
                sd = 5
            alpha = .9999 * alpha 
        return sd

    # find real std (w_z) assuming k = 1    
    def findSdSubjectIter(self, trials, sd_start, sd_subject_start):
        all_sds = np.zeros(len(trials))
        all_sd_is = np.zeros(len(trials))
        all_sd_zs = np.zeros(len(trials))
        sd = sd_start
        sd_z = sd_subject_start
        sum_Z2 = 0
        num = 0
        i = 0
        alpha = 1 / 600
        for trial in trials:
            t = trial.getDuration() / float(self.step_time)
            c = trial.getCoherence() / 1000
            if trial.getChoiceTarget() == trial.getCorrectTarget():
                dll = c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(c * (t ** .5) / sd)
            else:
                dll = -c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(-c * (t ** .5) / sd)
            #print ("sd:", sd, dll)
            sd = sd + alpha * dll
            Z = np.abs(np.random.normal(c * t, (t **.5) * sd, 1))[0]
            mu = Z / t 
            mu_n = np.random.normal(mu, sd_z/ float(t ** .5), 100)
            #Zw = np.abs(np.random.normal(Z / t, sd_z/ (t **.5) , 10)) 
            sum_Z2 += np.sum(mu_n **2)
            num += 100
            sd_i = (sum_Z2 / num)**.5
            #print ("sd_i:", sd_i)
            var_t = t * (sd_z ** 2) + ((sd_i ** -2) *(sd_z ** 4))
            d_gaus = -Z * .5 * (var_t ** -1.5) * (2 * t * sd_z + 4 * (sd_i ** -2) * (sd_z**3))
            if trial.getChoiceTarget() == trial.getCorrectTarget():
                dll = d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(Z /(var_t**.5)), EPSILON) 
            else:
                dll = -d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(-Z /(var_t**.5)), EPSILON)
            #print ("sd_z:", sd_z, dll)
            sd_z = sd_z + alpha * dll
            if sd < 0.1:
                sd = 0.1
            if sd > 5:
                sd = 5
            if sd_z < 0.1:
                sd_z = 0.1
            if sd_z > 5:
                sd_z = 5
            #alpha = (1 - 10**-5) * alpha
            all_sds[i] = sd
            all_sd_zs[i] = sd_z
            all_sd_is[i] = sd_i
            i +=1
        return all_sds, all_sd_zs, all_sd_is
    


    # find real std (w_z) assuming k = 1    
    def findSdSubjectIterBatch(self, trials, sd_start, sd_subject_start,  chunk_size = 100):
        sd = sd_start
        sd_z = sd_subject_start
        old_sd = 1000
        sd_i = 1
        trials_len = len(trials)
        for i in range(1, int(trials_len/chunk_size) + 1):
            dll = 10**30
            alpha = 10 ** -4 * int(trials_len/chunk_size) / i
            counter = 0
            while (abs(dll) > 20 and counter < 20):
                dll = 0
                counter +=1
                for trial in trials[0:int( i * chunk_size)]:
                    t = trial.getDuration() / float(self.step_time)
                    c = trial.getCoherence() / 1000
                    if trial.getChoiceTarget() == trial.getCorrectTarget():
                        dll += c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(c * (t ** .5) / sd)
                    else:
                        dll -= c * (t ** .5) * -(sd ** -2) * normal.pdf(c * (t ** .5) / sd) / normal.cdf(-c * (t ** .5) / sd)
                print ("w_z", sd, dll)
                sd = sd + alpha * dll
                if sd < 0.1:
                    sd = 0.1
                if sd > 10:
                    sd = 10
                alpha = .99 * alpha 
            # SD_subject 
            dll = 10**30
            alpha = 10 ** -4 * int(trials_len/chunk_size) / i
            counter = 0
            if abs(sd - old_sd) > .1:
                sd_i = self.findSdI(trials[0:int(i * chunk_size)], 1, sd)
            old_sd = sd
            print ("sd_i", sd_i)
            while (abs(dll) > 20 and counter < 20):
                dll = 0
                counter +=1
                for trial in trials[0:int(i * chunk_size)]:
                    t = trial.getDuration() / float(self.step_time)
                    c = trial.getCoherence() / 1000
                    Z = np.abs(np.random.normal(c * t, (t **.5) * sd, 1))[0]
                    var_t = t * (sd_z ** 2) + ((sd_i ** -2) *(sd_z ** 4))
                    d_gaus = -Z * .5 * (var_t ** -1.5) * (2 * t * sd_z + 4 * (sd_i ** -2) * (sd_z**3))
                    if trial.getChoiceTarget() == trial.getCorrectTarget():
                        dll += d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(Z /(var_t**.5)), EPSILON) 
                    else:
                        dll -= d_gaus * normal.pdf(Z /(var_t**.5)) / max(normal.cdf(-Z /(var_t**.5)), EPSILON)
                print ("sd_z:", sd_z, dll)
                sd_z = sd_z + alpha * dll
                if sd_z < 0.1:
                    sd_z = 0.1
                if sd_z > 15:
                    sd_z = 15
                alpha = .99 * alpha 
        return sd, sd_z
    
    @staticmethod
    def calcAccuracy(trials):
        correct_choices = 0
        for trial in trials:
            if trial.getChoiceTarget() == trial.getCorrectTarget():
                correct_choices +=1
        return correct_choices / len(trials)

    @staticmethod
    def calcProbSure(trails):
        correct_choices = 0
        for trial in trials:
            if trial.getChoiceTarget() == trial.getCorrectTarget():
                correct_choices +=1
        return correct_choices / len(trials)            

    
# for monkey 2 data set monkey to 2
monkey = 1
##### cost : 0.87 2e-05 0.44824569478922 1.5950203212145724 -22212.52311004393
####### cost 2 :  1.67 4e-05 0.88 3.6 -17600.572674803956

all_trials = Trial.readFile('beh_data.monkey' + str(monkey)+ '.mat')
if monkey == 1:
    trials = all_trials
else:
    trials = []
    for trial in all_trials:
        if trial.coherence == 16:
            continue
        trials.append(trial)
trials_sep_sure = Trial.seperateBySureShown (trials)
trials_acc = Trial.seperateByCoherence(trials_sep_sure[False])
trials_sure = Trial.seperateByCoherence(trials_sep_sure[True])
trials_sure_reject = Trial.seperateByCoherence(Trial.seperateBySureChosen(trials_sep_sure[True])[False])
cohs = sorted(trials_acc.keys())
print (cohs)

trial_points = []
for c in cohs:
    points = DataPoint.pointsFromTrialsFixedWindow(trials_acc[c], 10, False)
    trial_points.extend(points)


#
num_samples = 20000
num_search = 30
num_quantiles = 10
step_time = 10
mu_i = 0.0
fit = Fit(1, step_time)
trial_num = len(trials_sep_sure[False])

# chunk_size = 5000
# step = 4000
# start = 0
# end = start + chunk_size
# all_w = []
# all_i = []
# all_z = []
# while (end < trial_num):
#     print (start, end, Fit.calcAccuracy(trials_sep_sure[False][start:end]))
#     w_z = fit.findSd(trials_sep_sure[False][start:end], .9, 30)
#     sdi, sd_z = fit.findSdSubjectConsistant(trials_sep_sure[False][start:end], 1, w_z, 1.6, 30)
#     print (w_z, sdi, sd_z)
#     start = start + step
#     end = start + chunk_size
#     all_w.append(w_z)
#     all_i.append(sdi)
#     all_z.append(sd_z)

#print(fit.findSdSubjectIter(trials_sep_sure[False][:int(len(trials_sep_sure[False])/10)], 1, 1, 100))
#fit.fitWithCost(trials_sep_sure[False], trial_points, cohs, 1, num_samples)

#print (fit.findSdSubject(trials_sep_sure[False], 1 , .9, .29, 1, num_search = 50))

#print (fit.findSdSubjectConsistant(trials_sep_sure[False], 1 , .9, 1, num_search = 50))

#print (fit.findSdSubjectSZ(trials_sep_sure[False], 1.108, .29, 1, 30))
#print (fit.findSdSubjectSZ(trials_sep_sure[False], 1, 1.69, .43, 1, 30))

#ks = [1.1, 1.08, 1.09, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.3, 1.4, 1.5, 1.6]
#c1 = np.power(.1, np.array(range(15,0,-1)))
#costs = np.concatenate([[0], c1, c1 * 2.5, c1 * 5, c1*7.5])
#costs= np.sort(costs)
#print (costs)
#print (fit.findCost(trials, trial_points, cohs, ks, costs, num_samples))

#k = fit.findK(trials_sep_sure[False], .24, 30)
#print (k)
#dp = DiscretePOMDP(k,cohs, sd_z, step_time, num_samples)
trial_points = []
for c in cohs:
  points = DataPoint.pointsFromTrialsFixedWindow(trials_sure[c], 10, True)
  trial_points.extend(points)

#cpc = POMDPWithCost(1.108, mu_i, .329, cohs, 10 ** -11 , 1000, sd_z, step_time, num_samples)
ap = ContinuousPOMDP(1, 0, .46, 1.60, step_time, num_samples) #adjusted 
env = Environment(ap, 1, .90, step_time, num_samples)  


#because the sigma of adjusted is different
#th = Fit.findThbyPOMDP(trial_points, env, "generateWage", .7, num_search)
#print (th)
#cp = ContinuousPOMDP(k, mu_i, .33, 1.00, step_time, num_samples)
#env = Environment(cp, k, sd_z, step_time, num_samples)
#print (Fit.logLikelihood(trial_points, env, "generateWage", .67))
#print (Fit.logLikelihood(trial_points, env, "generateWage", .78))

#trial_points = []
#for c in cohs:
#    points = DataPoint.pointsFromTrialsFixedWindow(trials_acc[c], step_time, False)
#    trial_points.extend(points)
#ks = [1.108, 1.13]
#bounds = [50, 60, 70, 80, 100, 2000]
#print (fit.findBound(trials, trial_points, cohs, ks, bounds, num_samples))
   
   
   
    
    
    

#k = fit.findK(trials_sep_sure[False], k_start, num_search)
#print ("k = ", k)
#sd_i = fit.findConfidenceParams(trials, k, cohs)
#print ("sd_i = ", sd_i)
#sd_z_s = fit.findSdSubject(trials_sep_sure[False], k, sd_i, sd_z, num_search)
#print ("sd_z_s = ", sd_z_s)

#th, sd_i = fit.findWageParamsFreeVar(trials, k, 1, .8, num_search = 50)
#print (th, sd_i)

#for k in [1.08, 1.09, 1.10]:
#    print (k)
#    fit = Fit(sd_z, step_time)
#    sd_i = fit.findConfidenceParams(trials, k, cohs)
#    pomdp_params = {'k': k, 'mu_i' : mu_i, 'sd_i' : sd_i, 'all_cohs' : cohs , 'cost' : cost_start, 'time_max' : 900, 'sd_z' : sd_z , 'step_time' : step_time, 'time_windows' : [0], 'norm': norm, 'num_samples': num_samples}
#    trial_points = []
#    for c in cohs:
#	points = DataPoint.pointsFromTrialsFixedWindow(trials_acc[c], step_time, False)
#        trial_points.extend(points)
#    file_name = "ll_lists/" + str(k) + ".txt" 
#    f = open(file_name, "w")
#    cost, ll  = fit.findAccParamsWithCost(trials, trial_points, pomdp_params, 50)
#    f.write(str(cost) + ',' + str(ll) + '\n')


#num_search = 30
#cost_start = .05

#k_start = 1
#th_start = .75
#fit = Fit(sd_z, 5)

#k = 1.13 / 1.4  #fit.findAccParams(trials_sep_sure[False], k_start, num_search)
#print ("best k", k)
#sd_i = fit.findConfidenceParams(trials, k, cohs)
#print ("sdi", sd_i)
#th = fit.findWageParamsDirect(trials_sep_sure[True], k, sd_i, th_start, num_search)
#print ("best th", th)


#cost = 0.0002
#trials_acc = Trial.sortTrials(trials_acc)
#cohs = sorted(trials_acc.keys())
#cpc = POMDPWithCost(k, 0, sd_i, cohs, cost, 900, sd_z, step_time, 10.0 , 1, num_samples = 10000)
#print ("cpc initiated")
#th = fit.findWageParams(trials_sep_sure[True], cpc, th_start)
#print ("best th", th)
