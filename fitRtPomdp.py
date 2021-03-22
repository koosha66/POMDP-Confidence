import numpy as np
from pomdpReaction import POMDPReaction
from analysis import Analysis
from trialCont import TrialCont
from trial import Trial
from dataPoint import DataPoint

def accDurCoh (acc_r, sac_time = 0):
    cohs = sorted(trials_acc.keys())
    all_acc = np.zeros(len(cohs))
    all_dur = np.zeros(len(cohs))
    nt = np.zeros(len(cohs))
    for c, a_r_list in acc_r.items():
        i = np.where(cohs == c)[0][0]
        num_trials = 0
        for a_r_point in a_r_list:
            num_trials += a_r_point.nrData
            all_acc[i] += a_r_point.nrData * a_r_point.performance
            all_dur[i] += a_r_point.nrData * (a_r_point.avg_duration + sac_time) 
        if num_trials > 0:
            all_acc[i] = all_acc[i] / float(num_trials)
            all_dur[i] = all_dur[i] / float(num_trials)
        nt[i] = num_trials
    return all_acc, all_dur / 1000.0, nt


def likelihood(trials, model, sac_time):
    total_ll = 0
    for c, trials_c in trials.items():
        max_dur = trials_c[-1].duration 
        max_t = int(max_dur/model.step_time + 2) 
        lls = np.zeros([max_t, 2])
        for trial in trials_c:
            t = int(round((trial.duration-sac_time)/model.step_time))
            if t < 0:
                t = 0
            is_correct = (trial.correct_target == trial.choice_target)
            if lls[t, int(is_correct)] == 0:
                lls[t, int(is_correct)] = model.logLike(c/1000.0, t, is_correct)
            total_ll += lls[t, int(is_correct)]
    return total_ll 


first_k = 8
num_samples = 500
step_time = 25
mu_i = 0

num_quantiles = 2
trials = TrialCont.readFile('beh_data.S1.mat')
time_windows = Trial.sortAndDivideToQuantiles(trials, num_quantiles)
len_trials = len(trials)
#trials = trials[int(.025*len_trials):-int(.025*len_trials)] 
trials_sep_sure = Trial.seperateBySureShown(trials)
trials_acc = Trial.seperateByCoherence(trials_sep_sure[False])
analysis = Analysis(time_windows)
cohs = np.array(sorted(trials_acc.keys()))
num_trials = len(trials)
sac_time = trials[int(len(trials)/100)].duration

#acc_r = {}
#for c in cohs:
#    acc_r[c] = DataPoint.pointsFromTrialsQuantiles(trials_acc[c], time_windows, False)
 
#a, d, nt = accDurCoh(acc_r)



ks = np.array([3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.])
std_zs = np.array(range(0, 31))/10.00 + 1
costs = np.array(range(1, 15)) / 1000.00
std_is = np.array([.5, 1.0])



ks = np.array([ 2., 3., 4.])
std_zs = np.array(range(0, 41, 10))/10 + 1
costs = np.array(range(1, 15, 7)) / 1000
std_is = np.array([1])



#se_rt = np.zeros([len(ks), len(std_zs), len(costs), len(std_is)])
#se_acc = np.zeros([len(ks), len(std_zs), len(costs), len(std_is)])
ll = np.zeros([len(ks), len(std_zs), len(costs), len(std_is)])

for ki in range(len(ks)):
    for costi in range(len(costs)) :
        for std_zi in range(len(std_zs)):
            for std_ii in range(len(std_is)):
                k = ks[ki]
                cost = costs[costi]
                std_z = std_zs[std_zi]
                std_i = std_is[std_ii]
                rcpc = POMDPReaction(k, mu_i, std_i, cohs, cost, 20000, std_z, step_time, num_samples)
                ####
#                t_max = int (20000 / step_time) + 1
#                all_zs = np.array([])
#                for i in range(len(cohs)):
#                    mu = k * cohs[i]/ 1000.0
#                    file_name = 'observations/' + str(mu)  + '_' + str(int(std_i*100)) + '_' + str(int(std_z*100))  + '_' +  str(cost*1000) + '_' + str(t_max) + str(num_samples) + '.npy' 
#                    Z_t = np.load(file_name)
#                    for t in range(t_max-1):
#                        Z_cost = Z_t[:, int(t)]
#                        Z_post = Z_t[:, int(t + 1)]
#                        stopped = np.where (Z_post == 0)[0]
#                        Z_cost = Z_cost[stopped]
#                        Z_cost = Z_cost[np.where(Z_cost != 0)[0]]
#                        all_zs = np.concatenate([all_zs, Z_cost/t, -Z_cost/t])
#                print (np.std(all_zs)) 
                ####
                ll[ki, std_zi, costi, std_ii] = likelihood(trials_acc, rcpc, sac_time)
#                fake_trials = rcpc.generateTrials()
#                time_windows_f = Trial.sortAndDivideToQuantiles(fake_trials, num_quantiles) 
#                trials_sep_sure_f = Trial.seperateBySureShown(fake_trials)
#                trials_acc_f = Trial.seperateByCoherence(trials_sep_sure_f[False])
#                acc_f = analysis.generateForMultipleCohs(rcpc, trials_acc_f, cohs, 'Accuracy')                
#                af, df, ntf = accDurCoh(acc_f, sac_time)
#                se_acc[ki, std_zi, costi, std_ii] = np.sum(nt[1:]*(((a[1:] - af[1:])**2)**.5))/np.sum(nt[1:])
#                se_rt[ki, std_zi, costi, std_ii] = np.sum(nt*(((d - df)**2)**.5))/np.sum(nt)
#                comb = (se_acc[ki, std_zi, costi, std_ii] * (len(cohs)-1) + se_rt[ki, std_zi, costi, std_ii] * len(cohs))/(2*len(cohs)-1)
                #print (k, std_z, cost, std_i, comb, likelihood(trials_acc, rcpc, sac_time) )

#np.save(open('rt_errors','wb'), se_rt)
#np.save(open('acc_errors','wb'), se_acc) 
#np.save(open('ks_rt','wb'), ks)
#np.save(open('costs_rt','wb'), costs)
#np.save(open('std_is_rt','wb'), std_is)
#np.save(open('std_zs_rt','wb'), std_zs)

#combined_err =  (se_acc * (len(cohs)-1) + se_rt * len(cohs))/(2*len(cohs)-1)
cf = np.ndarray.flatten(ll)
s_cf = np.argsort(cf)
u_index = np.unravel_index(s_cf, ll.shape)
for i in range(1, first_k+1):
    print (ks[u_index[0][-i]], costs[u_index[2][-i]], std_zs[u_index[1][-i]], std_is[u_index[3][-i]], cf[s_cf[-i]])




