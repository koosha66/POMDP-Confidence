import numpy as np
from pomdpReaction import POMDPReaction


num_samples = 500 
std_i = 20 
std_z  = 3.5 
step_time = 25
mu_i = 0
cohs = [0, 32, 64, 128, 256, 512]

ks = np.array([1.6, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
std_zs = np.array(range(0, 41))/10 + 1
costs = np.array(range(1, 15)) / 1000
std_is = [.5, 1, 2, 5, 10, 20]

for k in ks:
    for cost in costs:
        for std_z in std_zs:
            for std_i in std_is:
                print (k, std_z, cost, std_i)
                rcpc = POMDPReaction(k, mu_i, std_i, cohs, cost, 20000, std_z, step_time, num_samples)


        
