from dataPoint import DataPoint
from scipy.stats import norm, mvn
import numpy as np
from continuousPomdp import ContinuousPOMDP

class ClosedFormContinuousPOMDP(ContinuousPOMDP):
    def __init__(self, k, mu_i, std_i, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(ClosedFormContinuousPOMDP, self).__init__(k, mu_i, std_i, std_z, step_time, num_samples)

    def generateConfidence(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = point.avg_duration  / self.step_time
            # same direction
            low = np.array([-200, -200])
            mu_2d = np.array([0, 0])
            var_1 = 2 * t + self.std_i ** (-2)
            var_2 =  t
            upp = np.array([t * mu / (var_1 ** .5), t * mu / (var_2 ** .5)])
            cor = (var_2/var_1) ** .5
            cov = np.array([[1, cor], [cor, 1]])
            p, i = mvn.mvnun(low, upp, mu_2d, cov)
            # opposite direction
            p2, i = mvn.mvnun(low, -upp, mu_2d, cov)
            mean_confidence =  p + p2
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(mean_confidence)
            prob_points.append(temp)
        return prob_points


    def generateBelief(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = point.avg_duration  / self.step_time
            belief = norm.cdf(t * mu / ((2 * t + self.std_i ** (-2)) ** .5))
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(belief)
            prob_points.append(temp)
        return prob_points

    def generateAccuracy(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = point.avg_duration  / self.step_time
            acc = norm.cdf((t ** .5) * mu)
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(acc)
            prob_points.append(temp)
        return prob_points