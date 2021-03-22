from __future__ import division, print_function
from dataPoint import DataPoint
from trial import Trial
from scipy.stats import norm
import numpy as np
from pomdp import POMDP

class Environment(POMDP): 
    EPSILON = 10 ** (-40) 
    def __init__(self, pomdp, k, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        super(Environment,self).__init__(k, std_z, step_time, num_samples)
        self.pomdp = pomdp
 
    def generateBelief(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            direction_prob_pos = self.pomdp.calcDirectionPositiveProb(c, t, Z_pos)
            direction_prob_neg = 1- self.pomdp.calcDirectionPositiveProb(c, t, Z_neg)
            direction_prob = np.concatenate((direction_prob_pos,direction_prob_neg))
            mean_belief = np.mean(direction_prob)
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(mean_belief)
            prob_points.append(temp)
        return prob_points

    def generateAccuracy(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            direction_prob_pos = self.pomdp.calcDirectionPositiveProb(c,t,Z_pos)
            direction_prob_neg = 1 - self.pomdp.calcDirectionPositiveProb(c,t,Z_neg)
            direction_prob = np.concatenate((direction_prob_pos,direction_prob_neg))
            accuracy = (np.where(direction_prob > .5)[0].size + .5 * np.where(direction_prob == .5)[0].size) / direction_prob.size
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(accuracy)
            prob_points.append(temp)
        return prob_points

    def generateConfidence(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            Z = np.concatenate((Z_pos, Z_neg))
            direction_prob = self.pomdp.calcDirectionPositiveProb(c,t,Z)
            direction_left = np.where(direction_prob < .5)[0]
            direction_prob[direction_left] = 1 - direction_prob[direction_left]
            mean_confidence = np.mean(direction_prob)
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(mean_confidence)
            prob_points.append(temp)
        return prob_points

    def generateWage(self, data_points, threshold):
        sure_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int(point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            Z = np.concatenate((Z_pos, Z_neg))
            #Z = Z_pos
            confidence = self.pomdp.calcDirectionPositiveProb(c, t, Z)
            confidence = confidence[np.where(confidence > 1 - threshold)[0]]
            num_sure = np.where(confidence <= threshold)[0].size 
            prob_sure = num_sure / float(Z.size)
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(prob_sure)
            sure_points.append(temp)
        return sure_points

    def calcDirectionPositiveProb(self, c, t, Z):
        raise NotImplementedError("Subclass must implement abstract method")

    def generateReject(self, data_points, threshold):
        reject_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int(point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            confidence = self.pomdp.calcDirectionPositiveProb(c, t, Z_pos)
            reject_indices = np.append(np.where(confidence > threshold)[0], np.where( 1 - confidence > threshold)[0])
            Z_pos = Z_pos[reject_indices]
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            #confidence = self.pomdp.calcDirectionPositiveProb(c, t, Z_pos)
            #reject_indices = np.append(np.where(confidence > threshold)[0], np.where(1 - confidence > threshold)[0])
            #Z_neg = Z_neg[reject_indices]
            accuracy = np.where(Z_pos > 0)[0].size / Z_pos.size
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(accuracy)
            reject_points.append(temp)
        return reject_points


