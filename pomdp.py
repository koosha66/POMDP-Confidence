from __future__ import division, print_function
from dataPoint import DataPoint
import numpy as np

class POMDP(object): 
    EPSILON = 10 ** (-40) 
    def __init__(self, k, std_z = 1.0, step_time = 10.0, num_samples = 20000):
        self.k = k
        self.std_z = float(std_z)
        self.step_time = step_time
        self.num_samples = num_samples
 
    def generateBelief(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            Z_pos = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            direction_prob_pos = self.calcDirectionPositiveProb(c, t, Z_pos)
            direction_prob_neg = 1- self.calcDirectionPositiveProb(c, t, Z_neg)
            direction_prob = np.concatenate((direction_prob_pos,direction_prob_neg))
            if direction_prob.size == 0:
                temp = DataPoint ()
                temp.change(point)
                temp.setPerformance(0)
                temp.nrData = 0
                prob_points.append(temp)
                continue
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
            Z = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            direction_prob = self.calcDirectionPositiveProb(c,t,Z)
            #direction_prob_neg = 1 - self.calcDirectionPositiveProb(c,t,Z_neg)
            #direction_prob = np.concatenate((direction_prob_pos,direction_prob_neg))
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
            Z = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z = np.concatenate((Z_pos, Z_neg))
            direction_prob = self.calcDirectionPositiveProb(c,t,Z)
            direction_left = np.where(direction_prob < .5)[0]
            direction_prob[direction_left] = 1 - direction_prob[direction_left]
            if direction_prob.size == 0:
                temp = DataPoint ()
                temp.change(point)
                temp.setPerformance(0)
                temp.nrData = 0
                prob_points.append(temp)
                continue
            mean_confidence = np.mean(direction_prob)
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(mean_confidence)
            prob_points.append(temp)
        return prob_points
    
    def generateConfidenceCorrect(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            if hasattr(self, 'calcDirectionPositiveProbCorrect'):
                direction_prob = self.calcDirectionPositiveProbCorrect(c, t, None)
            else:
                Z = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
                Z = Z[np.where(Z > 0)[0]]
                direction_prob = self.calcDirectionPositiveProb(c,t,Z)
                direction_prob = direction_prob[np.where(direction_prob > .5)[0]]
            temp = DataPoint ()
            temp.change(point)
            temp.nrData = direction_prob.size
            if direction_prob.size == 0:
                temp.setPerformance(0) 
            else:
                temp.setPerformance(np.mean(direction_prob))
            prob_points.append(temp)
        return prob_points
    
    def generateConfidenceIncorrect(self, data_points):
        prob_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = int (point.avg_duration  / self.step_time)
            if hasattr(self, 'calcDirectionPositiveProbIncorrect'):
                direction_prob = 1 - self.calcDirectionPositiveProbIncorrect(c, t, None)
            else:
                Z = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
                Z = Z[np.where(Z > 0)[0]]
                direction_prob = self.calcDirectionPositiveProb(c,t,Z)
                direction_prob = 1 - direction_prob[np.where(direction_prob < .5)[0]]
            temp = DataPoint()
            temp.change(point)
            temp.nrData = direction_prob.size
            if direction_prob.size == 0:
                temp.setPerformance(0) 
            else:
                temp.setPerformance(np.mean(direction_prob))
            prob_points.append(temp)
        return prob_points

    def generateWage(self, data_points, threshold):
        sure_points = []
        for point in data_points:
            c = point.avg_coherence / 1000.0
            mu = self.k * c
            t = point.avg_duration  / self.step_time
            Z = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            #Z = np.concatenate((Z_pos, Z_neg))
            confidence = self.calcDirectionPositiveProb(c, t, Z)
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
            t = point.avg_duration  / self.step_time
            Z = np.random.normal(mu * t, self.std_z * (t ** .5), self.num_samples)
            confidence = self.calcDirectionPositiveProb(c, t, Z)
            correct_rejects = np.where(confidence > threshold)[0].size
            wrong_rejects = np.where(1 - confidence > threshold)[0].size
            accuracy = correct_rejects / (correct_rejects + wrong_rejects)
            #Z_neg = np.random.normal(-mu * t, self.std_z * (t ** .5), self.num_samples)
            #confidence = 1 - self.calcDirectionPositiveProb(c, t, Z_neg)
            #correct_rejects += np.where(confidence > threshold)[0].size
            #wrong_rejects += np.where(1 - confidence > threshold)[0].size
            #accuracy = .5 * (accuracy + correct_rejects / (correct_rejects + wrong_rejects))
            temp = DataPoint ()
            temp.change(point)
            temp.setPerformance(accuracy)
            reject_points.append(temp)
        return reject_points
