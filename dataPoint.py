from trial import Trial
from trialCont import TrialCont
import numpy as np

class DataPoint:
    def __init__(self):
        self.nrData = 0
        self.avg_coherence = 0.0
        self.avg_duration = 0.0
        self.performance = 0.0
        self.conf_rep = 0.0

    def change(self, data_point):
        self.nrData = data_point.nrData
        self.avg_coherence = data_point.avg_coherence
        self.avg_duration = data_point.avg_duration
        self.performance = data_point.performance
        self.conf_rep = data_point.conf_rep
    
    def setPerformance (self, p):
        self.performance = p

    def getDuration(self):
        return self.avg_duration

    def addTrial(self, trial, trial_performance):
        self.avg_duration = (self.nrData * self.avg_duration + trial.getDuration()) / (float) (self.nrData + 1)
        self.avg_coherence = (self.nrData * self.avg_coherence + trial.getCoherence()) / (float) (self.nrData + 1)
        self.performance = (self.nrData * self.performance +  trial_performance) / (float) (self.nrData + 1)
        if isinstance(trial, TrialCont):
            self.conf_rep = (self.nrData * self.conf_rep + trial.getConfRep()) / (float) (self.nrData + 1)
        self.nrData = self.nrData + 1 

    def addPoint(self, point):
        if point.nrData == 0:
            return
        self.avg_duration = (self.nrData * self.avg_duration + point.nrData * point.avg_duration) / (float) (self.nrData + point.nrData)
        self.avg_coherence = (self.nrData * self.avg_coherence + point.nrData * point.avg_coherence) / (float) (self.nrData + point.nrData)
        self.performance = (self.nrData * self.performance + point.nrData * point.performance) / (float) (self.nrData + point.nrData)
        self.conf_rep = (self.nrData * self.conf_rep + point.nrData * point.conf_rep) / (float) (self.nrData + point.nrData)
        self.nrData = self.nrData + point.nrData
        
    @staticmethod
    def pointsFromTrialsFixedWindow(trials, window_len = 1, sure_option = False):
        if window_len < 1:
            window_len = 1
        data_points = {}
        for trial in trials:
            performance = (trial.getCorrectTarget() == trial.getChoiceTarget())
            if sure_option:
                performance = (trial.getChoiceTarget() == Trial.SURE)
            index = (int) (trial.getDuration() / window_len)
            if index not in data_points:
                data_points[index] = DataPoint()
            data_points[index].addTrial(trial, performance)
        return data_points.values()

    
    @staticmethod
    def pointsFromTrialsQuantiles(trials, time_windows = [0] , sure_option = False):
        data_points = {}
        for trial in trials:
            performance = (trial.getCorrectTarget() == trial.getChoiceTarget())
            if sure_option:
                performance = (trial.getChoiceTarget() == Trial.SURE)        
            for index in range(len(time_windows)):
                if trial.duration < time_windows[index]:
                    break
            if index not in data_points:
                data_points[index] = DataPoint()
            data_points[index].addTrial(trial, performance)
        return data_points.values()

    @staticmethod
    def pointsFromPoints(points, time_windows = []):
        data_points = {}
        for point in points:
            for index in range(len(time_windows)):
                if point.avg_duration < time_windows[index]:
                    break
            if index not in data_points:
                data_points[index] = DataPoint()
            data_points[index].addPoint(point)
        return data_points.values()

    ## return x(duration), y(performance)
    @staticmethod
    def pointsToPlotForm(points):
        per = []
        dur = []
        for point in points:
            per.append(point.performance)
            dur.append(point.avg_duration)
        return np.array(dur), np.array(per)

