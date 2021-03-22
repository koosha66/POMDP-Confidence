from __future__ import division, print_function
from dataPoint import DataPoint

class Analysis:
    def __init__(self, time_windows = [0]):
        self.time_windows = time_windows
    # Possible values of analysis_param: "Belief", "Confidence", "Wage", "Accuracy", "Reject"  
    def generateForMultipleCohs(self, pomdp, trials, cohs, analysis_param = 'Belief', threshold = None):
        results = {}
        for c in cohs:
            is_wage = (analysis_param == 'Wage') 
            real_points = DataPoint.pointsFromTrialsFixedWindow(trials[c], pomdp.step_time, is_wage) 
            method = getattr(pomdp, 'generate' + analysis_param)
            if threshold == None: 
                generated_points = method(real_points)
            else:
                generated_points = method(real_points, threshold) 
            generated_points_avg = DataPoint.pointsFromPoints(generated_points, self.time_windows)    
            results[c] = generated_points_avg
        return results
    
    @staticmethod
    def calcError(main_points, fitted_points, norm = 1):
        if len(main_points) != len(fitted_points):
            return 100000, 10000
        error = 0
        nr_data = 0
        list_main_points = list(main_points)
        list_fitted_points = list(fitted_points)
        for i in range(len(list_main_points)):
            p1 = list_main_points[i]
            p2 = list_fitted_points[i]
            #if p1.avg_duration != p2.avg_duration:
                #return 100000
            error += p1.nrData * (abs(p1.performance - p2.performance) ** norm)
            nr_data += p1.nrData
        return error, nr_data