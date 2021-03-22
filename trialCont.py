#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:23:09 2019

@author: koosha
"""

from trial import Trial
import scipy.io


class TrialCont(Trial):
    def __init__(self, coh, dur, cor_tar, ch_tar, sacc_point):
        super(TrialCont, self).__init__(coh, dur, cor_tar, ch_tar, 0) 
        self.conf_rep = sacc_point
    
    def getConfRep(self):
        return self.conf_rep
    
    @staticmethod
    def readFile(file_name):
        trials = []
        all_data = scipy.io.loadmat(file_name)['data'][0]
        num_trials = all_data[0][0].size
        for i in range(num_trials):
            coh = int(all_data[0][0][i][0] * 1000)
            dur = all_data[0][3][i][0]
            correct_target = all_data[0][1][i][0]
            choice_target = all_data[0][2][i][0]
            sacc_pos= all_data[0][4][i][0]
            trials.append(TrialCont(coh, dur, correct_target, choice_target, sacc_pos))
        return trials