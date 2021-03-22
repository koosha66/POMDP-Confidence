from operator import attrgetter
import scipy.io
import glob 

class Trial:
    SURE  = 3
    def __init__(self, coh, dur, cor_tar, ch_tar, sure):
        self.coherence = int(coh)
        self.duration = dur
        self.correct_target = cor_tar
        self.choice_target = ch_tar
        self.sure_shown = sure
    
    def getDuration(self):
        return self.duration

    def getCoherence(self):
        return self.coherence

    def getCorrectTarget (self):
        return self.correct_target
        
    def getChoiceTarget(self):
        return self.choice_target
    
    @staticmethod
    def readFile(file_name):
        trials = []
        all_data = scipy.io.loadmat(file_name)['data'][0]
        #monkey = all_data[0][0][0][0]
        num_trials = all_data[0][1].size
        for i in range(num_trials):
            coh = int(all_data[0][1][i][0])
            dur = all_data[0][2][i][0]
            correct_target = all_data[0][3][i][0]
            choice_target = all_data[0][4][i][0]
            sure_shown = all_data[0][5][i][0]
            trials.append(Trial(coh, dur, correct_target, choice_target, sure_shown))
        return trials

    @staticmethod
    def readFileRtM(directory_name='b_rt'):     
        trials = []
        for file_name in glob.glob(directory_name + '/*'):
            all_data = scipy.io.loadmat(file_name)['data'][0][1]
            num_trials = all_data.shape[0]
            for i in range(num_trials):
                coh = all_data[i][4]
                if not(coh > -1):
                    continue
                choice_target = all_data[i][11]
                correct_target = all_data[i][11]
                if all_data[i][12] < 1:
                    correct_target = 2 - correct_target
                dur = all_data[i][36] - all_data[i][33]
                sure_shown = False
                trials.append(Trial(coh, dur, correct_target, choice_target, sure_shown))
        return trials

    @staticmethod
    def seperateByCoherence (trials):
        trials_dict = {}
        for trial in trials:
            key = trial.getCoherence()
            if key not in trials_dict:
                trials_dict[key] = []
            trials_dict[key].append(trial)
        return trials_dict
        
    @staticmethod    
    def seperateByCoherenceAndTime(trials, time_window = 1):
        trials_dict = {}
        for trial in trials:
            key = (trial.getCoherence(), int(trial.getDuration()/time_window))
            if key not in trials_dict:
                trials_dict[key] = []
            trials_dict[key].append(trial)
        return trials_dict
    
    @staticmethod
    def seperateBySureShown (trials):
        trials_dict = {False:[], True:[]}
        for trial in trials:
            key = bool (trial.sure_shown)
            trials_dict[key].append(trial)
        return trials_dict
    
    @staticmethod
    def filterByTime (trials, min_time, max_time):
        f_trials = []
        for trial in trials:
            if trial.duration >= min_time and trial.duration <= max_time:
                f_trials.append(trial)
        return f_trials

    @staticmethod
    def seperateBySureChosen (trials):
        trials_dict = {False:[], True:[]}
        for trial in trials:
            key = bool (trial.choice_target == Trial.SURE)
            trials_dict[key].append(trial)
        return trials_dict

    @staticmethod
    def sortAndDivideToQuantiles(trials, num_quantiles): 
        trials.sort(key=attrgetter('duration'))        
        window_size = int(len(trials)/num_quantiles)
        time_windows = [] 
        for i in range (window_size, len(trials), window_size):
            time_windows.append(trials[i].duration + .5)
        return time_windows

    @staticmethod
    def copy(trial):
        return Trial(trial.coherence, trial.duration, trial.correct_target, trial.choice_target, trial.sure_shown)



    
