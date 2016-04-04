import time
from StepEnum import DataDictParameterNames as dd
import numpy as np

class Step():
    """ Step class - this is usefull comment, literally"""
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.execution_time_len = 15
        self.execution_time = 0
        self.execution_times = []
        self.mean_execution_time = 0

    def run(self, data_prev):
        self.data_prev = data_prev.copy()

        self.user_input = False # e.g. from snippet or gui
        if self.user_input == True:
            self.data_prev[dd.kernel] = (42,42) # from user
            self.data_prev[dd.take_all_def] = False
        else:
            self.data_prev[dd.take_all_def] = True

        start = time.time()
        self.data_post = self.function(self.data_prev)
        end = time.time()
        self.add_exec_times(end-start)

        return self.data_post

    def get_info_string(self):
        info = self.str_mean_execution_time()
        info += "\n" + str(self.data_post[dd.im].shape)
        return info

    def add_exec_times(self, tim):
        if len(self.execution_times) > self.execution_time_len:
            self.execution_times.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_times.append(tim)
        self.mean_execution_time = np.sum(self.execution_times) / len(self.execution_times)

    def str_mean_execution_time(self, sufix=' ms'):
        return '{0:.2f}'.format(round(self.mean_execution_time * 1000,2)) + sufix
