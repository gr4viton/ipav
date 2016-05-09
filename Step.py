import time
from StepEnum import DataDictParameterNames as dd
import numpy as np

class Step():
    """ Step class - this is usefull comment, literally"""

    steps_count = 0

    def __init__(self, name, function, narrowed=False):
        self.name = name
        self.function = function
        self.execution_time_len = 23
        self.execution_time = 0
        self.execution_times = []
        self.mean_execution_time = 0
        self.synonyms = []
        self.origin = None

        self.data_prev = None
        self.data_post = None

        self.narrowed = narrowed

        self.id = Step.steps_count
        Step.steps_count +=1


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
        data = self.data_post
        info = ""
        if data is None:
            return info
        # info += self.str_mean_execution_time()
        im = data[dd.im]
        if im is not None:
            info += str(im.shape) + 'px'
            info += ', ' + str(im.dtype)
            if im.dtype == np.float:
                info += ' (red=negative, green=positive)'

        if data[dd.info] == True:
            info += '\n' + data[dd.info_text]

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
