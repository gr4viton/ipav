import time
from StepEnum import DataDictParameterNames as dd
import numpy as np

class Step():
    """ Step class - this is usefull comment, literally"""

    steps_count = 0

    def __init__(self, name, function, narrowed=False, controls=None):
        self.name = name
        self.new_name = '' # if changed in rename_stepwidget_label it changes stepwidget label
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

        self.last_widget_name_label = name

        self.controls = controls

        if controls:
            print('Got controls for step [{}]'.format(self.name))
            print(self.controls)

    def rename_stepwidget_label(self, new_name):
        if self.last_widget_name_label is not new_name:
            self.name = new_name
            self.new_name = new_name

    def check_renaming(self):
        new_name = self.data_post.get(dd.new_name, '')
        if new_name:
            self.rename_stepwidget_label(new_name)
        self.data_post[dd.new_name] = ''

    def run(self, data_prev):
        self.data_prev = data_prev.copy()


        # self.user_input = False # e.g. from snippet or gui
        self.user_input = self.controls is not None

        if self.user_input == True:
            # self.data_prev[dd.kernel] = (42,42) # from user
            self.data_prev = self.controls.get_control_values(data_prev)
            self.data_prev[dd.take_all_def] = False
        else:
            self.data_prev[dd.take_all_def] = True

        self.data_prev[dd.info_text] = ''
        start = time.time()
        self.data_post = self.function(self.data_prev)
        end = time.time()

        self.check_renaming()
        # new_name = self.data_post.get(dd.new_name,'')
        # if new_name:
        #     self.rename_stepwidget_label(new_name)


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
                info += ' (R=negative, G=positive)'

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
