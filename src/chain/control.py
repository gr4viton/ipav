import time
import threading
import numpy as np

from step.control import StepControl
from step.enums import DataDictParameterNames as dd
from step.data import StepData
from chain.thread_safe_data import LockedValue, LockedNumpyArray
from chain.base import Chain


class ChainControl:
    """
    Shared class to control findtag algorythm execution
    """

    step_control = LockedNumpyArray()
    seen_tags = LockedNumpyArray()
    chain_computing = LockedValue(False)

    # execution_time = LockedValue([])
    mean_execution_time = LockedValue(0)

    # def __init__(self, capture_control, tag_names, selected_chain_name):
    def __init__(self, capture_control, current_chain):
        self.current_chain = current_chain

        self.capture_control = capture_control
        self.execution_time_len = 50
        self.execution_time = []
        self.resolution_multiplier = 0.5

        # self.streams = None

        self._step_control = StepControl(self.resolution_multiplier, self.current_chain)

    def reset_step_control(self):
        self._step_control.select_steps(self.current_chain)

    def add_show_load_chain(self, show_load_chain_fnc):
        self.show_load_chain_fnc = show_load_chain_fnc

    def show_load_chain(self):
        self.show_load_chain_fnc()

    def load_chain(self, string):
        self.on_stop()
        time.sleep(1)

        self.current_chain = Chain("new_chain")
        self.current_chain.load_steps_from_string(string)

        self.reset_step_control()
        self.start_computing()
        # string = self.show_load_chain_fnc()

        return self.current_chain.step_names

    def get_available_steps(self):
        # print(self._step_control.available_steps)
        return self._step_control.available_steps

    def start_computing(self):
        self.chain_computing = True
        self.thread = threading.Thread(target=self.chain_loop)
        self.thread.start()

    def toggle_computing(self):
        if not self.chain_computing:
            self.start_computing()
        else:
            self.chain_computing = False

    def on_stop(self):
        self.chain_computing = False

    def add_exec_times(self, tim):
        if len(self.execution_time) > self.execution_time_len:
            self.execution_time.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_time.append(tim)
        self.mean_execution_time = np.sum(self.execution_time) / len(
            self.execution_time
        )

    def chain_loop(self):
        while self.chain_computing:
            self.do_chain()

    def do_chain(self):
        start = time.time()
        data = StepData()

        data[dd.capture_control] = self.capture_control

        # data[dd.im] = self.capture_control.stream.frame

        # source_id = 0
        # stream = data[dd.capture_control].get_stream(source_id)

        index = 0
        stream = data[dd.capture_control].streams[index]
        data[dd.stream] = stream
        data[dd.im] = stream.frame

        # data[dd.captured] = [stream.frame for stream in self.capture_control.streams]
        # data[dd.im] = data[dd.captured][index]

        # data[dd.resolution_multiplier] = self.resolution_multiplier

        self._step_control.step_all(data)

        end = time.time()
        self.add_exec_times(end - start)

        # not thread safe
        self.step_control = self._step_control
        self.seen_tags = self._step_control.seen_tags

        # here raise an event for the conversion and redrawing to happen
        # time.sleep(0.0001)

    def update_findtag_gui(self, frame, tag_model, running_findtag):

        while True:
            if running_findtag:
                self.do_chain()

    def set_resolution_div(self, resolution_multiplier):
        self.resolution_multiplier = resolution_multiplier
