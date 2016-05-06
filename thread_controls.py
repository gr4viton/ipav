import timeit
import time
import threading

from multiprocessing import Queue

import cv2
import numpy as np

import findHomeography as fh

from StepControl import *

class LockedValue(object):
    """
    Thread safe numpy array
    """
    def __init__(self, init_val = None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        self.val = val
        self.lock.release()

class LockedNumpyArray(object):
    """
    Thread safe numpy array
    """
    def __init__(self, init_val = None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            # ret_val = self.val.copy()
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        # self.val = val.copy() # ????????????????????????????????????????? do i need a copy??
        self.val = val
        self.lock.release()


class Chain():
    tag_names = []
    chain_names = []
    load_data_chain_names = []

    data = LockedValue()

    delimiter = ','
    strip_chars = ' \t'
    def load_steps_from_file(self, path):
        with open (path, "r") as myfile:
            string = myfile.read
        self.load_steps_from_string(string)

    def load_steps_from_string(self, string):
        self.step_names_list = string.replace('\n', self.delimiter).split(self.delimiter)
        # print(self.step_names)
        self.step_names_list = [step_name.strip(self.strip_chars) for step_name in self.step_names_list]

        self.step_names = [step for step in self.step_names_list if step != '']
        # print(self.step_names)

    def __init__(self, name, start_chain=True, path='', ):
        self.name = name
        self.tag_search = name in self.tag_names

        if path != '':
            self.load_steps_from_file(path)
        else:
            if self.name in ['standard']:

                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'sobel vertical']
                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'sobel horizontal']\
                self.step_names = ['original', 'resize', 'gray', 'thresholded', 'laplacian']
                self.step_names = ['original', 'gauss', 'resize']

                # self.step_names = ['original', 'resize', 'gray', 'detect red', 'blender cube']
                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'blender cube']
                # self.step_names = ['original', 'resize', 'detect red']
                # self.step_names = ['original', 'resize', 'rgb stack']

                # string = 'original, resize, gauss, resize'
                string = 'original'
                if start_chain:
                    self.load_steps_from_string(string)

        if self.name in self.load_data_chain_names:
            self.load_data()


    def load_data(self):
        # self.data = LockedValue( [1,2,3,42,69] )
        self.data = [1, 2, 3, 42, 69]
        pass


class ChainControl():
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

        self.current_chain = Chain('new_chain')
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
        if self.chain_computing == False:
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
        self.mean_execution_time = np.sum(self.execution_time) / len(self.execution_time)

    def chain_loop(self):
        while self.chain_computing:
            self.do_chain()

    def do_chain(self):
        start = time.time()
        data = StepData()
        data[dd.captured] = [stream.frame for stream in self.capture_control.streams]
        # data[dd.im] = self.capture_control.image_stream_control.frame
        data[dd.im] = data[dd.captured][0]
        data[dd.resolution_multiplier] = self.resolution_multiplier

        self._step_control.step_all(data)

        end = time.time()
        self.add_exec_times(end-start)

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




class ImageStreamControl():
    """
    Shared class to control source capture execution
    """

    # frame = np.ones( (32,24,3,), np.uint8 ) * 128
    # frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 128 )

    def __init__(self, source_id=0):
        # self.frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 128 )
        self.frame = np.ones( (32,24,3,), np.uint8 ) * 128

        self.capturing = LockedValue(False)

        self.capture_lock = threading.Lock()
        self.capture = None

        self.init_capture()

        self.source_id = source_id

        self.sleepTime = 0.0

    def init_capture(self):
        self.capture_lock.acquire()
        self.capture = cv2.VideoCapture()
        self.capture_lock.release()

    def open_source_id(self, new_source_id):
        self.capture_lock.acquire()
        self.source_id = new_source_id
        self.capture_lock.release()
        self.open_source_id()

    def open_capture(self):
        self.capture_lock.acquire()
        self.capture.open(self.source_id)
        if self.capture.isOpened() != True:
            print('Source[', self.source_id, '] Cannot open capture.')
            return False
        print('Source[', self.source_id, '] Opened capture.')
        self.capture_lock.release()
        return True

    def toggle_source_id(self):
        self.capture_lock.acquire()
        #self.source_id = np.mod(self.source_id+1, 2)
        # self.open_capture()
        try_next = True
        while try_next:
            self.source_id += 1
            self.capture.open(self.source_id)

            if self.capture.isOpened() != True:
                print('Source[', self.source_id, '] Cannot open capture.')
                self.source_id = -1
                continue

            ret, frame = self.capture.read()
            if ret == False:
                print('Source[', self.source_id, '] Cannot be read from')
                self.source_id = -1
                continue
            print('Source[', self.source_id, '] Opened capture')
            try_next = False
        self.capture_lock.release()

    def close_capture(self):
        # self.capture_lock.acquire()
        self.capture.release()
        print('Source[', self.source_id, '] Released capture.')
        # self.capture_lock.release()

    def start_capturing(self, blocking = True):
        if blocking == False:
            if self.open_capture() == False:
                return False
            self.capturing = True
            self.thread = threading.Thread(target=self.capture_loop)
            self.thread.start()
            return True
        else:
            return self.start_capturing_blocking()

    def start_capturing_blocking(self, min_height = 50, iterations = 9999999):
        if self.start_capturing(blocking=False) == False:
            return False

        print('Source[', self.source_id, '] Captured frame with dimensions', self.frame.shape,
              '. Waiting until the height is greater than', min_height, 'px')
        looping = iterations + 1

        while looping > 1:
            if self.frame is not None:
                if self.frame.shape[0] < min_height:
                    pass
                else:
                    print('Source[', self.source_id, '] Captured frame with dimensions', self.frame.shape,
                          '. Continuing with program execution.')
                    return True
            looping -= 1
        print('Source[', self.source_id, '] Did not captured frame with greater height than',
              min_height, 'px in ', iterations, 'iterations.')
        return False

    def stop_capturing(self):
        print('Source[', self.source_id, '] Stopped capturing.')
        self.capturing = False

    def toggle_capturing(self):
        if self.capturing == False:
            self.start_capturing()
        else:
            self.capturing = False

    def on_stop(self):
        """ stops capturing and releases capture object """
        self.capturing = False
        self.close_capture()

    def capture_loop(self):
        while self.capturing:
            self.capture_frame()
            time.sleep(self.sleepTime)  # frames?

    def capture_frame(self):
        self.capture_lock.acquire()
        if self.capture.isOpened() != True:
            # self.open_capture()
            print('Source[', self.source_id, '] Cannot read frame as the capture is not opened')
            self.capture_lock.release()
            return False
        else:
            ret, frame = self.capture.read()
            self.capture_lock.release()
            self.frame = frame
            return True
        # print(self.frame.shape)