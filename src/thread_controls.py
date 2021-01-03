import time
import threading


import cv2
import numpy as np

import os

import capture_control as cc  # circular dependency if CaptureControl imported right away

from chain.thread_safe_data import LockedValue  # , LockedNumpyArray


class ImageStreamControl:
    """
    Shared class to control source capture execution
    """

    # frame = np.ones( (32,24,3,), np.uint8 ) * 128
    # frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 128 )
    already_selected = []
    count = 0
    invalid_values = [-1, 2009211520]

    def __init__(self, source_id=0):
        # self.frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 128 )
        self.frame = (
            np.ones(
                (
                    32,
                    24,
                    3,
                ),
                np.uint8,
            )
            * 128
        )

        self.dir_cv2_cap_prop = cc.CaptureControl.dir_cv2_cap_prop
        self.cv2_dict_name = {}
        for cv2_name in self.dir_cv2_cap_prop:
            self.cv2_dict_name[getattr(cv2, cv2_name)] = cv2_name
            # print(word, '=', getattr(cv2, word))

        self.capturing = LockedValue(False)

        self.capture_lock = threading.Lock()
        self.capture = None

        self.init_capture()

        self.source_id = source_id

        self.sleepTime = 0.0

        self.focal = 400

        self.name = "unitialized" + str(ImageStreamControl.count)
        ImageStreamControl.count += 1

    def print_all_properties(self):
        # width = 640
        # height = 480
        # width, height = [640, 480]
        # width, height = [800, 600]
        width, height = [1024, 768]

        # stream.capture.set(3, width)
        # stream.capture.set(4, height)
        print("Source [{}] Printing all properties:".format(self.source_id))

        txt = ""
        for prop_name in self.dir_cv2_cap_prop:
            # super(cv2,prop_name)
            # print(globals())
            # print('loc')
            # print(locals())
            # cv2_prop = locals()[prop_name]
            cv2_prop = getattr(cv2, prop_name)
            # print(cv2_prop)
            prop_val = self.capture.get(cv2_prop)
            # cv2

            if prop_val not in self.invalid_values:
                length = len("CAP_PROP_")
                txt += "{}={} | ".format(prop_name[length:], prop_val)
        print(txt)

        # print(prop)

    def add_set_prop(self, prop_key, value, prop_list=[]):
        prop_list.append([self.cv2_dict_name[prop_key], prop_key, value])
        return prop_list

    def set_property(self, prop_key, value):
        prop_list = self.add_set_prop(prop_key, value)
        # print(prop_list)
        self.set_properties(prop_list)

    def set_properties(self, prop_list=[]):
        # width = 640
        # height = 480
        # width, height = [640, 480]
        # width, height = [800, 600]
        # width, height = [1024, 768]

        # prop_list
        def add_set_prop(prop_key, value):
            self.add_set_prop(prop_key, value, prop_list)

        # for q in range(8,1,-1):
        #     add_set_prop(cv2.CAP_PROP_GAMMA, q)
        # # add_set_prop(cv2.CAP_PROP_GAMMA, 5)
        # add_set_prop(cv2.CAP_PROP_GAMMA, 0.01)
        # add_set_prop(cv2.CAP_PROP_GAMMA, 42)
        # add_set_prop(cv2.CAP_PROP_GAMMA, 4)
        # add_set_prop(cv2.CAP_PROP_FPS, 2001)
        # add_set_prop(cv2.CAP_PROP_SETTINGS, 0)
        # add_set_prop(cv2.CAP_PROP_CONTRAST, 120)

        # try to write all props which value is zero
        # for prop_key in self.cv2_dict_name:
        #
        #     val = self.capture.get(prop_key)
        #     # if val == -1:
        #     if val == 2009211520:
        #         add_set_prop(prop_key, 1.2)
        # print(prop_list)

        # OPENNI_REGISTRATION = ok

        for prop_cv2_name, prop_key, prop_val in [prop for prop in prop_list]:
            # self.capture_lock.acquire()

            last_val = self.capture.get(prop_key)
            self.capture.set(prop_key, prop_val)
            read_val = self.capture.get(prop_key)

            # self.capture_lock.release()

            if last_val != read_val:
                print(
                    "Source[{}] Property set: {} = {} (before = {})".format(
                        self.source_id, prop_cv2_name, prop_val, last_val
                    )
                )
            else:
                print(
                    "Source[{}] Property could not be set! {} = {} (wanted= {})".format(
                        self.source_id, prop_cv2_name, read_val, prop_val
                    )
                )
        # for stream in self.streams:
        # stream.capture.set(3, width)
        # stream.capture.set(4, height)

        # stream.capture.get(cv2.CAP_PROP_APERTURE)

    def init_capture(self):
        self.capture_lock.acquire()
        self.capture = cv2.VideoCapture()
        self.capture_lock.release()

    def open_source_id(self, new_source_id):
        self.capture_lock.acquire()
        self.source_id = new_source_id
        self.capture_lock.release()
        self.open_source_id()

    def get_source_info(self):

        name = self.name

        fourcc = self.capture.get(cv2.CAP_PROP_FOURCC)
        white = self.capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
        # contrast = self.capture.get(cv2.CAP_PROP_CONTRAST)
        # contrast_def = 130

        already = ImageStreamControl.already_selected
        print(already)

        rnd = "round"
        blu = "blue"

        blk = "black"
        gra = "gray"
        cli = "clips"
        # # con_blk = 129
        # # con_gra = 128
        # # con_cli = 127
        # indistinguishable = [
        #         ['clips', 129],
        #         ['gray', 128],
        #         ['black', 127]
        #         ]
        # con_val = [cam[1] for cam in indistinguishable]
        #
        # def set_contrast(contrast_val):
        #     self.set_property(cv2.CAP_PROP_CONTRAST, contrast_val)
        #
        # def is_defined(name):
        #     return any(name in w for w in already)
        # # print(fourcc, white, contrast)
        # print(white, fourcc)
        if fourcc in [1196444237, 844715353.0]:
            if white != 2009211520.0:
                name = rnd
                # round -> potlacit blikani 50Hz
        if fourcc == 844715353.0:
            # round, clips, gray, black
            if white == 2009211520.0:
                # clips, gray, black
                if self.source_id == 1:
                    name = blk
                elif self.source_id == 2:
                    name = gra
                elif self.source_id == 3:
                    name = cli

        elif fourcc == -466162819:
            # blue
            name = blu

        ImageStreamControl.already_selected.append(name)

        if name == rnd:
            self.focal *= 1
        # prop = CAP_PROP_
        # self.capture.get()

        self.name = name
        # if name == rnd:
        #     self.set_property(cv2.CAP_PROP_SETTINGS, 1) # open settings on rnd cam

        folder = r"D:\DEV\PYTHON\pyCV\calibration\_pics"

        def load_matrix(folder, file):
            path = os.path.join(folder, name, file)
            try:
                mat = np.loadtxt(path)
            except Exception as exc:
                print(exc)
                mat = None
            return np.array(mat)

        mtx = load_matrix(folder, "Intrinsic.txt")
        dist = load_matrix(folder, "Distortion.txt")
        # print('Intrinsic: {}\nDistortion: {}'.format(mtx,dist))

        self.intrinsic = mtx
        self.distortion = dist

        h0 = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w0 = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_resolution = (h0, w0)
        # self.focal = (self.intrinsic[0][0] + self.intrinsic[1][1]) / 2
        # print(self.focal)
        pass

    def open_capture(self):
        self.capture_lock.acquire()
        self.capture.open(self.source_id)
        if not self.capture.isOpened():
            print("Source[", self.source_id, "] Cannot open capture.")
            return False

        print("Source[{}] Opened capture.".format(self.source_id))
        self.get_source_info()
        print("Source[{}] Renamed to {}.".format(self.source_id, self.name))
        # print('Source[', self.source_id, '] renamed to {}.')
        self.capture_lock.release()
        return True

    def toggle_source_id(self):
        self.capture_lock.acquire()
        # self.source_id = np.mod(self.source_id+1, 2)
        # self.open_capture()
        try_next = True
        while try_next:
            self.source_id += 1
            self.capture.open(self.source_id)

            if not self.capture.isOpened():
                print("Source[", self.source_id, "] Cannot open capture.")
                self.source_id = -1
                continue

            ret, frame = self.capture.read()
            if ret is False:
                print("Source[", self.source_id, "] Cannot be read from")
                self.source_id = -1
                continue
            print("Source[", self.source_id, "] Opened capture")
            try_next = False
        self.capture_lock.release()

    def close_capture(self):
        # self.capture_lock.acquire()
        self.capture.release()
        print("Source[", self.source_id, "] Released capture.")
        # self.capture_lock.release()

    def start_capturing(self, blocking=True):
        if not blocking:
            if not self.open_capture():
                return False
            self.capturing = True
            self.thread = threading.Thread(target=self.capture_loop)
            self.thread.start()
            return True
        else:
            return self.start_capturing_blocking()

    def start_capturing_blocking(self, min_height=50, iterations=9999999):
        if not self.start_capturing(blocking=False):
            return False

        print(
            "Source[",
            self.source_id,
            "] Captured frame with dimensions",
            self.frame.shape,
            ". Waiting until the height is greater than",
            min_height,
            "px",
        )
        looping = iterations + 1

        while looping > 1:
            if self.frame is not None:
                if self.frame.shape[0] < min_height:
                    pass
                else:
                    print(
                        "Source[",
                        self.source_id,
                        "] Captured frame with dimensions",
                        self.frame.shape,
                        ". Continuing with program execution.",
                    )
                    return True
            looping -= 1
        print(
            "Source[",
            self.source_id,
            "] Did not captured frame with greater height than",
            min_height,
            "px in ",
            iterations,
            "iterations.",
        )
        return False

    def stop_capturing(self):
        print("Source[", self.source_id, "] Stopped capturing.")
        self.capturing = False

    def toggle_capturing(self):
        if not self.capturing:
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
        if not self.capture.isOpened():
            # self.open_capture()
            print(
                "Source[",
                self.source_id,
                "] Cannot read frame as the capture is not opened",
            )
            self.capture_lock.release()
            return False
        else:
            ret, frame = self.capture.read()
            self.capture_lock.release()
            self.frame = frame
            return True
        # print(self.frame.shape)

# class StreamInfo():
#     def __init__(self, stream):
#         self.stream = stream
#         self.resolution = stream.frame.shape
#         self.source_id =
