from thread_controls import ImageStreamControl #, ChainControl, Chain
import cv2

from camera_distinguish import print_usb_info

def init_cap_prop():
    prefix = 'CAP_PROP_'
    length = len(prefix)
    dir_cv2_cap_prop = [word for word in dir(cv2) if word[:length] == prefix]
    # self.dir_cv2_cap_prop = [print(word[:length]) for word in dir(cv2)]
    print(dir_cv2_cap_prop)
    return dir_cv2_cap_prop

class CaptureControl():
    streams = []
    stream_id = 0
    dir_cv2_cap_prop = init_cap_prop()

    def __init__(self):

        # self.init_cap_prop()

        # ids = [0] # latest connection
        ids = [0]
        # 0 hd evolve
        # 1 webcam black5
        # 2 webcam clips
        # 3 usbcam blue

        # FPS:
        # RED-round = 2000


        # ids = [0,1,2,3,4]
        ids = [0,1,2,3]
        # ids = [0,1]

        [self.streams.append(ImageStreamControl(id)) for id in ids]

        # self.streams.append(ImageStreamControl(0)) # over
        # self.streams.append(ImageStreamControl(1)) # clips
        # self.streams.append(ImageStreamControl(2)) # high def

        # self.streams.append(ImageStreamControl(3))
        # self.streams[-1].toggle_source_id()

        # print(self.streams)
        self.cur_index = 0
        self.activate_stream()

        # time.sleep(10)
        # self.set_all_settings()

    # def init_all_streams(self):
    #     # for stream in self.stre
    #     self.set_all_settings()

    def get_stream(self, id):
        rng = range(len(self.streams))
        for (q, stream) in zip(rng, self.streams):
        # stream_w_id = [stream for stream in self.streams if stream.source_id == id]
            if stream.source_id == id:
                self.cur_index = q
                return stream



    def get_next_stream(self):
        self.cur_index += 1
        if self.cur_index > len(self.streams):
            self.cur_index = 0
        return self.streams[self.cur_index]
        # next_stream = [stream for stream in self.streams if stream.source_id == id]
        # print(type(stream_w_id[0]))


    def start_all_capturing(self):
        for stream in self.streams:
            if stream.start_capturing_blocking() == False:
                stream.stop_capturing()
            else:
                stream.print_all_properties()
                stream.set_properties()
                # stream.print_all_properties()

        print_usb_info()


    def on_stop(self):
        for stream in self.streams:
            stream.on_stop()

    def activate_stream(self):
        self.stream = self.streams[self.stream_id]

    def switch_camera(self):
        self.stream_id += 1
        if self.stream_id >= len(self.streams):
            self.stream_id = 0
        self.activate_stream()

    def toggle_capturing(self):
        for stream in self.streams:
            stream.toggle_capturing()
