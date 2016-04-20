from thread_controls import ImageStreamControl #, ChainControl, Chain

class CaptureControl():
    streams = []
    stream_id = 0
    def __init__(self):

        # ids = [0,1,2,3,4]
        ids = [0] # latest connection
        # 0 hd evolve
        # 1 webcam black
        # 2 webcam clips
        # 3 usbcam blue

        [self.streams.append(ImageStreamControl(id)) for id in ids]
        # self.streams.append(ImageStreamControl(0)) # over
        # self.streams.append(ImageStreamControl(1)) # clips
        # self.streams.append(ImageStreamControl(2)) # high def

        # self.streams.append(ImageStreamControl(3))
        # self.streams[-1].toggle_source_id()

        # print(self.streams)
        self.activate_stream()

        # time.sleep(10)
        self.set_all_settings()

    # def init_all_streams(self):
    #     # for stream in self.stre
    #     self.set_all_settings()

    def set_all_settings(self):
        # width = 640
        # height = 480
        width, height = [640, 480]
        width, height = [800, 600]
        width, height = [1024, 768]
        for stream in self.streams:
            stream.capture.set(3, width)
            stream.capture.set(4, height)

    def start_all_capturing(self):
        for image_stream_control in self.streams:
            if image_stream_control.start_capturing_blocking() == False:
                image_stream_control.stop_capturing()

    def on_stop(self):
        for image_stream_control in self.streams:
            image_stream_control.on_stop()

    def activate_stream(self):
        self.image_stream_control = self.streams[self.stream_id]

    def switch_camera(self):
        self.stream_id += 1
        if self.stream_id >= len(self.streams):
            self.stream_id = 0
        self.activate_stream()

    def toggle_capturing(self):
        for image_stream_control in self.streams:
            image_stream_control.toggle_capturing()
