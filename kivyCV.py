from kivy.app import App
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
# from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.config import Config

import cv2
import numpy as np
# import sys
import threading
import time

from findHomeography import Error as tag_error
from CaptureControl import CaptureControl
from StepWidget import StepWidgetControl
from ChangeChainWidget import ChangeChainWidget
from StepControl import *

from thread_controls import ImageStreamControl, ChainControl, Chain


def convert_to_texture(im):
    return convert_rgb_to_texture(fh.colorify(im))

def convert_rgb_to_texture(im_rgb):
    buf1 = cv2.flip(im_rgb, 0)
    buf = buf1.tostring()
    texture1 = Texture.create(size=(im_rgb.shape[1], im_rgb.shape[0]), colorfmt='bgr')
    texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture1

# class ImageButton(ButtonBehavior, Image):
#
#     # def __init__(self, toggle_drawing, **kwargs):
#         # super(Image, self).__init__(**kwargs)
#         # super(ButtonBehavior, self).__init__(**kwargs)
#         # self.toggle_drawing = toggle_drawing
#     pass



class Multicopter(GridLayout):
    gl_left = ObjectProperty()
    gl_middle = ObjectProperty()
    gl_right = ObjectProperty()
    img_webcam = ObjectProperty()
    img_tags = ObjectProperty()
    # txt_numFound = StringProperty()
    # str_num_found = StringProperty()
    sla_tags = ObjectProperty()
    layout_steps = ObjectProperty()
    # img_steps = ObjectProperty()
    label_mean_exec_time = StringProperty('??')
    label_mean_exec_time_last = StringProperty('??')
    # img_tags_background = ObjectProperty((0.08, 0.16 , 0.24))
    grid_img_tags = ObjectProperty()
    lb_webcam_resolution = StringProperty('? x ? x ?')

    tag_error_count_text = StringProperty('No tags found')

    label_chain_string_text = StringProperty('...loading...')
    # popup_chain_string_text = StringProperty('...loading...')
    chain_string = ''

    def update_chain_string_from_popup(self, whatever=None):
        # preprocessing error checking
        self.set_chain_string(self.change_chain_widget.chain_string_text)
        print('Updating')

    def set_chain_string(self, new_chain_string,  whatever=None):
        self.chain_string = new_chain_string
        self.popup_chain_string_text = self.chain_string
        self.chain_control.load_chain(self.chain_string)
        self.change_chain_widget.dismiss()
        self.label_chain_string_text = self.chain_string
        print("Chain string =[", self.chain_string, ']')

    def show_load_chain(self, whatever=None):
        # self.load_popup.open()
        self.change_chain_widget.open()

    def __init__(self, capture_control, chain_control, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)

        # self.init_load_popup()

        self.capture_control = capture_control
        self.chain_control = chain_control
        self.chain_control.add_show_load_chain(self.show_load_chain)

        self.step_widgets_control = StepWidgetControl(self.layout_steps)

        new_chain_string = 'original, gray'
        available_steps_dict = self.chain_control.get_available_steps()
        self.change_chain_widget = ChangeChainWidget(new_chain_string,
                                                     self.update_chain_string_from_popup,
                                                     available_steps_dict)

        self.set_chain_string(new_chain_string )
        self.update_chain_string_from_popup()


class multicopterApp(App):
    # frame = []
    # running_findtag = False
    title = ''
    def build(self):
        # root.bind(size=self._update_rect, pos=self._update_rect)
        h = 700
        w = 1300
        Config.set('kivy', 'show_fps', 1)
        Config.set('kivy', 'desktop', 1)
        # Config.set('kivy', 'name', 'a')

        # Config.set('graphics', 'window_state', 'maximized')
        Config.set('graphics', 'position', 'custom')
        Config.set('graphics', 'height', h)
        Config.set('graphics', 'width', w)
        Config.set('graphics', 'top', 15)
        Config.set('graphics', 'left', 4)
        Config.set('graphics', 'multisamples', 0) # to correct bug from kivy 1.9.1 - https://github.com/kivy/kivy/issues/3576


        Config.set('input', 'mouse', 'mouse,disable_multitouch')

        # Config.set('graphics', 'fullscreen', 'fake')
        # Config.set('graphics', 'fullscreen', 1)

        self.capture_control = CaptureControl()
        self.capture_control.start_all_capturing()

        # selected_chain_name = 'c2'
        # selected_chain_name = '3L'
        selected_chain_name = 'standard'

        Chain.chain_names = ['2L', '3L', 'c2', 'standard']
        Chain.tag_names = ['2L', '3L', 'c2']
        Chain.load_data_chain_names = Chain.tag_names
        current_chain = Chain(selected_chain_name, start_chain = False)


        self.chain_control = ChainControl(self.capture_control, current_chain)
        self.chain_control.start_running()


        self.tag_errors_count = {}
        [self.tag_errors_count.update({str(name): int(0)}) for name, member in tag_error.__members__.items()]

        self.root = root = Multicopter(self.capture_control, self.chain_control)
        self.build_opencv()

        # self.capture_control.toggle_source_id() # take the second input source
        return root

    def build_opencv(self):
        self.fps_redraw = 1.0/50.0
        self.fps_findtag = 1.0/50.0

        Clock.schedule_interval(self.redraw_capture, self.fps_redraw )
        print('Scheduled redraw_capture with fps = ', 1/self.fps_redraw)

        Clock.schedule_interval(self.redraw_findtag, self.fps_findtag )
        print('Scheduled redraw_findtag with fps = ', 1/self.fps_findtag)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # redraw_capture() create one texture object - dont create every time!

    # why is it so black?

    # step control to work
    # timeit individual steps and display on widgets
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def redraw_capture(self, dt):
        frames = []
        for image_stream_control in self.capture_control.streams:
            # frame = self.capture_control.frame
            frame = image_stream_control.frame
            if frame is not None:
                frames.append([frame])
                #
                # self.root.lb_webcam_resolution = str(frame.shape)
                # self.root.img_webcam.texture = convert_to_texture(frame)
                #


                self.root.lb_webcam_resolution = str(frame.shape)

        preview = fh.joinIm(frames,1)
        self.root.img_webcam.texture = convert_to_texture(preview )

        # print('redraw')
        # frame = self.capture_control.image_stream_control.frame
        # self.root.lb_webcam_resolution = str(frame.shape)
        #
        # if frame is not None:
        #     self.root.img_webcam.texture = convert_to_texture(frame)


    def set_tags_found(self, found = False):
        if(found == False):
            self.root.grid_img_tags.color = (.08, .16 , .24)
        else:
            self.root.grid_img_tags.color = (.08, .96 , .24)

    def redraw_findtag(self, dt):
        # step_control = self.chain_control.step_control
        # if step_control is not None:
        #     self.root.img_steps.texture = convert_to_texture(step_control)

        step_control = self.chain_control.step_control
        self.root.step_widgets_control.update_layout_steps(step_control)
        # self.root.update_layout_steps(step_control)


        if len(self.chain_control.execution_time) > 0:
            self.root.label_mean_exec_time = str(np.round(self.chain_control.execution_time[-1], 5) * 1000)
            self.root.label_mean_exec_time_last = str(np.round(self.chain_control.mean_execution_time, 5) * 1000)


        for key in self.tag_errors_count.keys():
            self.tag_errors_count[key] = 0

        seen_tags = self.chain_control.seen_tags
        im_list = []

        # take from optionbox
        show_tag_set = [tag_error.flawless, tag_error.no_tag_rotations_found]

        if seen_tags is not None:

            for tag in seen_tags:
                # print(tag.error)
                self.tag_errors_count[tag.error.name] = self.tag_errors_count[tag.error.name] + 1

                # set flag
                if tag.error == tag_error.flawless:
                    self.set_tags_found(True)


                if tag.error in show_tag_set:
                    # print(tag.tag_warped.shape)
                    if tag.imWarped is not None:
                        im_list.append([tag.imWarped.copy()])

                else:
                    self.set_tags_found(False)


        else:
            self.set_tags_found(False)

        self.set_tag_error_count_text()
        if im_list is not None:
            if len(im_list) > 0:

                # im_list_right = [ im for im in im_list if im is not None]

                imAllTags = fh.joinIm(im_list, 1)
                # if len(imAllTags.shape) == 2:
                #     imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
                self.root.img_tags.texture = convert_to_texture(imAllTags.copy())

        # if len(imAllTags.shape) == 2:
        #     imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)

        # print(imAllTags.shape)
        # self.root.img_tags.texture = convert_to_texture(imAllTags.copy())

    def set_tag_error_count_text(self):
        list = [str(self.tag_errors_count.get(name)) + ' = ' + str(name)
                for name, member in tag_error.__members__.items()]
        self.root.tag_error_count_text = '\n'.join(list)

        # name = tag_error.flawless.name
        # print(name)
        # print(self.tag_errors_count)
        self.root.txt_numFound.text = str(self.tag_errors_count.get(tag_error.flawless.name))

    def on_stop(self):
        print("Stopping capture")
        self.capture_control.on_stop()
        self.chain_control.on_stop()

if __name__ == '__main__':
    multicopterApp().run()
    # comment


