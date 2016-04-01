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

from findHomeography import Error as tag_error
import cv2
import numpy as np
# import sys
import threading
import time

import findHomeography as fh
from thisCV import *

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

class StepWidget(GridLayout):

    name = StringProperty()
    drawing = ObjectProperty('down')
    kivy_image = ObjectProperty()

    info_label_bottom = ObjectProperty()
    info_label_right = ObjectProperty()
    time_label = ObjectProperty()

    toggle_show_img_object = ObjectProperty()
    toggle_object = ObjectProperty()
    # layout_steps_height = NumericProperty(1600)

    otherwhere = -6666

    def __init__(self, **kwargs):
        super(StepWidget, self).__init__(**kwargs)
        # self.layout_steps = kwargs['parent']
        self.name = ''
        self.drawing = True
        init_shape = (0, 0)
        self.texture = Texture.create(size = init_shape, colorfmt='bgr')
        self.texture_shape = init_shape
        self.name = 'default name'
        self.info_position = 'b'

        self.informing = True
        self.narrowed = False

        # self.kivy_image = ImageButton(self.toggle_drawing)
        # self.kivy_image = ImageButton()
        # self.add_widget(self.kivy_image)

    def recreate_texture(self, cv_image):
        self.texture_shape = cv_image.shape
        self.texture = Texture.create(
            size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
        self.update_texture(cv_image)

    def recreate_widget(self, cv_image, name, info_position='r'):
        self.recreate_texture(cv_image)
        self.recreate_info_label()
        self.name = name
        self.info_position = info_position
        print('Recreated widget:', cv_image.shape, '[px] name: [', name,'] info_pos:', info_position)

    def recreate_info_label(self):
        if self.info_position == 'b':
            self.info_label = self.info_label_bottom
            self.info_label_right.size_hint_x = 0
        elif self.info_position == 'r':
            self.info_label = self.info_label_right

    def info_label_hide(self, which):
        if which == 'all':
            self.info_label_right.size_hint_x = 0


    def update_widget(self, step):
        if not self.narrowed:
            self.time_label.text = step.str_mean_execution_time('')
            if self.informing:
                self.update_info_label(step)
            if self.drawing: # called only if intended to draw
                im = np.uint8(step.ret.copy())
                if self.texture_shape != im.shape:
                    self.recreate_texture(im)
                else:
                    self.update_texture(im)

    def update_info_label(self, step):
        self.info_label.text = step.get_info_string()

    def update_texture(self, im):
        self.update_texture_from_rgb(fh.colorify(im))

    def update_texture_from_rgb(self, im_rgb):
        buf1 = cv2.flip(im_rgb, 0)
        buf = buf1.tostring()
        self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # print(im_rgb.shape)
        self.kivy_image.texture = self.texture

    def sync_info(self, value):
        self.informing = value


    def set_narrow(self, value):
        self.narrowed = not value
        if value == True:
            # self.toggle_object.state = 'down'
            self.size_hint_x = 0.33
        if value == False:
            # self.toggle_object.state = 'normal'
            self.size_hint_x = 0.33/9

    def show_img(self, value):
        self.img_showed = value
        if value == True:
            self.toggle_show_img_object.state = 'down'
            self.kivy_image.y = self.kivy_image_y
        if value == False:
            self.toggle_show_img_object.state = 'normal'
            self.kivy_image_y = self.kivy_image.y
            self.kivy_image.y = self.otherwhere

    def show_info(self, value):
        self.info_showed = value
        if value == True:
            # self.toggle_object.state = 'down'

            pass
        if value == False:
            # self.toggle_object.state = 'normal'
            pass

    def set_drawing(self, value):
        self.drawing = value
        if value == True:
            self.toggle_object.state = 'down'
        if value == False:
            self.toggle_object.state = 'normal'

    def toggle_show_img(self, whatever):
        if self.toggle_show_img_object.state == 'down':
            self.show_img(False)
        else:
            self.show_img(True)

    def toggle_drawing(self):
        if self.toggle_object.state == 'down':
            self.set_drawing(False)
        else:
            self.set_drawing(True)

class StepWidgetControl():

    def __init__(self, layout_steps):
        self.layout_steps = layout_steps

    def show(self, command):
        if command == 'all':
            [widget.set_drawing(True) for widget in self.layout_steps.children]
        elif command == 'none':
            [widget.set_drawing(False) for widget in self.layout_steps.children]

    def layout_steps_add_widgets(self, step_control):
        diff = len(step_control.steps) - len(self.layout_steps.children)
        if diff > 0: # create widgets
            for num in range(0, np.abs(diff)):
                self.layout_steps.add_widget(StepWidget())
                print('added widget')
        else:
            for num in range(0, np.abs(diff)):
                self.layout_steps.remove_widget( self.layout_steps.children[-1])
                print('removed widget')

        [widget.recreate_widget(np.uint8(step.ret), step.name)
         for (widget, step) in zip(self.layout_steps.children, step_control.steps)]

    def update_layout_steps(self, step_control):

        if step_control is not None:
            if len(step_control.steps) != len(self.layout_steps.children):
                self.layout_steps_add_widgets(step_control)
            else:
                [widget.update_widget(step)
                 for (step, widget)
                 in zip(step_control.steps, self.layout_steps.children)]




def rgb_to_str(rgb):
    """Returns: string representation of RGB without alpha

    Parameter rgb: the color object to display
    Precondition: rgb is an RGB object"""
    return '[ '+str(rgb[0])+', '+str(rgb[1])+', '+str(rgb[2])+' ]'

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

    chain_string = ''

    def load_and_close_popup(self, whatever):
        self.chain_string = self.chain_string_text.text
        self.chain_control.load_chain(self.chain_string)
        self.load_popup.dismiss()

    def show_step_names(self, whatever):
        available_steps_dict = self.chain_control.get_available_steps()
        available_steps_list = [key for key in available_steps_dict.keys()]
        self.available_step_string = '\n'.join(available_steps_list)

        print(self.available_step_string)
        text_input = TextInput(text=self.available_step_string, readonly=True)
        popup = Popup(title='Available step names', content=text_input)
        popup.open()

    def init_load_popup(self):

        layout = GridLayout(cols=1)
        layout.add_widget(Label(text='Insert chain step names separated by commas'))
        self.chain_string_text =TextInput(text='original, gray')
        # self.chain_string_text =TextInput(text=self.chain_string)
        layout.add_widget(self.chain_string_text )

        btn_close = Button(text='Close withouth change')
        btn_show = Button(text='Show available step names')
        btn_change = Button(text='Change chain steps and close')

        layout.add_widget(btn_show)
        layout.add_widget(btn_close)
        layout.add_widget(btn_change)

        self.load_popup = Popup(title='Change chain steps', content=layout,
                                auto_dismiss=False,
                                size_hint=(1, 0.3), )

        btn_close.bind(on_press=self.load_popup.dismiss)
        btn_show.bind(on_press=self.show_step_names)
        btn_change.bind(on_press=self.load_and_close_popup)
    #     lambda im: make_flood(im, 0)

    def show_load_chain(self):
        self.load_popup.open()


    def __init__(self, capture_control, chain_control, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)

        self.init_load_popup()

        self.capture_control = capture_control
        self.chain_control = chain_control
        self.chain_control.add_show_load_chain(self.show_load_chain)

        self.step_widgets_control = StepWidgetControl(self.layout_steps)


class CaptureControl():
    streams = []
    stream_id = 0
    def __init__(self):

        # ids = [0,1,2,3,4]
        ids = [0]
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


        Config.set('input','mouse','mouse,disable_multitouch')

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
        current_chain = Chain(selected_chain_name)


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


