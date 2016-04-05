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
from StepData import *
import findHomeography as fh

class StepWidget(GridLayout):

    name = StringProperty()
    drawing = ObjectProperty('down')
    kivy_image = ObjectProperty()

    info_label_bottom = ObjectProperty()
    info_label_right = ObjectProperty()
    time_label = ObjectProperty()

    tbt_narrow = ObjectProperty()
    tbt_show_info = ObjectProperty()
    tbt_informing = ObjectProperty()
    tbt_show_img = ObjectProperty()
    tbt_draw = ObjectProperty()

    # layout_steps_height = NumericProperty(1600)

    elsewhere = -6666


    def __init__(self, **kwargs):
        super(StepWidget, self).__init__(**kwargs)
        # self.layout_steps = kwargs['parent']
        self.name = ''
        self.drawing = True
        init_shape = (0, 0)
        self.texture = Texture.create(size = init_shape, colorfmt='bgr')
        self.texture_shape = init_shape
        self.name = 'default name'
        self.info_label_position = 'b'
        self.info_label_hide('all')

        self.informing = True
        self.narrowed = False
        self.kivy_image_y = self.kivy_image.y

        # self.kivy_image = ImageButton(self.toggle_drawing)
        # self.kivy_image = ImageButton()
        # self.add_widget(self.kivy_image)

    def recreate_texture(self, cv_image):
        self.texture_shape = cv_image.shape
        self.texture = Texture.create(
            size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
        self.update_texture(cv_image)

    def recreate_widget(self, cv_image, name, info_position='b'):
        self.recreate_texture(cv_image)
        self.recreate_info_label()
        self.name = name
        self.info_label_position = info_position
        print('Recreated widget:', cv_image.shape, '[px] name: [', name,'] info_pos:', info_position)

    def recreate_info_label(self):
        if self.info_label_position == 'b':
            self.info_label = self.info_label_bottom
            self.info_label_right.size_hint_x = 0
        elif self.info_label_position == 'r':
            self.info_label = self.info_label_right

    def update_widget(self, step):
        if not self.narrowed:
            self.time_label.text = step.str_mean_execution_time('')
            if self.informing:
                self.update_info_label(step)
            if self.drawing: # called only if intended to draw
                im = np.uint8(step.data_post[dd.im].copy())
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


    def info_label_show(self, which):
        if which == 'all':
            self.info_label_show('b')
            self.info_label_show('r')
            self.info_showed = True
            self.tbt_show_info.state = 'down'
        elif which == 'current':
            # print('showin\' current', self.info_label_position)
            self.info_label_show(self.info_label_position)
        elif which == 'b':
            # print('showin\' current')
            self.info_label_bottom.y = self.info_label_bottom_y
            self.info_label_bottom.size_hint_y = 0.2
        elif which == 'r':
            self.info_label_right.y = self.info_label_right_y
            self.info_label_bottom.size_hint_x = 0.3


    def info_label_hide(self, which):
        if which == 'all':
            # self.info_label_right.size_hint_x = 0
            self.info_label_hide('b')
            self.info_label_hide('r')
            self.info_showed = False
            self.tbt_show_info.state = 'normal'
        elif which == 'current':
            self.info_label_hide(self.info_label_position)
        elif which == 'b':
            self.info_label_bottom_y = self.info_label_bottom.y
            self.info_label_bottom.y = self.elsewhere
            self.info_label_bottom.size_hint_y = 0
        elif which == 'r':
            self.info_label_right_y = self.info_label_right.y
            self.info_label_right.y = self.elsewhere
            self.info_label_right.size_hint_y = 0
    def sync_info(self, value):
        self.informing = value


    def set_narrow(self, value):
        self.narrowed = not value
        if value == True:
            self.tbt_narrow.state = 'down'
            # self.size_hint_x = 0.33
            # self.width = 282
            self.width /= 3
        if value == False:
            self.tbt_narrow.state = 'normal'
            # self.size_hint_x = 0.33/9
            self.width = 282/3
            self.width *= 3

    def show_img(self, value):
        self.img_showed = value
        if value == True:
            self.tbt_show_img.state = 'down'
            if self.info_showed == True:
                b = self.info_label_bottom.height
            else:
                b = 0
            self.kivy_image.y = self.kivy_image_y + b


        if value == False:
            self.tbt_show_img.state = 'normal'
            # self.kivy_image_y = self.kivy_image.y
            self.kivy_image.y = self.elsewhere
            # self.kivy_image.size_hint_y = 0

    def toggle_show_img(self, whatever=None):
        if self.tbt_show_img.state == 'down':
            self.show_img(False)
        else:
            self.show_img(True)



    def show_info(self, value):
        self.info_showed = value
        if value == True:
            # self.tbt_show_img.state = 'down'
            self.info_label_show('current')
        if value == False:
            # self.tbt_show_img.state = 'normal'
            self.info_label_hide('current')

    def set_drawing(self, value):
        self.drawing = value
        if value == True:
            self.tbt_draw.state = 'down'
        if value == False:
            self.tbt_draw.state = 'normal'

    def toggle_drawing(self):
        if self.tbt_draw.state == 'down':
            self.set_drawing(False)
        else:
            self.set_drawing(True)



class StepWidgetControl():

    def __init__(self, layout_steps):
        self.layout_steps = layout_steps


    def do_all_steps(self, fnc, subset):
        if subset == 'all':
            [fnc(widget, True) for widget in self.layout_steps.children]
            # [widget.fnc(True) for widget in self.layout_steps.children]
        if subset == 'none':
            [fnc(widget, False) for widget in self.layout_steps.children]

    def draw(self, subset):
        self.do_all_steps(StepWidget.set_drawing, subset)

    def show(self, subset):
        self.do_all_steps(StepWidget.show_img, subset)

    def narrow(self, subset):
        self.do_all_steps(StepWidget.set_narrow, subset)

    def info(self, subset):
        self.do_all_steps(StepWidget.show_info, subset)


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

        [widget.recreate_widget(np.uint8(step.data_post[dd.im]), step.name)
         for (widget, step) in zip(self.layout_steps.children, step_control.steps)]

    def update_layout_steps(self, step_control):

        if step_control is not None:
            if len(step_control.steps) != len(self.layout_steps.children):
                self.layout_steps_add_widgets(step_control)
            else:
                [widget.update_widget(step)
                 for (step, widget)
                 in zip(step_control.steps, self.layout_steps.children)]
