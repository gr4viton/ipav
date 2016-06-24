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

    # info_label_bottom = ObjectProperty()
    info_label_layout = ObjectProperty()
    control_layout = ObjectProperty()
    bottom_layout = ObjectProperty()

    # info_label_right = ObjectProperty()
    time_label = ObjectProperty()

    tbt_narrow = ObjectProperty()
    tbt_show_info = ObjectProperty()
    tbt_informing = ObjectProperty()
    tbt_show_img = ObjectProperty()
    tbt_draw = ObjectProperty()


    # layout_steps_height = NumericProperty(1600)

    elsewhere = +6666


    def __init__(self, **kwargs):
        super(StepWidget, self).__init__(**kwargs)
        # self.layout_steps = kwargs['parent']
        self.name = ''
        self.drawing = True
        init_shape = (0, 0)
        self.texture = Texture.create(size = init_shape, colorfmt='bgr')
        self.texture_shape = init_shape
        self.name = 'default name'

        self.narrowed = False

        self.controls = None

        self.informing = True
        self.info_showed = False
        self.padding_y = 4


        self.kivy_image_y_normal = self.kivy_image.y + self.padding_y
        self.kivy_image_y_info = self.padding_y + 40


        # self.info_label_bottom_y = self.info_label_bottom.y
        self.info_label_bottom = None
        self.info_label_hide()


        # self.kivy_image = ImageButton(self.toggle_drawing)
        # self.kivy_image = ImageButton()
        # self.add_widget(self.kivy_image)

    def recreate_texture(self, cv_image):
        self.texture_shape = cv_image.shape
        self.texture = Texture.create(
            size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
        self.update_texture(cv_image)

    def recreate_widget(self, step):
        # cv_image, name, narrowed=False, info_position='b'
        self.step = step

        cv_image = np.uint8(step.data_post[dd.im])
        cv_image = cv_image
        # print(len(cv_image[0]))
        name = step.name
        narrowed = step.narrowed
        info_position = 'b'

        self.recreate_texture(cv_image)
        self.recreate_info_label()
        self.name = name
        print('Setting [{}].narrowed = {}'.format(name, narrowed))
        self.set_narrowed(narrowed)

        self.info_label_position = info_position

        # recreate control widgets
        # self.control_layout.clear_widgets()
        if self.controls:
            self.bottom_layout.clear_widgets(self.controls)
            print('Deleted old controls widgets')


        self.controls = self.step.controls
        if self.controls:
            self.control_layout.add_widget(self.controls)
            print('Created new controls widgets')


        print('Recreated widget:', cv_image.shape, '[px] name: [', name,
              '] info_pos:', info_position)




    def update_widget(self, step):
        if not self.narrowed:
            # print(step.name,'.narrowed = ', self.narrowed)
            self.time_label.text = step.str_mean_execution_time('')
            if step.data_post == None:
                return
            if self.informing:
                self.update_info_label(step)
            if step.new_name:
                self.name = step.new_name
            if self.drawing: # called only if intended to draw
                # do I need a copy?
                im = step.data_post[dd.im].copy()
                im_uint8 = self.convert_image(im)
                if self.texture_shape != im_uint8.shape:
                    self.recreate_texture(im_uint8)
                else:
                    self.update_texture(im_uint8)


    def convert_image(self, im_orig):
        im_type = im_orig.dtype
        if im_type != np.uint8:
            if im_type == np.float:
                im = self.polarize_image(im_orig)
            else:
                im = np.uint8(im_orig)
        else:
            im = im_orig

        return im

    def polarize_image(self, im):
        """create image with color polarization = negative values are with color1 and positive with color2"""

        if len(im.shape) == 2:
            pos = im.copy()
            pos[pos<0] = 0
            pos = np.uint8(pos)

            neg = im.copy()
            neg[neg>0] = 0
            neg = np.uint8(-neg)

            nul = np.uint8(0*im)

            b = nul
            g = pos
            r = neg
            im_out = cv2.merge((b,g,r))
        else:
            im_out = im

        return im_out



    def update_info_label(self, step):
        if self.info_label_bottom is not None:
            self.info_label_bottom.text = step.get_info_string()
            step.data_post[dd.info] = False

    def update_texture(self, im):
        self.update_texture_from_rgb(self.colorify(im))

    def update_texture_from_rgb(self, im_rgb):
        buf1 = cv2.flip(im_rgb, 0)
        buf = buf1.tostring()
        self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # print(im_rgb.shape)
        self.kivy_image.texture = self.texture


    def colorify(self, im):
        if len(im.shape) == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            return im.copy()



    def sync_info(self, value):
        self.informing = value


    def set_narrowed(self, narrowed, from_gui=False):
        # self.narrowed = not value
        self.narrowed = narrowed
        if narrowed == True:
            if from_gui==False:
                self.tbt_narrow.state = 'normal'
            # self.size_hint_x = 0.33
            # self.width = 282
            self.width = 282/3
        if narrowed == False:
            if from_gui==False:
                self.tbt_narrow.state = 'down'
            # self.size_hint_x = 0.33/9
            self.width = 282/3
            self.width *= 3

    def show_img(self, value):
        self.img_showed = value
        if value == True:
            self.tbt_show_img.state = 'down'

            if self.info_showed == True:
                self.kivy_image.y = self.kivy_image_y_info

            else:
                self.kivy_image.y = self.kivy_image_y_normal

        if value == False:
            self.tbt_show_img.state = 'normal'
            self.kivy_image.y = self.elsewhere



    def toggle_show_img(self, whatever=None):
        if self.tbt_show_img.state == 'down':
            self.show_img(False)
        else:
            self.show_img(True)

    def recreate_info_label(self):
        self.info_label_hide()

    def info_label_show(self):
        children = self.info_label_layout.children
        if children is not None:
            self.info_label_layout.add_widget(InfoLabelWidget())
            children = self.info_label_layout.children
            if len(children) > 0:
                self.info_label_bottom =children[0]
                # print(0.2*self.info_label_bottom.parrent.size)
                # self.kivy_image_y_info = self.padding_y + self.info_label_bottom.size[1]
                self.kivy_image_y_info = self.padding_y + 40
            self.info_label_layout.size_hint_y = 0.2

    def info_label_hide(self):
        children = self.info_label_layout.children
        if len(children) > 0:
            self.info_label_layout.remove_widget(children[0])
            self.info_label_layout.size_hint_y = 0
            self.info_label_bottom = None
            #
            # self.info_label_bottom.y = self.elsewhere
            # self.info_label_bottom.size_hint_y = None
            # # self.info_label_bottom.min_height = 0
            # self.info_label_bottom.height = 1000

    def show_info(self, value):
        self.info_showed = value
        if value == True:
            self.tbt_show_info.state = 'down'
            self.info_label_show()
        if value == False:
            self.tbt_show_info.state = 'normal'
            self.info_label_hide()

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


class InfoLabelWidget(TextInput):
    pass

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
        self.do_all_steps(StepWidget.set_narrowed, subset)

    def info(self, subset):
        self.do_all_steps(StepWidget.show_info, subset)


    def layout_steps_add_widgets(self, step_control):
        diff = len(step_control.steps) - len(self.layout_steps.children)
        if diff > 0: # create widgets
            for num in range(0, np.abs(diff)):
                self.layout_steps.add_widget(StepWidget())
                # print('added widget')
        else:
            for num in range(0, np.abs(diff)):
                self.layout_steps.remove_widget( self.layout_steps.children[-1])
                # print('removed widget')

        ziplist = list(zip(self.layout_steps.children, step_control.steps))

        # ziplist = ziplist[::-1]
        # print(ziplist)
        # print(ziplist[::-1])

        [widget.recreate_widget(step) for (widget, step) in ziplist
         if step.data_post is not None]

        # [widget.recreate_widget(
        #     np.uint8(step.data_post[dd.im]),
        #     step.name,
        #     narrowed=step.narrowed)
        #     for (widget, step) in ziplist if step.data_post is not None]





    def update_layout_steps(self, step_control):

        if step_control is not None:
            if len(step_control.steps) != len(self.layout_steps.children):
                self.layout_steps_add_widgets(step_control)
            else:
                ziplist = list(zip(self.layout_steps.children, step_control.steps))
                # ziplist = ziplist[::-1]
                [widget.update_widget(step) for (widget, step) in ziplist]
