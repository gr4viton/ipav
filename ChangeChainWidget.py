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

# from kivy.modules import inspector

from StepData import *

class ChangeChainWidget(Popup):

    chain_string = StringProperty('')
    chain_string_input = ObjectProperty()
    layout_available_steps = ObjectProperty()

    chain_delimiter = ','
    def __init__(self, new_chain_string, update_chain_string_from_popup,
                 available_steps_dict,  **kwargs):

        super(ChangeChainWidget, self).__init__(**kwargs)

        self.chain_string = new_chain_string
        self.update_chain_string_from_popup = update_chain_string_from_popup

        self.create_available_step_widgets(available_steps_dict)
        self.where = 'end'

    # def on_dismiss(self, **kwargs):
    #     print('dismissing')
    #     super(ChangeChainWidget, self).on_dismiss(**kwargs)

    def open(self, *largs):
        super(Popup, self).open(*largs)
        self.chain_string_input.focus = True

    def add_step(self, step_name, where=''):
        if where == '':
            where = self.where

        if where=='end':
            # self.add_step_string(str(step_name))
            self.chain_string_input.text += self.chain_delimiter + ' ' + str(step_name)

    def add_step_widget(self, step_name):
        self.layout_available_steps.add_widget(
            Button(text=step_name, on_press=lambda step_name: self.add_step(step_name=step_name.text)))

    def create_available_step_widgets(self, available_steps_dict, whatever=None):
        available_steps_list = [key for key in available_steps_dict.keys()]

        [self.add_step_widget(step_name) for step_name in available_steps_list]
        # self.available_step_string = '\n'.join(available_steps_list)
        # print(self.available_step_string)
        # text_input = TextInput(text=self.available_step_string, readonly=True)
        # popup = Popup(title='Available step names', content=text_input)
        # popup.open()


    def update_chain_string(self, whatever=None):
        print(whatever)
        self.update_chain_string_from_popup()

    def on_text_change(self, instance, value):
        # print('The widget', instance, 'have:', value)
        pass

