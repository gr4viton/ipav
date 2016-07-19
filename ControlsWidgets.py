
from kivy.uix.gridlayout import GridLayout

from kivy.properties import ObjectProperty, BoundedNumericProperty

from StepEnum import DataDictParameterNames as dd


class DetectColorControls(GridLayout):
    # selected_color = StringProperty('green')
    selected_color = 'green'

    def select_color(self, selected_color):
        self.selected_color = selected_color

    def get_control_values(self, data):
        data[dd.color_name] = self.selected_color
        # print('selected_color= ', self.selected_color)
        return data

class ResolutionControls(GridLayout):
    slider_value = BoundedNumericProperty(0.5, min=0.05, max=2)
    slider = ObjectProperty()

    def get_control_values(self, data):
        # val = self.slider_value
        val = self.slider.value
        data[dd.fxfy] = [val, val]
        # print('getting data = ', val)
        return data