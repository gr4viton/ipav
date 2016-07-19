
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.properties import ObjectProperty, StringProperty
import os.path

from kivy.core.text import Label as CoreLabel

class Fonter():
    fonts_paths = CoreLabel.get_system_fonts_dir()
    # fonts_path = fonts_paths[1] + '\\'
    fonts_path = fonts_paths[0] + '\\'

class ButtonLeft(Button):
    pass

class ChangeChainWidget(Popup):

    chain_string = StringProperty('')
    chain_string_input = ObjectProperty()
    layout_available_steps = ObjectProperty()
    chain_history_layout = ObjectProperty()

    chain_delimiter = ','
    history_file_path = 'chain_history.txt'
    default_chain_history = 'original'
    chain_history = None
    def __init__(self, new_chain_string, update_chain_string_from_popup,
                 available_steps_dict,  **kwargs):

        super(ChangeChainWidget, self).__init__(**kwargs)
        # a = unichr()


        self.chain_string = new_chain_string
        self.update_chain_string_from_popup = update_chain_string_from_popup

        self.create_available_step_widgets(available_steps_dict)
        self.where = 'end'

        self.load_chain_history()

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

    def add_step_button(self, step_name):
        self.layout_available_steps.add_widget(
            ButtonLeft(text = step_name,
                       width = 12*len(step_name),
                       on_press = lambda step_name: self.add_step(step_name=step_name.text)))

    def create_available_step_widgets(self, available_steps_dict, whatever=None):
        available_steps_list = [key for key in available_steps_dict.keys()]

        # sort the list by opened_ids

        sorted_available_steps_list = \
            sorted(available_steps_list,
                   key=lambda step: available_steps_dict[step].id)

        [self.add_step_button( step_name
            # '='.join([step_name, str(available_steps_dict[step_name].id)])
        ) for step_name in sorted_available_steps_list
         if available_steps_dict[step_name].origin == None]

        # self.available_step_string = '\n'.join(available_steps_list)
        # print(self.available_step_string)
        # text_input = TextInput(text=self.available_step_string, readonly=True)
        # popup = Popup(title='Available step names', content=text_input)
        # popup.open()

    def update_chain_string(self, whatever=None):
        print(whatever)
        self.update_chain_string_from_popup()
        self.update_chain_history(self.chain_string_input.text)

    def use_chain(self, new_chain, use=True):
        self.chain_string_input.text = new_chain
        if use:
            self.update_chain_string()

    def on_text_change(self, instance, value):
        # print('The widget', instance, 'have:', value)
        pass

    def load_chain_history(self):
        if os.path.isfile(self.history_file_path):
            with open(self.history_file_path, 'r') as f:
                self.chain_history_text = f.read()
            self.chain_history = self.chain_history_text.split('\n')
            for chain_string in self.chain_history:
                self.chain_history_layout.add_widget(
                    ChainHistory(text=chain_string, use_chain=self.use_chain))
        else:
            with open(self.history_file_path, 'w+') as f:
                f.write(self.default_chain_history)


# save history!!
# after click load of chain!

    def save_chain_history(self):
        if os.path.isfile(self.history_file_path):
            with open(self.history_file_path, 'w') as f:
                f.write('\n'.join(self.chain_history))
        # else:
        #     with open(self.history_file_path, 'w+') as f:
        #         f.write(self.default_chain_history)
        #         f.write(self.chain_history_text)

    def update_chain_history(self, chain_string, force_add=False):
        print('Chain history = ', self.chain_history)
        print('last = ', self.chain_history[-1])
        if chain_string is not self.chain_history[-1]:
            self.chain_history.append(chain_string)
            self.chain_history_layout.add_widget(
                ChainHistory(text=chain_string, use_chain=self.use_chain))

class UnicodeButton(Button, Fonter):
    # unifont = str(Fonter.fonts_path) + 'SourceCodePro-Regular'
    unifont = str(Fonter.fonts_path) + 'calibrib'
    # unifont = str(Fonter.fonts_path) + 'Carlito-Bold'
    # unifont = str(Fonter.fonts_path) + 'DejaVuSans'

class ChainHistory(GridLayout, Fonter):
    text = StringProperty()

    def __init__(self, text, use_chain, **kwargs):
        self.text = text
        super(ChainHistory, self).__init__(**kwargs)
        self.use_chain = use_chain
