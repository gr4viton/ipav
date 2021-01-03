from kivy.app import App
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


from kivy.uix.togglebutton import ToggleButton
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.config import Config


from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout

# from kivy.uix.scrollview import ScrollView

txt = """
#:kivy 1.9

<steps_demo>:
    layout_steps: layout_steps
    # cols: 1
    GridLayout: # right tab
        cols: 1
        padding: 5
        GridLayout: # step window layout
            cols:1
            id: grid_steps
            ScrollView: # step scrolling
                id: scroll_steps

                min_move: 5
                scroll_distance: min(self.width, self.height) * 0.8
                do_scroll_x: 0

                StackLayout:
                    height: self.minimum_height - 0
                    size_hint_y: None
                    id: layout_steps
                    orientation: 'lr-tb'


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
<StepWidgetInfo>
    multiline: 'True'
    text: '2\\n3'
    size_hint: 1, None
    height: (len(self._lines)+1) * self.line_height
    on_text: self.update_parent()
#    on_enter: self.update_parent()
    on_focus: self.update_parent()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
<StepWidget>:

    bottom_layout: bottom_layout

#    orientation: 'vertical'
    cols:1

    size_hint_y: None
    size_hint_x: None

    width: 225
#    height: sum([child.height for child in self.children])
#    height: self.minimum_height
    cols: 1

    padding: (4,4,4,4)
    spacing: (4,4)
    margin: 1
    color: .08,.16 , .24
    canvas.before:
        Color:
            rgb: self.color
        Rectangle:
            pos: self.x + self.margin, self.y + self.margin + 1
            size: self.width - 2 * self.margin , root.height - 2 * self.margin

    id: bottom_layout

    GridLayout:
        cols: 2
        size_hint_y: None
        height: 0.15 * 250

        ToggleButton:
            id: tbt_narrow
            text: root.name
            size_hint_x: 0.9
            state: 'down'
#            on_state: root.set_narrowed(narrowed=(self.state != 'down'), from_gui=True)

            on_state: root.clear_widgets()


"""

# Builder.load_string(txt)


# class StepWidgetInfo(ScrollView):
class StepWidgetInfo(TextInput):
    update_height = None
    last_lines = 0

    def update_parent(self):
        if len(self._lines) != self.last_lines:
            self.last_lines = len(self._lines)
            self.height = (len(self._lines) + 1) * self.line_height
            self.update_height()


# class StepWidget(BoxLayout):
# class StepWidget(StackLayout):
# class StepWidget(AnchorLayout):
class StepWidget(GridLayout):
    name = "step"
    bottom_layout = ObjectProperty()

    def update_height(self):
        heights = [child.height for child in self.children]
        spacing = len(heights) - 1
        self.height = (
            sum(heights)
            + (spacing * self.spacing[0])
            + self.padding[0]
            + self.padding[2]
        )
        print("updating")

    def __init__(self, **kwargs):
        super(StepWidget, self).__init__(**kwargs)
        self.update_height()

    def clear_widgets(self, **kwargs):
        super(StepWidget, self).clear_widgets(**kwargs)
        self.add_widget(Button())
        self.update_height()

    pass


class steps_demo(GridLayout):
    layout_steps = ObjectProperty()
    # txx = StringProperty()

    # def callback(self, whatever=None):
    #     self.txx+='asdasd\n'
    # def update_parent(self, **kwargs):
    #     sum_height = sum([child.height for child in sw.children])
    #     self.height = sum_height

    def __init__(self, **kwargs):
        super(steps_demo, self).__init__(**kwargs)

        for i in range(6):
            # sw = StepWidget(height=250+i*10)
            sw = StepWidget(size_hint_y=None)
            for q in range(i):
                # tx = StepWidgetInfo()
                tx = StepWidgetInfo()
                tx.update_height = sw.update_height
                sw.bottom_layout.add_widget(tx)
                # tx.text = self.txx

            sw.update_height()
            self.layout_steps.add_widget(sw)
        #
        # btn = Button(text='add', on_press=callback)
        # self.layout_steps.add_widget(btn)

    pass


class steps_demoApp(App):
    # frame = []
    # running_findtag = False
    title = ""

    def build(self):
        # root.bind(size=self._update_rect, pos=self._update_rect)
        h = 700
        w = 1360
        # 1305 714
        Config.set("kivy", "show_fps", 1)
        Config.set("kivy", "desktop", 1)
        # Config.set('kivy', 'name', 'a')

        # Config.set('graphics', 'window_state', 'maximized')
        Config.set("graphics", "position", "custom")
        Config.set("graphics", "height", h)
        Config.set("graphics", "width", w)
        Config.set("graphics", "top", 15)
        Config.set("graphics", "left", 4)
        Config.set(
            "graphics", "multisamples", 0
        )  # to correct bug from kivy 1.9.1 - https://github.com/kivy/kivy/issues/3576

        Config.set("input", "mouse", "mouse,disable_multitouch")

        self.root = steps_demo()


if __name__ == "__main__":
    steps_demoApp().run()
    # comment
