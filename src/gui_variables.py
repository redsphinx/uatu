'''
Holds variables for the gui
'''
from Tkinter import Tk, Label, PhotoImage, Button


class GuiVariable(object):
    def __init__(self):
        self._root = Tk()
        self._window_title = 'uatu-0.0.0'
        self._window_width = 500
        self._window_height = 300
        self._window_position = 200
        self._run_conditions = 0
        self._tk_bb_image_1 = PhotoImage(file='../images/bb-3.png')
        self._tk_bb_image_2 = PhotoImage(file='../images/bb-2.png')
        self._loaded_model_text = None   # Label(root, text='No model loaded')
        self._loaded_test_text = None   # Label(root, text='No test loaded')
        self._bb_image_left = None   # Label(root, image=tk_bb_image)
        self._bb_image_right = None   # Label(root, image=tk_bb_image)
        self._run_button = None   # Button(root, text='Run', state='disabled')
        self._step_button = None   # Button(root, text='Step', state='disabled')
        self._reset_button = None   # Button(root, text='Stop', state='disabled')
        self._certainty_text = None   # Label(root, text='1.00')
        self._prediction_text = None   # Label(root, text='MATCH')
        self._is_model_loaded = False
        self._is_test_loaded = False
        self._load_model_button = None  # Button(gv.root, text='Load Model', command=lambda: load_model_button(gv)).grid(row=0, column=2)
        self._load_test_button = None  # Button(gv.root, text='Load Test', command=lambda: load_test_button(gv)).grid(row=2, column=2)
        self._model = None
        self._test = None
        self._step_position = 0
        self._max_position = 0
        self._stop_button = None
        self._stop_press = False
        self._pred = 0
        self._truth_text = None
        self._all_truth = None
        self._number_of_data = 100
        self._run_speed = 50


    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, value):
        self._window_title = value

    # self._window_width = 300
    @property
    def window_width(self):
        return self._window_width

    @window_width.setter
    def window_width(self, value):
        self._window_width = value

    # self._window_height = 300
    @property
    def window_height(self):
        return self._window_height

    @window_height.setter
    def window_height(self, value):
        self._window_height = value

    # self._window_position = 100
    @property
    def window_position(self):
        return self._window_position

    @window_position.setter
    def window_position(self, value):
        self._window_position = value

    # self._run_conditions = 0
    @property
    def run_conditions(self):
        return self._run_conditions

    @run_conditions.setter
    def run_conditions(self, value):
        self._run_conditions = value

    # self._tk_bb_image = PhotoImage(file='/home/gabi/PycharmProjects/uatu/images/bb.png')@property
    @property
    def tk_bb_image_1(self):
        return self._tk_bb_image_1

    @tk_bb_image_1.setter
    def tk_bb_image_1(self, value):
        self._tk_bb_image_1 = value

    @property
    def tk_bb_image_2(self):
        return self._tk_bb_image_2

    @tk_bb_image_2.setter
    def tk_bb_image_2(self, value):
        self._tk_bb_image_2 = value

    # self._loaded_model_text = Label(root, text='No model loaded')
    @property
    def loaded_model_text(self):
        return self._loaded_model_text

    @loaded_model_text.setter
    def loaded_model_text(self, value):
        self._loaded_model_text = value

    # self._loaded_test_text = Label(root, text='No test loaded')
    @property
    def loaded_test_text(self):
        return self._loaded_test_text

    @loaded_test_text.setter
    def loaded_test_text(self, value):
        self._loaded_test_text = value

    # self._bb_image_left = Label(root, image=tk_bb_image)
    @property
    def bb_image_left(self):
        return self._bb_image_left

    @bb_image_left.setter
    def bb_image_left(self, value):
        self._bb_image_left = value

    # self._bb_image_right = Label(root, image=tk_bb_image)
    @property
    def bb_image_right(self):
        return self._bb_image_right

    @bb_image_right.setter
    def bb_image_right(self, value):
        self._bb_image_right = value

    # self._run_button = Button(root, text='Run', state='disabled')
    @property
    def run_button(self):
        return self._run_button

    @run_button.setter
    def run_button(self, value):
        self._run_button = value

    # self._step_button = Button(root, text='Step', state='disabled')
    @property
    def step_button(self):
        return self._step_button

    @step_button.setter
    def step_button(self, value):
        self._step_button = value

    # self._reset_button = Button(root, text='Stop', state='disabled')
    @property
    def reset_button(self):
        return self._reset_button

    @reset_button.setter
    def reset_button(self, value):
        self._reset_button = value

    # self._certainty_text = Label(root, text='1.00')
    @property
    def certainty_text(self):
        return self._certainty_text

    @certainty_text.setter
    def certainty_text(self, value):
        self._certainty_text = value

    # self._prediction_text = Label(root, text='MATCH')    @property
    @property
    def prediction_text(self):
        return self._prediction_text

    @prediction_text.setter
    def prediction_text(self, value):
        self._prediction_text = value

    @property
    def is_model_loaded(self):
        return self._is_model_loaded

    @is_model_loaded.setter
    def is_model_loaded(self, value):
        self._is_model_loaded = value

    @property
    def is_test_loaded(self):
        return self._is_test_loaded

    @is_test_loaded.setter
    def is_test_loaded(self, value):
        self._is_test_loaded = value

    @property
    def load_model_button(self):
        return self._load_model_button

    @load_model_button.setter
    def load_model_button(self, value):
        self._load_model_button = value

    @property
    def load_test_button(self):
        return self._load_test_button

    @load_test_button.setter
    def load_test_button(self, value):
        self._load_test_button = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def step_position(self):
        return self._step_position

    @step_position.setter
    def step_position(self, value):
        self._step_position = value

    @property
    def max_position(self):
        return self._max_position

    @max_position.setter
    def max_position(self, value):
        self._max_position = value

    @property
    def stop_button(self):
        return self._stop_button

    @stop_button.setter
    def stop_button(self, value):
        self._stop_button = value

    @property
    def stop_press(self):
        return self._stop_press

    @stop_press.setter
    def stop_press(self, value):
        self._stop_press = value

    @property
    def pred(self):
        return self._pred

    @pred.setter
    def pred(self, value):
        self._pred = value

    @property
    def truth_text(self):
        return self._truth_text

    @truth_text.setter
    def truth_text(self, value):
        self._truth_text = value
        
    @property
    def all_truth(self):
        return self._all_truth

    @all_truth.setter
    def all_truth(self, value):
        self._all_truth = value
        
    @property
    def number_of_data(self):
        return self._number_of_data

    @number_of_data.setter
    def number_of_data(self, value):
        self._number_of_data = value

    @property
    def run_speed(self):
        return self._run_speed

    @run_speed.setter
    def run_speed(self, value):
        self._run_speed = value