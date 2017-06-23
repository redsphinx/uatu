'''
Holds constants for the gui
'''
from Tkinter import Tk, Label, PhotoImage, Button

class GuiVariable(object):
    def __init__(self):
        self._root = Tk()
        self._window_width = 300
        self._window_height = 300
        self._window_position = 100
        self._run_conditions = 0
        self._tk_bb_image= PhotoImage(file='/home/gabi/PycharmProjects/uatu/images/bb.png')
        self._loaded_model_text = Label(root, text='No model loaded')
        self._loaded_test_text = Label(root, text='No test loaded')
        self._bb_image_left = Label(root, image=tk_bb_image)
        self._bb_image_right = Label(root, image=tk_bb_image)
        self._run_button = Button(root, text='Run', state='disabled')
        self._step_button = Button(root, text='Step', state='disabled')
        self._stop_button = Button(root, text='Stop', state='disabled')
        self._certainty_text = Label(root, text='1.00')
        self._prediction_text = Label(root, text='MATCH')

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
    
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
        


    # self._tk_bb_image = PhotoImage(file='/home/gabi/PycharmProjects/uatu/images/bb.png')
    # self._loaded_model_text = Label(root, text='No model loaded')
    # self._loaded_test_text = Label(root, text='No test loaded')
    # self._bb_image_left = Label(root, image=tk_bb_image)
    # self._bb_image_right = Label(root, image=tk_bb_image)
    # self._run_button = Button(root, text='Run', state='disabled')
    # self._step_button = Button(root, text='Step', state='disabled')
    # self._stop_button = Button(root, text='Stop', state='disabled')
    # self._certainty_text = Label(root, text='1.00')
    # self._prediction_text = Label(root, text='MATCH')