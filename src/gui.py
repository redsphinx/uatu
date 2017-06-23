from gui_variables import GuiVariable
from Tkinter import Button, Label


def set_run_possible(gv):
    if gv.is_model_loaded and gv.is_test_loaded:
        if gv.run_button['state'] == 'disabled':
            gv.run_button.config(state='active', command=lambda: run(gv))
            gv.step_button.config(state='active', command=lambda: step(gv))
            gv.reset_button.config(state='active', command=lambda: reset(gv))
    else:
        pass


def initialize(b):
    b.loaded_model_text = Label(b.root, text='No model loaded')
    b.loaded_test_text = Label(b.root, text='No test loaded')
    b.bb_image_left =  Label(b.root, image=b.tk_bb_image)
    b.bb_image_right =  Label(b.root, image=b.tk_bb_image)
    b.run_button =  Button(b.root, text='Run', state='disabled')
    b.step_button =  Button(b.root, text='Step', state='disabled')
    b.reset_button =  Button(b.root, text='Reset', state='disabled')
    b.certainty_text =  Label(b.root, text='1.00')
    b.prediction_text = Label(b.root, text='Match')
    b.load_model_button = Button(b.root, text='Load Model', command=lambda: load_model(b))
    b.load_test_button = Button(b.root, text='Load Test', command=lambda: load_test(b))


def load_model(gv):
    gv.loaded_model_text.config(text='model loaded')
    gv.is_model_loaded = True
    print('load model')
    set_run_possible(gv)


def load_test(gv):
    gv.loaded_test_text.config(text='test loaded')
    gv.is_test_loaded = True
    print('load test')
    set_run_possible(gv)


def run(gv):
    print('run')

    gv.load_model_button.config(state='disabled')
    gv.load_test_button.config(state='disabled')

    if gv.run_button['text'] == 'Run':
        gv.run_button.config(text='Pause')
    elif gv.run_button['text'] == 'Pause':
        gv.run_button.config(text='Run')
    else:
        gv.run_button.config(text='???')



def step(gv):
    if gv.run_button['text'] == 'Pause':
        gv.run_button.config(text='Run')
    print('step')


def reset(gv):
    gv.run_button.config(text='Run', state='disabled')
    gv.loaded_model_text.config(text='No model loaded')
    gv.is_model_loaded = False
    gv.loaded_test_text.config(text='No test loaded')
    gv.is_test_loaded = False
    gv.step_button.config(state='disabled')
    gv.load_model_button.config(state='active')
    gv.load_test_button.config(state='active')

    gv.reset_button.config(state='disabled')

    print('reset')


def create_window(gv):
    """ Creates a window and implements basic things
    """
    # set title and size and position
    gv.root.title(gv.window_title)
    gv.root.geometry('%dx%d+%d+%d' % (gv.window_width, gv.window_width, gv.window_position, gv.window_position))

    # make button for loading model
    load_model_button = gv.load_model_button.grid(row=0, column=2)
    # make text field indicating the selected model
    model_text = gv.loaded_model_text.grid(row=1, column=2)
    # make button for loading test
    load_test_button = gv.load_test_button.grid(row=2, column=2)
    # make button for selecting test set
    load_test_text = gv.loaded_test_text.grid(row=3, column=2)
    # create run button
    run_button = gv.run_button.grid(row=4, column=2)
    # create step button
    step_button = gv.step_button.grid(row=5, column=2)
    # create reset button
    reset_button = gv.reset_button.grid(row=6, column=2)
    # create dummy images
    image_left = gv.bb_image_left.grid(row=0, column=0)
    image_right = gv.bb_image_right.grid(row=0, column=1)
    # create certainty label
    prediction_text = gv.certainty_text.grid(row=6, column=0)
    # create prediction label
    prediction_text = gv.prediction_text.grid(row=6, column=1)

    return gv.root


def main():
    b = GuiVariable()
    initialize(b)
    root = create_window(b)
    root.mainloop()


main()