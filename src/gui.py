import gui_variables as gv
from Tkinter import Button



def load_model():
    loaded_model_text = gv.loaded_model_text
    loaded_model_text.config(text='model loaded')
    run_conditions = 1
    print('load model')


def load_test():
    loaded_test_text = gv.loaded_test_text
    loaded_test_text.config(text='test loaded')
    print('load test')


def run():
    if gv.run_conditions == 2:
        print('run')


def step():
    print('step')


def stop():
    print('stop')


def create_window(root):
    """ Creates a window and implements basic things
    """
    # set title and size and position
    root.title(gv.window_title)
    root.geometry('%dx%d+%d+%d' % (gv.window_width, gv.window_width, gv.window_position, gv.window_position))

    # make button for loading model
    load_model_button = Button(root, text='Load Model', command=load_model).grid(row=0, column=2)
    # make text field indicating the selected model
    model_text = gv.loaded_model_text.grid(row=1, column=2)
    # make button for loading test
    load_test_button = Button(root, text='Load Test', command=load_test).grid(row=2, column=2)
    # make button for selecting test set
    load_test_text = gv.loaded_test_text.grid(row=3, column=2)
    # create run button
    run_button = gv.run_button.grid(row=4, column=2)
    # create step button
    step_button = gv.step_button.grid(row=5, column=2)
    # create stop button
    stop_button = gv.stop_button.grid(row=6, column=2)
    # create dummy images
    image_left = gv.bb_image_left.grid(row=0, column=0)
    image_right = gv.bb_image_right.grid(row=0, column=1)
    # create certainty label
    prediction_text = gv.certainty_text.grid(row=6, column=0)
    # create prediction label
    prediction_text = gv.prediction_text.grid(row=6, column=1)

    return root


def main():
    root = gv.root
    root = create_window(root)
    root.mainloop()


main()