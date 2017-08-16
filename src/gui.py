from gui_variables import GuiVariable
from Tkinter import Button, Label, PhotoImage
import tkFileDialog as filedialog
# from tensorflow.contrib.keras import models
from keras import models
import numpy as np
import h5py as h5
import os
import project_utils as pu
from PIL import Image
from PIL import ImageTk
import time


def reduce_float_length(a_list, decimals):
    for i in range(len(a_list)):
        a_list[i] = float(format(a_list[i], decimals))
    return a_list


def set_run_possible(gv):
    if gv.is_model_loaded and gv.is_test_loaded:
        if gv.run_button['state'] == 'disabled':
            gv.run_button.config(state='active')
            gv.step_button.config(state='active')
            gv.reset_button.config(state='active')
    else:
        pass


def initialize(b):
    b.loaded_model_text = Label(b.root, text='No model loaded')
    b.loaded_test_text = Label(b.root, text='No data loaded')
    b.bb_image_left = Label(b.root, image=b.tk_bb_image_1)
    b.bb_image_left.image = b.tk_bb_image_1
    b.bb_image_right = Label(b.root, image=b.tk_bb_image_2)
    b.bb_image_right.image = b.tk_bb_image_2

    b.run_button = Button(b.root, text='Run', state='disabled', command=lambda: run(b))
    b.stop_button = Button(b.root, text='Stop', command=lambda: stop(b))
    b.step_button = Button(b.root, text='Step', state='disabled', command=lambda: step(b))
    b.reset_button = Button(b.root, text='Reset', state='disabled', command=lambda: reset(b))

    b.certainty_text = Label(b.root, text='1.00')
    b.prediction_text = Label(b.root, text='Predict:    Match')
    b.load_model_button = Button(b.root, text='Load Model', command=lambda: load_model(b))
    b.load_test_button = Button(b.root, text='Load Data', command=lambda: load_test(b))

    b.truth_text = Label(b.root, text='Truth:   Match')


def load_model(gv):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_path = filedialog.askopenfilename(initialdir='../model_weights')
    # model_path = filedialog.askopenfilename(initialdir='/home/gabi/PycharmProjects/testhings/models_weights')
    model_name = model_path.strip().split('/')[-1]
    gv.model = models.load_model(model_path)
    gv.loaded_model_text.config(text='Model loaded: %s' % (model_name))
    gv.is_model_loaded = True
    set_run_possible(gv)


def get_dataset(name):
    bare_name = name.strip().split('_')[0]
    print('bare name: %s' % bare_name)
    if bare_name == 'viper':
        dataset = h5.File('../data/VIPER/viper.h5', 'r')
    elif bare_name == 'cuhk02':
        dataset = h5.File('../data/CUHK02/cuhk02.h5', 'r')
    elif bare_name == 'market':
        dataset = h5.File('../data/market/market.h5', 'r')
    elif bare_name == 'grid':
        dataset = h5.File('../data/GRID/grid.h5', 'r')
    elif bare_name == 'prid450':
        dataset = h5.File('../data/prid450/prid450.h5', 'r')
    elif bare_name == 'caviar':
        dataset = h5.File('../data/caviar/caviar.h5', 'r')
    else:
        print('something is wrong. Dataset not found')
        dataset = None
    return dataset


def load_test(gv):
    test_path = filedialog.askopenfilename(initialdir='../ranking_files')
    # test_path = filedialog.askopenfilename(initialdir='/home/gabi/PycharmProjects/testhings/viper')
    test_name = test_path.strip().split('/')[-1]
    dataset = get_dataset(test_name)

    gv.test, gv.all_truth = pu.get_data(test_path, dataset, gv.number_of_data)

    print(np.shape(gv.test))

    if gv.number_of_data > len(list(np.genfromtxt(test_path, dtype=str))):
        gv.max_position = len(list(np.genfromtxt(test_path, dtype=str)))
    else:
        gv.max_position = gv.number_of_data

    gv.loaded_test_text.config(text='Test set loaded: %s' % (test_name))
    gv.is_test_loaded = True
    set_run_possible(gv)


def do_prediction_1(gv):
    pil_image = Image.fromarray(np.uint8(gv.test[gv.step_position, 0]), 'RGB')
    image_left = ImageTk.PhotoImage(pil_image)
    gv.tk_bb_image_1 = image_left
    gv.bb_image_left.config(image=gv.tk_bb_image_1)

    pil_image = Image.fromarray(np.uint8(gv.test[gv.step_position, 1]), 'RGB')
    image_right = ImageTk.PhotoImage(pil_image)
    gv.tk_bb_image_2 = image_right
    gv.bb_image_right.config(image=gv.tk_bb_image_2)

    test_pair = gv.test[gv.step_position]
    test_pair = np.array([test_pair])

    # For streaming data, edit here
    # test_pair = np.array(np.shape(1, 2, 128, 64, 3))
    # test_pair[0] = some image of 128x64x3
    # test_pair[1] = some image of 128x64x3
    # for multiple models
    # for model in models:
    #     predictions.append(model.predict(test))


    prediction = gv.model.predict([test_pair[:, 0], test_pair[:, 1]])
    prediction = reduce_float_length([prediction[0][1]], '.2f')
    prediction = prediction[0]
    gv.certainty_text.config(text=str(prediction))
    if prediction >= 0.5:
        gv.prediction_text.config(text='Predict:    Match')
    else:
        gv.prediction_text.config(text='Predict:    Mismatch')

    truth = gv.all_truth[gv.step_position]
    print(truth)
    if truth[1] >= 0.5:
        gv.truth_text.config(text='Truth:    Match')
    else:
        gv.truth_text.config(text='Truth:    Mismatch')

    if truth[1] >= 0.5 and prediction >= 0.5:
        gv.truth_text.config(fg='green')
        gv.prediction_text.config(fg='green')
    elif truth[1] < 0.5 and prediction < 0.5:
        gv.truth_text.config(fg='green')
        gv.prediction_text.config(fg='green')
    else:
        stop(gv)
        gv.truth_text.config(fg='red')
        gv.prediction_text.config(fg='red')
    gv.step_position += 1
    print('step position: %s' % gv.step_position)

    if gv.step_position == gv.max_position:
        gv.step_position = 0


def do_prediction_2(gv):
    do_prediction_1(gv)

    if gv.stop_press:
        gv.root.after(gv.run_speed, lambda: do_prediction_2(gv))


def step(gv):
    do_prediction_1(gv)
    print('step')


def run(gv):
    print('run')
    gv.stop_press = True

    if gv.load_model_button['state'] == 'active':
        gv.load_model_button.config(state='disabled')
    if gv.load_test_button['state'] == 'active':
        gv.load_test_button.config(state='disabled')

    do_prediction_2(gv)


def stop(gv):
    print('stop')
    gv.stop_press = False


def reset(gv):
    gv.run_button.config(text='Run', state='disabled')
    gv.loaded_model_text.config(text='No model loaded')
    gv.is_model_loaded = False
    gv.loaded_test_text.config(text='No data loaded')
    gv.is_test_loaded = False
    gv.step_button.config(state='disabled')
    gv.load_model_button.config(state='active')
    gv.load_test_button.config(state='active')

    # set images back to original
    gv.tk_bb_image_1 = PhotoImage(file='../images/bb-3.png')
    gv.bb_image_left.config(image=gv.tk_bb_image_1)

    gv.tk_bb_image_2 = PhotoImage(file='../images/bb-2.png')
    gv.bb_image_right.config(image=gv.tk_bb_image_2)

    # set step_position back to zero
    gv.step_position = 0

    # set max_position back to zero
    gv.max_position = 0

    gv.certainty_text.config(text='1.00')
    gv.prediction_text.config(text='Predict:    Match', fg='black')

    gv.truth_text.config(text='Truth:   Match', fg='black')

    gv.reset_button.config(state='disabled')
    print('reset')


def create_window(gv):
    """ Creates a window and implements basic things
    """
    # set title and size and position
    gv.root.title(gv.window_title)
    gv.root.geometry('%dx%d+%d+%d' % (gv.window_width, gv.window_width, gv.window_position, gv.window_position))

    # make button for loading model
    # load_model_button = gv.load_model_button.grid(row=0, column=2)
    load_model_button = gv.load_model_button.place(x=208, y=20)

    # make text field indicating the selected model
    # model_text = gv.loaded_model_text.grid(row=1, column=2)
    model_text = gv.loaded_model_text.place(x=208, y=50)
    # make button for loading test
    # load_test_button = gv.load_test_button.grid(row=2, column=2)
    load_test_button = gv.load_test_button.place(x=208, y=70)
    # make button for selecting test set
    # load_test_text = gv.loaded_test_text.grid(row=3, column=2)
    load_test_text = gv.loaded_test_text.place(x=208, y=100)
    # create run button
    # run_button = gv.run_button.grid(row=4, column=2)
    run_button = gv.run_button.place(x=208, y=130)
    # create stop button
    stop_button = gv.stop_button.place(x=208, y=160)
    # create step button
    # step_button = gv.step_button.grid(row=5, column=2)
    step_button = gv.step_button.place(x=208, y=190)
    # create reset button
    # reset_button = gv.reset_button.grid(row=6, column=2)
    reset_button = gv.reset_button.place(x=208, y=220)
    # create dummy images
    # image_left = gv.bb_image_left.grid(row=0, column=0)
    image_left = gv.bb_image_left.place(x=20, y=20)
    # image_right = gv.bb_image_right.grid(row=0, column=1)
    image_right = gv.bb_image_right.place(x=104, y=20)
    # create certainty label
    # prediction_text = gv.certainty_text.grid(row=6, column=0)
    certainty_text = gv.certainty_text.place(x=20, y=178)
    # create prediction label
    # prediction_text = gv.prediction_text.grid(row=6, column=1)
    prediction_text = gv.prediction_text.place(x=20, y=198)
    # the ground truth if file contains labels
    truth_text = gv.truth_text.place(x=20, y=218)

    # return gv.root


def main():
    b = GuiVariable()
    initialize(b)
    # root = create_window(b)

    create_window(b)
    root = b.root

    root.mainloop()
    print(b.pred)


main()