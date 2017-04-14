import tensorflow as tf
import keras
from keras import backend as K

import project_constants as pc
import project_utils as pu
import cnn_clean as cnn
# import siamese_cnn_clean as scnn

import os
import numpy as np
import time


def experiment_0():
    # testing stuff
    experiment_name = 'running the CNN on 20 epochs batchsize 128'
    cnn.main(experiment_name)


def experiment_1():
    experiment_name = 'normalizing the image to be between [0, 1]'


def main():
    experiment_0()


main()