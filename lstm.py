import tensorflow as tf
import numpy as np
import sys
import argparse
import os
from PIL import Image
import shutil


IMAGE_HEIGHT = 20
IMAGE_WIDTH = 10
CHANNELS = 3


LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/noise/positive/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/noise/negative/'

data_paths = [LOCATION_DATA_POSITIVE, LOCATION_DATA_NEGATIVE]


def create_random_images(data_paths, num_sequences, num_images):
    for path_name in data_paths:
        FLAG_CORRUPTED = 0
        print('checking: ' + str(path_name))

        if os.path.exists(path_name) and len(os.listdir(path_name)) == num_sequences:
            for number in range(0, num_sequences):
                if len(os.listdir(os.path.join(path_name, str(number)))) == num_images:
                    if number == num_images-1:
                        print('everything looks ok')
                else:
                    print('error in ' + str(path_name) + str(number))
                    FLAG_CORRUPTED = 1
                    break
        else:
            print('folder ' + str(path_name) + ' does not exist or it is corrupted')
            FLAG_CORRUPTED = 1

        if FLAG_CORRUPTED:
            print('removing corrupted folder ' + str(path_name) + ' and creating it again')
            shutil.rmtree(path_name)
            os.mkdir(path_name)
            print('generating images')
            for number in range(0, num_sequences):
                path = os.path.join(path_name, str(number))
                print('made: ' + str(path))
                os.mkdir(path)

                for number in range(0, num_images):
                    imarray = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) * 255
                    im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
                    file = os.path.join(path, path_name.split('/')[-2] + str(number) + '.jpg')
                    im.save(file)


def load_data():

    pass


def make_labeled_sequences():

    # shape = [None, HEIGHT, WIDTH, COLORS]
    # return [train_data, validation_data, test_data]
    pass


def main(_):
    [train_data, validation_data, test_data] = make_labeled_sequences()

    # do RNN stuff

    pass


create_random_images(data_paths, 5, 4)