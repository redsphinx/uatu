import tensorflow as tf
import numpy as np
import sys
import argparse
import os
from PIL import Image
import shutil
import random as rd
from scipy import ndimage


IMAGE_HEIGHT = 20
IMAGE_WIDTH = 10
CHANNELS = 3


LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/noise/positive/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/noise/negative/'

data_paths = [LOCATION_DATA_POSITIVE, LOCATION_DATA_NEGATIVE]


def create_random_sequences(data_paths, num_sequences, num_images):
    # TODO set flag to update the_noise_files in load_data()
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


def create_labels(number):
    return [1]*number + [0]*number


def make_list_with_full_path(path, list):
    list_with_full_path = []
    for item in range(0, len(list)):
        list_with_full_path.append(os.path.join(path, list[item]))
    return list_with_full_path


def load_data():
    create_random_sequences(data_paths, 200, 5)
    the_noise_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noise_folder')

    if os.path.exists(the_noise_folder):
        print('loading data from files')
        train_data = np.genfromtxt(os.path.join(the_noise_folder, 'train_data.csv'))
        train_labels = np.genfromtxt(os.path.join(the_noise_folder, 'train_labels.csv'))
        validation_data = np.genfromtxt(os.path.join(the_noise_folder, 'validation_data.csv'))
        validation_labels = np.genfromtxt(os.path.join(the_noise_folder, 'validation_labels.csv'))
        testing_data = np.genfromtxt(os.path.join(the_noise_folder, 'testing_data.csv'))
        testing_labels = np.genfromtxt(os.path.join(the_noise_folder, 'testing_labels.csv'))

    else:
        print('creating files')
        os.mkdir(the_noise_folder)
        files_positive = make_list_with_full_path(LOCATION_DATA_POSITIVE, os.listdir(LOCATION_DATA_POSITIVE))
        files_negative = make_list_with_full_path(LOCATION_DATA_NEGATIVE, os.listdir(LOCATION_DATA_NEGATIVE))
        all_files = files_positive + files_negative

        total_number_of_sequences = len(all_files)
        labels = create_labels(total_number_of_sequences/2)

        # assuming all sequences are the same length
        number_of_images_in_a_sequence = len(os.listdir(os.path.join(data_paths[0], os.listdir(data_paths[0])[0])))
        data = np.zeros(shape=(total_number_of_sequences, number_of_images_in_a_sequence, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        print('shape data:' + str(np.shape(data)))

        train_percentage = 0.5
        validation_percentage = (1 - train_percentage) / 2

        everything = list(zip(all_files, labels))
        rd.shuffle(everything)
        all_files, labels = zip(*everything)

        for sequence in range(0, len(all_files)):
            for image in range(0, number_of_images_in_a_sequence):
                data[sequence][image] = ndimage.imread(os.path.join(all_files[sequence], os.listdir(all_files[sequence])[image]))

        split_here_1 = int(len(all_files) * train_percentage)
        split_here_2 = int(len(all_files) * validation_percentage)

        train_data = all_files[0:split_here_1]
        train_labels = labels[0:split_here_1]
        validation_data = all_files[split_here_1:split_here_1 + split_here_2]
        validation_labels = labels[split_here_1:split_here_1 + split_here_2]
        testing_data = all_files[split_here_2:len(all_files)]
        testing_labels = labels[split_here_2:len(all_files)]

        train_data_file = os.path.join(the_noise_folder, 'train_data.csv')
        train_labels_file =  os.path.join(the_noise_folder, 'train_labels.csv')
        validation_data_file =  os.path.join(the_noise_folder, 'validation_data.csv')
        validation_labels_file =  os.path.join(the_noise_folder, 'validation_labels.csv')
        testing_data_file = os.path.join(the_noise_folder, 'testing_data.csv')
        testing_labels_file = os.path.join(the_noise_folder, 'testing_labels.csv')

        with open(train_data_file, 'wr') as file:
            for item in range(0, len(train_data)):
                file.write(str(train_data[item]) + '\n')
        with open(train_labels_file, 'wr') as file:
            for item in range(0, len(train_labels)):
                file.write(str(train_labels[item]) + '\n')

        with open(validation_data_file, 'wr') as file:
            for item in range(0, len(validation_data)):
                file.write(str(validation_data[item]) + '\n')
        with open(validation_labels_file, 'wr') as file:
            for item in range(0, len(validation_labels)):
                file.write(str(validation_labels[item]) + '\n')

        with open(testing_data_file, 'wr') as file:
            for item in range(0, len(testing_data)):
                file.write(str(testing_data[item]) + '\n')
        with open(testing_labels_file, 'wr') as file:
            for item in range(0, len(testing_labels)):
                file.write(str(testing_labels[item]) + '\n')

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        validation_data = np.asarray(validation_data)
        validation_labels = np.asarray(validation_labels)
        testing_data = np.asarray(testing_data)
        testing_labels = np.asarray(testing_labels)

    return [train_data, train_labels, validation_data, validation_labels, testing_data, testing_labels]

def main(_):

    # do RNN stuff

    pass


load_data()