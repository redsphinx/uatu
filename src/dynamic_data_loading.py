import numpy as np
import project_constants as pc
from PIL import Image
import os
import random as rd
from scipy import ndimage
from shutil import copyfile
import shutil
from itertools import combinations
import random
import csv
import time
import keras


def analyze_data_set(dataset):
    data_list = list(csv.reader(np.genfromtxt(dataset, dtype=None)))
    labels = np.asarray([data_list[row][2] for row in range(0, len(data_list))], dtype=int)
    positives_percentage = np.sum(labels) * 1.0 / len(labels)
    negatives_percentage = 1.0 - positives_percentage
    return [positives_percentage, negatives_percentage]


# dataset is a list with ['path_to_image.png,0'] format
def make_specific_balanced_set(dataset, positives_percentage, set_size):
    data_list = np.asarray(dataset)
    labels = np.asarray([dataset[row].split(',')[2] for row in range(0, len(dataset))])
    num_of_positives = np.floor(positives_percentage * set_size).astype(int)
    balanced_data = []
    new_data_list = []
    count_pos = 0
    count_neg = 0
    for row in range(0, len(data_list)):
        if labels[row] == '1' and count_pos < num_of_positives:
            balanced_data.append(dataset[row])
            count_pos += 1
        elif labels[row] == '0' and count_neg < set_size - num_of_positives:
            balanced_data.append(dataset[row])
            count_neg += 1
        else:
            new_data_list.append(dataset[row])
    return balanced_data, new_data_list


def make_specific_balanced_set_given_pos_neg(dataset_pos, dataset_neg, positives_percentage, set_size):
    data_list_pos = np.asarray(dataset_pos)
    data_list_neg = np.asarray(dataset_neg)

    random.shuffle(data_list_pos)
    random.shuffle(data_list_neg)

    num_of_pos = np.floor(positives_percentage * set_size).astype(int)
    num_of_neg = set_size - num_of_pos

    balanced_data = np.zeros((set_size))
    balanced_data[0:num_of_pos] = data_list_pos[0:num_of_pos]
    balanced_data[num_of_pos:] = data_list_neg[0:num_of_neg]
    random.shuffle(balanced_data)

    new_data_list_pos = data_list_pos[num_of_pos:]
    new_data_list_neg = data_list_neg[num_of_neg:]
    return  balanced_data, new_data_list_pos, new_data_list_neg


def make_batch_queue(data_size, batch_size):
    if not data_size > batch_size:
        print('Error: train_size smaller than batch_size')
        return

    rest = data_size
    queue = []
    while rest > batch_size:
        queue.append(batch_size)
        rest = rest - batch_size
    queue.append(rest)
    return queue


'''
ASSUMPTION: there is a positive and negative list in each dataset
if there are no positives or negatives in the dataset, then merge the set with another set that contains these
'''
# todo IMPORTANT: data_list has to contain the full path to the image
def make_validation_test_list(total_data_list_pos, total_data_list_neg, val_percent=0.1, test_percent=0.1,
          val_pos_percent=0.3, test_pos_percent=0.1):

    num_pos = len(total_data_list_pos)
    num_neg = len(total_data_list_neg)

    max = np.max(num_pos, num_neg)
    balanced_total = 2 * max

    val_size = np.floor(val_percent * balanced_total).astype(int)
    test_size = np.floor(test_percent * balanced_total).astype(int)

    val_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                        total_data_list_neg, val_pos_percent, val_size)

    test_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                        total_data_list_neg, test_pos_percent, test_size)

    return val_list, test_list, total_data_list_pos, total_data_list_neg


def make_train_batches(total_data_list_pos, total_data_list_neg):
    random.shuffle(total_data_list_pos)
    random.shuffle(total_data_list_neg)
    total_len = 2*total_data_list_pos
    total_data_list_neg = total_data_list_neg[0:total_len]
    total_data_list = total_data_list_pos + total_data_list_neg
    random.shuffle(total_data_list)
    return total_data_list


def load_in_array(data_list, heads=1):
    # return data array and labels list which have been categorical
    if heads == 1:
        data_array = np.zeros(shape=(len(data_list), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
        labels = np.zeros(len(data_list))
        for image in range(0, len(data_list)):
            name = data_list[image].split(',')[0]
            data_array[image] = ndimage.imread(name)[:, :, 0:3]
            labels[image] = int(data_list[image].split(',')[1])
    else:
        # do siamese loading
        pass

    labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
    return [data_array, labels]
