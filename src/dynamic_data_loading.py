import numpy as np
import project_constants as pc
from scipy import ndimage
import random
import csv
import keras
import h5py
import time


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


    pos = data_list_pos[0:num_of_pos]
    neg = data_list_neg[0:num_of_neg]
    balanced_data = np.concatenate((pos, neg))
    random.shuffle(balanced_data)

    new_data_list_pos = data_list_pos[num_of_pos:]
    new_data_list_neg = data_list_neg[num_of_neg:]
    return  balanced_data, new_data_list_pos, new_data_list_neg

def make_slice_queue(data_size, batch_size):
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
# FIXME modify 'make_validation_test_list' to make it output pos and neg lists for val and test
# todo IMPORTANT: data_list has to contain the full path to the image
def make_validation_test_list(total_data_list_pos, total_data_list_neg, val_percent=0.001, test_percent=0.001,
          val_pos_percent=0.3, test_pos_percent=0.1):

    num_pos = len(total_data_list_pos)
    num_neg = len(total_data_list_neg)

    min_num = min(num_pos, num_neg)
    balanced_total = 2 * min_num

    val_size = np.floor(val_percent * balanced_total).astype(int)
    test_size = np.floor(test_percent * balanced_total).astype(int)

    val_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                        total_data_list_neg, val_pos_percent, val_size)

    test_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                        total_data_list_neg, test_pos_percent, test_size)

    return val_list, test_list, total_data_list_pos, total_data_list_neg


# FIXME modify 'make_train_batches' to output a pos and neg list of indices
def make_train_batches(total_data_list_pos, total_data_list_neg):
    random.shuffle(total_data_list_pos)
    random.shuffle(total_data_list_neg)
    total_len_half = min(len(total_data_list_neg), len(total_data_list_pos))
    total_data_list = np.concatenate((total_data_list_pos[0:total_len_half], total_data_list_neg[0:total_len_half]))
    random.shuffle(total_data_list)
    return total_data_list


# FIXME modify 'load_in_array' to accept 2 lists, a pos and a neg indices list. Shuffle data after loading
def load_in_array(data_list, heads=1):
    # return data array and labels list which have been categorical
    if heads == 1:
        data_array = np.zeros(shape=(len(data_list), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
        labels = np.zeros(len(data_list))
        for image in xrange(0, len(data_list)):
            name = data_list[image].split(',')[0]
            data_array[image] = ndimage.imread(name)[:, :, 0:3]
            labels[image] = int(data_list[image].split(',')[1])
    else:
        # do siamese loading
        pass

    labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
    return [data_array, labels]


def fetch_dummy_data():
    print('fetching data')
    path_pos = '/home/gabi/PycharmProjects/uatu/data/all_positives.txt'
    path_neg = '/home/gabi/PycharmProjects/uatu/data/all_negatives.txt'

    list_pos = np.genfromtxt(path_pos, dtype=None).tolist()
    list_neg = np.genfromtxt(path_neg, dtype=None).tolist()

    return list_pos, list_neg


def data_pos_to_hdf5():
    data_pos, data_neg = fetch_dummy_data()
    # amount = 40000
    h5_path = '/home/gabi/PycharmProjects/uatu/data/all_data_uncompressed.h5'

    print('loading positive data into array')
    start = time.time()
    pos, lab_pos = load_in_array(data_pos)
    time_loading_pos = time.time() - start
    print('time loading pos: %0.2f' %time_loading_pos)

    with h5py.File(h5_path, 'a') as myFile:
        print('loading pos array into hdf5')
        start = time.time()
        pos_data = myFile.create_dataset(name='positives', data=pos)
        time_loading = time.time() - start
        print('time loading: %0.2f' % time_loading)


def data_neg_to_hdf5():
    data_pos, data_neg = fetch_dummy_data()
    # amount = 40000
    h5_path = '/home/gabi/PycharmProjects/uatu/data/all_data_uncompressed.h5'

    print('loading negative data into array')
    start = time.time()
    neg, lab_neg = load_in_array(data_neg)
    time_loading_neg = time.time() - start

    print('time loading neg: %0.2f' %time_loading_neg)

    with h5py.File(h5_path, 'w') as myFile:
        print('loading neg array into hdf5')
        start = time.time()
        neg_data = myFile.create_dataset(name='negatives', data=neg)
        time_loading = time.time() - start
        print('time loading: %0.2f' % time_loading)


def load_from_hdf5():
    h5_path = '/home/gabi/PycharmProjects/uatu/data/all_data_uncompressed.h5'


    with h5py.File(h5_path, 'r') as hf:
        start = time.time()
        data_pos = hf['positives'][0:10000]
        total = time.time() - start
        print('total time loading all pos: %0.2f' % total)

        start = time.time()
        data_neg = hf['negatives'][0:10000]
        total = time.time() - start
        print('total time loading all neg: %0.2f' % total)

