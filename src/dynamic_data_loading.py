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


def make_specific_balanced_set_given_pos_neg(dataset_pos, dataset_neg, positives_percentage, set_size, data_type='hdf5'):

    if data_type == 'hdf5':
        random.shuffle(dataset_pos)
        random.shuffle(dataset_neg)

        num_of_pos = np.floor(positives_percentage * set_size).astype(int)
        num_of_neg = set_size - num_of_pos

        pos_data = dataset_pos[0:num_of_pos]
        neg_data = dataset_neg[0:num_of_neg]

        new_data_list_pos = dataset_pos[num_of_pos:]
        new_data_list_neg = dataset_neg[num_of_neg:]

        return pos_data, neg_data, new_data_list_pos, new_data_list_neg
    else:
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
# note: if data_type == 'image', total_data_list has to contain the full path to the image
# note: if data_type == 'hdf5', total_data_list has to contain indices according to the saved hdf5 file
def make_validation_test_list(total_data_list_pos, total_data_list_neg, val_percent=0.001, test_percent=0.001,
          val_pos_percent=0.3, test_pos_percent=0.1, data_type='hdf5'):
    num_pos = len(total_data_list_pos)
    num_neg = len(total_data_list_neg)

    min_num = min(num_pos, num_neg)
    balanced_total = 2 * min_num

    val_size = np.floor(val_percent * balanced_total).astype(int)
    test_size = np.floor(test_percent * balanced_total).astype(int)

    if data_type == 'hdf5':
        val_list_pos, val_list_neg, \
        total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                                                            total_data_list_neg,
                                                                                            val_pos_percent, val_size,
                                                                                            data_type=data_type)

        test_list_pos, test_list_neg, \
        total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(total_data_list_pos,
                                                                                            total_data_list_neg,
                                                                                            val_pos_percent, test_size,
                                                                                            data_type=data_type)

        return val_list_pos, val_list_neg, test_list_pos, test_list_neg, total_data_list_pos, total_data_list_neg

    else:
        val_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(
            total_data_list_pos, total_data_list_neg, val_pos_percent, val_size, data_type=data_type)

        test_list, total_data_list_pos, total_data_list_neg = make_specific_balanced_set_given_pos_neg(
            total_data_list_pos, total_data_list_neg, test_pos_percent, test_size, data_type=data_type)

        return val_list, test_list, total_data_list_pos, total_data_list_neg


def make_train_batches(total_data_list_pos, total_data_list_neg, data_type='hdf5'):
    random.shuffle(total_data_list_pos)
    random.shuffle(total_data_list_neg)
    total_len_half = min(len(total_data_list_neg), len(total_data_list_pos))
    
    if data_type == 'hdf5':
        total_data_list_pos = total_data_list_pos[0:total_len_half]
        total_data_list_neg = total_data_list_neg[0:total_len_half]
        return total_data_list_pos, total_data_list_neg
    else:
        
        total_data_list = np.concatenate((total_data_list_pos[0:total_len_half], total_data_list_neg[0:total_len_half]))
        random.shuffle(total_data_list)
        return total_data_list


def load_in_array(data_pos, data_neg, hdf5_file=None, data_list=None, heads=1, data_type='hdf5'):
    if heads == 1:
        if data_type == 'hdf5':

            data_array = np.zeros(shape=(len(data_pos)+len(data_neg), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

            for image_1 in xrange(len(data_pos)):
                data_array[image_1] = hdf5_file['positives'][data_pos[image_1]]
            for image_2 in xrange(len(data_neg)):
                data_array[len(data_pos) + image_2] = hdf5_file['negatives'][data_neg[image_2]]

            labels = np.append(np.ones(len(data_pos)), np.zeros(len(data_neg)))

            everything = zip(data_array, labels)
            random.shuffle(everything)
            data_array, labels = zip(*everything)
            labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
            return np.asarray(data_array), labels

        else:
            data_array = np.zeros(shape=(len(data_list), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
            labels = np.zeros(len(data_list))
            for image in xrange(0, len(data_list)):
                name = data_list[image].split(',')[0]
                data_array[image] = ndimage.imread(name)[:, :, 0:3]
                labels[image] = int(data_list[image].split(',')[1])
                labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
                return data_array, labels
    else:
        if data_type == 'hdf5':

            pass
        else:
            pass




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

