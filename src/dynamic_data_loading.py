"""
The purpose of this file is to perform manipulations with the list of keys pointing
to the location of the image files.
key_handling
"""

"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

File description:

File has multiple functions:

1) Transforms raw images and video to the specified format that we need to feed the network.
2) Creates pairs of images and videos HDF5 datasets
3) Saves image and video data to
"""


import numpy as np
import project_constants as pc
from scipy import ndimage
import random
import csv
import keras
import h5py
import time
import os
from project_variables import ProjectVariable as pv
import data_pipeline as dp
from random import randint


# unused
def analyze_data_set(dataset):
    data_list = list(csv.reader(np.genfromtxt(dataset, dtype=None)))
    labels = np.asarray([data_list[row][2] for row in range(0, len(data_list))], dtype=int)
    positives_percentage = np.sum(labels) * 1.0 / len(labels)
    negatives_percentage = 1.0 - positives_percentage
    return [positives_percentage, negatives_percentage]


# unused
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


# used in unused method
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


# NOTE: assume a 20 ranking
def make_ranking_test(rank_list_pos, data):
    if data == 'cuhk':
        def match(one, two):
            return list(one)[0:4] == list(two)[0:4]

        random.shuffle(rank_list_pos)

        rank_pos = []
        pos_tally = []
        #select 20 unique pairs
        for item in range(len(rank_list_pos)):
            i1 = rank_list_pos[item].split(',')[0].split('/')[-1][0:4]
            i2 = rank_list_pos[item].split(',')[1].split('/')[-1][0:4]
            if i1 == i2:
                if i1 not in pos_tally:
                    pos_tally.append(i1)
                    rank_pos.append(rank_list_pos[item])

        rank_list_pos = rank_pos

    list_0 = [rank_list_pos[index].split(',')[0] for index in range(len(rank_list_pos))]
    list_1 = [rank_list_pos[index].split(',')[1] for index in range(len(rank_list_pos))]
    ranking_test_file = '../data/ranking_test.txt'

    with open(ranking_test_file, 'wr') as myFile:
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                num = 1 if img0 == img1 else 0
                line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
                myFile.write(line)

    ranking_test = np.genfromtxt(ranking_test_file, dtype=None).tolist()

    return ranking_test


'''
ASSUMPTION: there is a positive and negative list in each dataset
if there are no positives or negatives in the dataset, then merge the set with another set that contains these
'''
#FIXME remove test_data stuff in the argumetns
# note: if data_type == 'image', total_data_list has to contain the full path to the image
# note: if data_type == 'hdf5', total_data_list has to contain indices according to the saved hdf5 file
def make_validation_test_list(total_data_list_pos, total_data_list_neg, val_percent=0.01, test_percent=0.01,
          val_pos_percent=0.3, test_pos_percent=0.1, data_type='hdf5', ranking=False):
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

        if ranking:
            rank_list_viper_pos = np.genfromtxt('../data/VIPER/ranking_pos.txt',
                                                dtype=None).tolist()

            test_list_viper = make_ranking_test(rank_list_viper_pos, data='viper')

            rank_list_cuhk_pos = np.genfromtxt('../data/CUHK/ranking_pos.txt',
                                                dtype=None).tolist()
            test_list_cuhk = make_ranking_test(rank_list_cuhk_pos, data='cuhk')

            return val_list, test_list_viper, test_list_cuhk, total_data_list_pos, total_data_list_neg

        else:
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


def load_in_array(adjustable, data_pos=None, data_neg=None, hdf5_file=None, data_list=None, heads=1,
                  data_type='hdf5'):
    if heads == 1:
        if data_type == 'hdf5':

            data_array = np.zeros(shape=(len(data_pos)+len(data_neg), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

            for image_1 in range(len(data_pos)):
                data_array[image_1] = hdf5_file['positives'][data_pos[image_1]]
            for image_2 in range(len(data_neg)):
                data_array[len(data_pos) + image_2] = hdf5_file['negatives'][data_neg[image_2]]

            labels = np.append(np.ones(len(data_pos)), np.zeros(len(data_neg)))

            everything = zip(data_array, labels)
            random.shuffle(everything)
            data_array, labels = zip(*everything)
            if adjustable.cost_module_type == 'neural_network':
                labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
            return np.asarray(data_array), labels

        else:
            data_array = np.zeros(shape=(len(data_list), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
            labels = np.zeros(len(data_list))
            for image in range(0, len(data_list)):
                name = data_list[image].split(',')[0]
                data_array[image] = ndimage.imread(name)[:, :, 0:3]
                labels[image] = int(data_list[image].split(',')[1])
                if adjustable.cost_module_type == 'neural_network':
                    labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
                return data_array, labels
    else:
        if data_type == 'images':
            data_array = np.zeros(shape=(len(data_list),  pc.NUM_SIAMESE_HEADS, pc.IMAGE_HEIGHT,
                                         pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
            labels = np.zeros(len(data_list))
            for pair in range(0, len(data_list)):
                for image in range(0,2):
                    name = data_list[pair].split(',')[image]
                    data_array[pair][image] = ndimage.imread(name)[:, :, 0:3]
                    labels[pair] = int(data_list[pair].split(',')[2])

            if adjustable.cost_module_type == 'neural_network':
                labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
            return data_array, labels

        else:
            pass


# used in unused methods
def fetch_dummy_data():
    print('fetching data')
    path_pos = '../data/all_positives.txt'
    path_neg = '../data/all_negatives.txt'

    list_pos = np.genfromtxt(path_pos, dtype=None).tolist()
    list_neg = np.genfromtxt(path_neg, dtype=None).tolist()

    return list_pos, list_neg


# unused
def data_pos_to_hdf5():
    data_pos, data_neg = fetch_dummy_data()
    # amount = 40000
    h5_path = '../data/all_data_uncompressed.h5'

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


# unused
def data_neg_to_hdf5():
    data_pos, data_neg = fetch_dummy_data()
    # amount = 40000
    h5_path = '../data/all_data_uncompressed.h5'

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


# unused
def load_from_hdf5():
    h5_path = '../data/all_data_uncompressed.h5'

    with h5py.File(h5_path, 'r') as hf:
        start = time.time()
        data_pos = hf['positives'][0:10000]
        total = time.time() - start
        print('total time loading all pos: %0.2f' % total)

        start = time.time()
        data_neg = hf['negatives'][0:10000]
        total = time.time() - start
        print('total time loading all neg: %0.2f' % total)


# unused
def txt_to_hdf5(text_file, hdf5_file_name):
    h5_path = os.path.join('../data/', hdf5_file_name)

    data_list = np.genfromtxt(text_file, dtype=None)
    with h5py.File(h5_path, 'w') as myFile:
        print('loading data into hdf5')
        start = time.time()
        data = myFile.create_dataset(name='data', data=data_list)
        time_loading = time.time() - start
        print('time loading: %0.2f' % time_loading)


# unused
def get_data_scnn(adjustable):
    total_data_list_pos = np.genfromtxt(pc.POSITIVE_DATA, dtype=None)
    with h5py.File(pc.NEGATIVE_DATA, 'r') as hf:
        total_data_list_neg = hf['data'][()]

    val_list, test_list_viper, test_list_cuhk, total_data_list_pos, total_data_list_neg = make_validation_test_list(
        total_data_list_pos, total_data_list_neg, val_pos_percent=0.1, test_pos_percent=0.1, data_type='images',
        ranking=True)

    validation_data, validation_labels = load_in_array(adjustable, data_list=val_list,
                                                        data_type='images',
                                                        heads=2)
    test_data_viper, test_labels_viper = load_in_array(adjustable, data_list=test_list_viper,
                                                        data_type='images',
                                                        heads=2)

    test_data_cuhk, test_labels_cuhk = load_in_array(adjustable, data_list=test_list_cuhk,
                                                        data_type='images',
                                                        heads=2)

    total_data_list = [total_data_list_pos, total_data_list_neg]
    validation = [validation_data, validation_labels]
    test = ['viper', test_data_viper, test_labels_viper,
            'cuhk01', test_data_cuhk, test_labels_cuhk]

    return total_data_list, validation, test


# text_file = '../data/reid_all_negatives.txt'
# h5_name = 'reid_all_negatives_uncompressed.h5'
# txt_to_hdf5(text_file, h5_name)
def get_dataset(name):
    """ Given the name of a dataset it returns that dataset as a h5 file
        note that you cannot learn on cuhk01 and cuhk02 at the same time
        since cuhk02 includes cuhk01
    """
    if name == 'viper':
        dataset_h5 = h5py.File('../data/VIPER/viper.h5', 'r')
    elif name == 'cuhk01':
        dataset_h5 = h5py.File('../data/CUHK/cuhk01.h5', 'r')
    elif name == 'cuhk02':
        dataset_h5 = h5py.File('../data/CUHK02/cuhk02.h5', 'r')
    elif name == 'market':
        dataset_h5 = h5py.File('../data/market/market.h5', 'r')
    elif name == 'caviar':
        dataset_h5 = h5py.File('../data/caviar/caviar.h5', 'r')
    elif name == 'grid':
        dataset_h5 = h5py.File('../data/GRID/grid.h5', 'r')
    elif name == 'prid450':
        dataset_h5 = h5py.File('../data/prid450/prid450.h5', 'r')
    elif name == 'ilids-vid':
        dataset_h5 = h5py.File('../data/ilids-vid/ilids-vid.h5', 'r')
    elif name == 'prid2011':
        dataset_h5 = h5py.File('../data/prid2011/prid2011.h5', 'r')
    elif name == 'inria':
        dataset_h5 = h5py.File('../data/INRIA/inria.h5', 'r')
    else:
        dataset_h5 = None

    return dataset_h5


def load_datasets_from_h5(list_of_datasets):
    """ Do this at the beginning of the experiment
    """
    h5_data = []
    for dataset in list_of_datasets:
        h5_data.append(get_dataset(dataset))

    return h5_data

def create_training_and_ranking_set(name, adjustable):
    """ Do this at the beginning of each iteration
    """
    if name == 'viper':
        # ranking, training_pos, training_neg = dp.make_pairs_viper(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.VIPER_DATA_STORAGE, pc.VIPER_FIXED)
    elif name == 'cuhk01':
        # ranking, training_pos, training_neg = dp.make_pairs_cuhk1(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.CUHK01_DATA_STORAGE, pc.CUHK01_FIXED)
    elif name == 'cuhk02':
        ranking, training_pos, training_neg = dp.make_pairs_cuhk2(adjustable)
    elif name == 'market':
        # ranking, training_pos, training_neg = dp.make_pairs_market(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.MARKET_DATA_STORAGE, pc.MARKET_FIXED)
    elif name == 'caviar':
        # ranking, training_pos, training_neg = dp.make_pairs_caviar(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.CAVIAR_DATA_STORAGE, pc.CAVIAR_FIXED)
    elif name == 'grid':
        # ranking, training_pos, training_neg = dp.make_pairs_grid(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.GRID_DATA_STORAGE, pc.GRID_FIXED)
    elif name == 'prid450':
        # ranking, training_pos, training_neg = dp.make_pairs_prid450(adjustable)
        ranking, training_pos, training_neg = dp.make_pairs_image(adjustable, pc.PRID450_DATA_STORAGE, pc.PRID450_FIXED)
    elif name == 'ilids-vid':
        ranking, training_pos, training_neg = dp.make_pairs_video(adjustable)
    elif name == 'prid2011':
        ranking, training_pos, training_neg = dp.make_pairs_video(adjustable)
    else:
        ranking, training_pos, training_neg = None, None, None

    return ranking, training_pos, training_neg


def merge_datasets(adjustable, list_training_pos, list_training_neg, balance=False):
    """ Merges specified datasets by shuffling together the positive and negative training instances.
        There will be many more negative instances than positive instances. This method needs to be excecuted
        once, right after 'create_trainin_and_ranking_set()'.

    :param adjustable:              object of class ProjectVariable created in 'running_experiments.py'
    :param list_training_pos:       A list of all training positive instances of the different datasets
    :param list_training_neg:       A list of all training negative instances of the different datasets
    :return:                        Two lists composed of the specified compositions of the data for negative
                                    and positive instances.
    """
    number_of_datasets = len(adjustable.datasets)

    merged_training_pos = []
    merged_training_neg = []

    # balance the datasets such that each dataset has the same presence in the training set
    count_pos = [len(item) for item in list_training_pos]
    min_pos_1 = min(count_pos)
    count_neg = [len(item) for item in list_training_neg]
    min_neg_1 = min(count_neg)

    if number_of_datasets > 1:
        for dataset in range(number_of_datasets):
            pos = list_training_pos[dataset]
            neg = list_training_neg[dataset]
            random.shuffle(pos)
            random.shuffle(neg)

            if balance:
                min_pos = min_pos_1
                min_neg = min_neg_1
            else:
                min_pos = len(list_training_pos[dataset])
                min_neg = len(list_training_neg[dataset])

            merged_training_pos = merged_training_pos + pos[0:min_pos]
            merged_training_neg = merged_training_neg + neg[0:min_neg]
    else:
        merged_training_pos = list_training_pos[0]
        merged_training_neg = list_training_neg[0]

    random.shuffle(merged_training_pos)
    random.shuffle(merged_training_neg)
    return merged_training_pos, merged_training_neg


def get_dataset_to_map(name, data_list, data_names):

    # get hdf dataset filename, used only for pairs of images
    # split on '/' get second to last item
    # mathc with name
    # return data_list[indexof(match)]

    if name == 'padded':
        dataset = 'VIPER'
    elif name == 'identities':
        dataset = 'market'
    elif name == 'images':
        dataset = 'CUHK'
    elif name == 'all':
        dataset = 'CUHK02'
    elif name == 'fixed_caviar':
        dataset = 'caviar'
    elif name == 'fixed_prid':
        dataset = 'prid450'
    elif name == 'fixed_grid':
        dataset = 'GRID'
    else:
        print("sorry, we don't serve '%s'. would you like some fries with that?" % name)
        dataset = None

    return data_list[data_names.index(dataset)]



def create_key_dataset_mapping(key_list, h5_dataset_list):
    """ Creates a mapping from the keys to the datasets. This way we know where to find each key
    :param key_list:            list of keys in form of tuples with a label "img1,img2,1"
    :param h5_dataset_list:     list of the h5 datasets to search in
    :return:                    a mapping from the keys to the datasets
    """
    # print('CREATING KEY MAPPING')
    key_dataset_mapping = []

    if len(h5_dataset_list) == 1:

        for key in key_list:
            key_1 = key.split(',')[0]
            key_2 = key.split(',')[1]

            mapping_1 = [key_1, h5_dataset_list[0]]
            mapping_2 = [key_2, h5_dataset_list[0]]

            key_dataset_mapping.append(mapping_1)
            key_dataset_mapping.append(mapping_2)

    else:
        h5_filenames = [str(item.file.filename.split('/')[-2]) for item in h5_dataset_list]
        for key in key_list:
            key_1 = key.split(',')[0]
            key_2 = key.split(',')[1]
            folder_key_1 = key_1.split('+')[-2]
            folder_key_2 = key_2.split('+')[-2]

            dataset_key_1 = get_dataset_to_map(folder_key_1, h5_dataset_list, h5_filenames)
            dataset_key_2 = get_dataset_to_map(folder_key_2, h5_dataset_list, h5_filenames)

            mapping_1 = [key_1, dataset_key_1]
            mapping_2 = [key_2, dataset_key_2]

            key_dataset_mapping.append(mapping_1)
            key_dataset_mapping.append(mapping_2)
    return key_dataset_mapping


def grab_em_by_the_keys(key_list, h5_dataset_list):
    """ Returns a training set
    :param key_list:                list of keys
    :param key_dataset_mapping:     list of the participating h5 datasets
    :return:
    """

    # create mapping from keys to dataset
    key_dataset_mapping = create_key_dataset_mapping(key_list, h5_dataset_list)
    # isolate the different keys
    all_key_1 = [item.split(',')[0] for item in key_list]
    all_key_2 = [item.split(',')[1] for item in key_list]
    all_keys_in_mapping = [item[0] for item in key_dataset_mapping]
    # # isolate the values
    only_values = [item[1] for item in key_dataset_mapping]
    # get the index of the value that key in all_key points to
    # FIXME: taking incredibly fucking long if we do big ranking numbers
    # TODO: fix it, optimize it somehow so that we can run with bigger ranking numbers
    the_index_key_1 = [all_keys_in_mapping.index(key_1) for key_1 in all_key_1]
    the_index_key_2 = [all_keys_in_mapping.index(key_2) for key_2 in all_key_2]
    # get the values from the h5 file given the indices
    values_key_1 = []
    for item in range(len(all_key_1)):
        a = all_key_1[item]
        b = the_index_key_1[item]
        c = only_values[b][a][:]
        values_key_1.append(c)

    values_key_1 = [only_values[the_index_key_1[item]][all_key_1[item]][:] for item in range(len(all_key_1))]
    values_key_2 = [only_values[the_index_key_2[item]][all_key_2[item]][:] for item in range(len(all_key_2))]
    return values_key_1, values_key_2


def get_positive_keys(name_dataset, partition, the_id, seen_list):
    """For priming
        Gets a list of related keys based on the id you are looking for
    """
    # print('the id: %s' % str(the_id))
    # print('the seen list: %s' % str(seen_list))

    # note: `partition` is only applicable for CUHK02
    if name_dataset == 'cuhk02':
        all_partition_ids_in_order = list(np.genfromtxt('../data/CUHK02/%s/id_all_file.txt' % partition, dtype=None))
        indices_matching_id = [item for item in range(len(all_partition_ids_in_order)) if
                               all_partition_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/CUHK02/%s/short_image_names_file.txt' % partition, dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]

        # make sure the probe is in the training set by removing it from seen
        updated_indices_seen = [indices_seen_image[-1]]

        # updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
        #                    item not in indices_seen_image]
        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in updated_indices_seen]


        all_partition_keys_in_order = list(
            np.genfromtxt('../data/CUHK02/%s/fullpath_image_names_file.txt' % partition, dtype=None))
        keys = [all_partition_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_partition_keys_in_order[item] for item in indices_seen_image]
    elif name_dataset == 'market':
        all_ids_in_order = list(np.genfromtxt('../data/market/id_all_file.txt', dtype=str))
        indices_matching_id = [item for item in range(len(all_ids_in_order)) if all_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/market/short_image_names_file.txt', dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]

        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in indices_seen_image]

        # truncate list to 4 images -- for faster testing
        # add the probe
        probe = all_image_names.index(seen_list[0])
# FIXME why is this set to 2?
        updated_indices = updated_indices[0:2]
        updated_indices += [probe]

        all_keys_in_order = list(np.genfromtxt('../data/market/fullpath_image_names_file.txt', dtype=None))
        keys = [all_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_keys_in_order[item] for item in indices_seen_image]
    elif name_dataset == 'grid':
        all_ids_in_order = list(np.genfromtxt('../data/GRID/id_all_file.txt', dtype=str))
        indices_matching_id = [item for item in range(len(all_ids_in_order)) if all_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/GRID/short_image_names_file.txt', dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        # print('image names matching id: %s' % str(image_names_matching_id))
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]
        # print('indices seen image: %s' % str(indices_seen_image))

        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in indices_seen_image]
        # print('updated indices: %s' % str(updated_indices))

        # truncate list to 4 images -- for faster testing
        # add the probe
        probe = all_image_names.index(seen_list[0])
        # print('probe: %s' % str(probe))
        updated_indices = updated_indices[0:2]
        # print('updated indices: %s' % str(updated_indices))
        updated_indices += [probe]
        # print('updated indices: %s' % str(updated_indices))

        all_keys_in_order = list(np.genfromtxt('../data/GRID/fullpath_image_names_file.txt', dtype=None))
        keys = [all_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_keys_in_order[item] for item in indices_seen_image]
    elif name_dataset == 'prid450':

        all_ids_in_order = list(np.genfromtxt('../data/prid450/id_all_file.txt', dtype=str))
        indices_matching_id = [item for item in range(len(all_ids_in_order)) if all_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/prid450/short_image_names_file.txt', dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]

        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in indices_seen_image]

        # truncate list to 4 images -- for faster testing
        # add the probe
        probe = all_image_names.index(seen_list[0])
        updated_indices = updated_indices[0:2]
        updated_indices += [probe]

        all_keys_in_order = list(np.genfromtxt('../data/prid450/fullpath_image_names_file.txt', dtype=None))
        keys = [all_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_keys_in_order[item] for item in indices_seen_image]
    elif name_dataset == 'viper':

        all_ids_in_order = list(np.genfromtxt('../data/VIPER/id_all_file.txt', dtype=str))
        indices_matching_id = [item for item in range(len(all_ids_in_order)) if all_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/VIPER/short_image_names_file.txt', dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]

        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in indices_seen_image]

        # truncate list to 4 images -- for faster testing
        # add the probe
        probe = all_image_names.index(seen_list[0])
        updated_indices = updated_indices[0:2]
        updated_indices += [probe]

        all_keys_in_order = list(np.genfromtxt('../data/VIPER/fullpath_image_names_file.txt', dtype=None))
        keys = [all_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_keys_in_order[item] for item in indices_seen_image]
    elif name_dataset == 'caviar':

        all_ids_in_order = list(np.genfromtxt('../data/caviar/id_all_file.txt', dtype=str))
        indices_matching_id = [item for item in range(len(all_ids_in_order)) if all_ids_in_order[item] == the_id]

        all_image_names = list(np.genfromtxt('../data/caviar/short_image_names_file.txt', dtype=None))
        image_names_matching_id = [all_image_names[item] for item in indices_matching_id]
        indices_seen_image = [image_names_matching_id.index(im) for im in seen_list for name in image_names_matching_id
                              if im == name]

        updated_indices = [indices_matching_id[item] for item in range(len(indices_matching_id)) if
                           item not in indices_seen_image]

        # truncate list to 4 images -- for faster testing
        # add the probe
        probe = all_image_names.index(seen_list[0])
        updated_indices = updated_indices[0:2]
        updated_indices += [probe]

        all_keys_in_order = list(np.genfromtxt('../data/caviar/fullpath_image_names_file.txt', dtype=None))
        keys = [all_keys_in_order[item] for item in updated_indices]
        # TODO switch back
        # keys = [all_keys_in_order[item] for item in indices_seen_image]
    else:
        keys = None

    return keys


def get_negative_keys(adjustable, name_dataset, partition, seen_list, this_ranking, positive_keys):
    """ For priming
        get negative keys. get key that could have been seen before in the training set, but that is not an id in
        the test set
    """
    number_positive_keys = len(positive_keys)

    if name_dataset == 'cuhk02':
        rank_ordered_partitions = [this_ranking[item * pc.RANKING_DICT['cuhk02'] + item].strip().split(',')[0].split('+')[-3]
                                   for item in range(pc.RANKING_DICT['cuhk02'])]
        rank_ordered_ids = [
            dp.my_join(list(this_ranking[item * pc.RANKING_DICT['cuhk02'] + item].strip().split(',')[0].split('+')[-1])[0:4])
            for item in range(pc.RANKING_DICT['cuhk02'])]
        # create list in the form of [(partition, id), ...]
        joined_unique = list(set(zip(rank_ordered_partitions, rank_ordered_ids)))

        negative_keys = []

        for num in range(number_positive_keys):
            # choose a random partition
            p = 'P%s' % str(randint(1, 5))
            # get the list of unique cuhk02 IDs
            unique_ids = list(np.genfromtxt('../data/CUHK02/%s/unique_id_file.txt' % p, dtype=None))
            # get list of swapped fullpath
            swapped_fullpath = list(
                np.genfromtxt('../data/CUHK02/%s/fullpath_image_names_file.txt' % p, dtype=None))
            # get list of all ids
            all_ids = list(np.genfromtxt('../data/CUHK02/%s/id_all_file.txt' % p, dtype=None))
            # at random choose an id from that list
            index = randint(0, len(unique_ids)-1)
            chosen = unique_ids.pop(index)
            # check if id already in ranking list
            i = 1
            while (p, chosen) in joined_unique:
                index = randint(0, len(unique_ids) - 1 - i)
                chosen = unique_ids.pop(index)
                i += 1
            # get the first index matching the chosen id from the list of all ids
            index = all_ids.index(chosen)
            # get the key with the index and append to the list
            negative_keys.append(swapped_fullpath[index])
    else:
        if adjustable.ranking_number == 'half':
            ranking_number = pc.RANKING_DICT[name_dataset]
        elif isinstance(adjustable.ranking_number, int):
            ranking_number = adjustable.ranking_number
        else:
            ranking_number = None

        # print('ranking number: %s' % str(ranking_number))

        # rank_ordered_ids = [
            # dp.my_join(list(this_ranking[item * ranking_number + item].strip().split(',')[0].split('+')[-1])[0:4])
            # for item in range(ranking_number)]
        rank_ordered_ids = []

        for thingy in this_ranking:
            print(thingy)

        for item in range(ranking_number):
            a = list(this_ranking[item * ranking_number + item].strip().split(',')[0].split('+')[-1])
            # print('a: %s' % str(a))
            b = dp.my_join(a[0:4])
            # print('b: %s' % str(b))
            rank_ordered_ids.append(b)

        negative_keys = []

        if name_dataset == 'market':
            folder_name = 'market'
        elif name_dataset == 'grid':
            folder_name = 'GRID'
        elif name_dataset == 'viper':
            folder_name = 'VIPER'
        elif name_dataset == 'prid450':
            folder_name = 'prid450'
        elif name_dataset == 'caviar':
            folder_name = 'caviar'
        else:
            folder_name = None

        # get the list of unique dataset IDs
        unique_ids = list(np.genfromtxt('../data/%s/unique_id_file.txt' % folder_name, dtype=None))
        # get list of swapped fullpath
        swapped_fullpath = list(np.genfromtxt('../data/%s/fullpath_image_names_file.txt' % folder_name, dtype=None))
        # get list of all ids
        all_ids = list(np.genfromtxt('../data/%s/id_all_file.txt' % folder_name, dtype=None))

        for num in range(number_positive_keys):
            # at random choose an id from that list
            index = randint(0, len(unique_ids)-1)
            chosen = unique_ids.pop(index)
            # check if id already in ranking list
            i = 1
            while chosen in rank_ordered_ids:
                index = randint(0, len(unique_ids) - 1 - i)
                chosen = unique_ids.pop(index)
                i += 1
            # get the first index matching the chosen id from the list of all ids
            index = all_ids.index(chosen)
            # get the key with the index and append to the list
            negative_keys.append(swapped_fullpath[index])

    return negative_keys


def get_related_keys(adjustable, name_dataset, partition, seen_list, this_ranking, the_id):
    pos_keys = get_positive_keys(name_dataset, partition, the_id, seen_list)
    neg_keys = get_negative_keys(adjustable, name_dataset, partition, seen_list, this_ranking, pos_keys)
    keys = pos_keys + neg_keys

    return keys


def get_human_data(keys, h5data):
    h5data = h5data[0]
    len_keys = len(keys)

    thingy = []
    for item in range(len_keys):
        key = keys[item]
        thingy.append(h5data[key][:,:,0:3])

    thingy = np.asarray(thingy)

    return thingy