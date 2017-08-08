"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

File has multiple functions:

1) Creates pairs of images and videos and stored as text data
2) Performs manipulations with data pairs (image and video)
"""
import numpy as np
import os
import random as rd
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from random import randint
import random
import project_constants as pc
import project_utils as pu


def make_image_data_files(fixed_dataset_path, project_data_storage):
    """
    Creates 5 text files, and saves them in the dataset directory:
    id_all_file.txt                 contains the all unique IDs in order
    unique_id_file.txt              the set of id_all_file.txt (so no duplicates)
    short_image_names_file.txt      contains all the image names, without the full path
    fullpath_image_names_file.txt   short_image_names_file.txt with full path
    swapped_list_of_paths.txt       fullpath_image_names_file.txt but with '+' instead of '/'

    :param fixed_dataset_path:      string path to standardized dataset directory
    :param project_data_storage:    string path where the image data gets stored
    """
    folder_path = fixed_dataset_path
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')
    swapped_list_of_paths = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')

    pu.write_to_file(id_all_file, id_all)
    pu.write_to_file(unique_id_file, unique_id)
    pu.write_to_file(short_image_names_file, short_image_names)
    pu.write_to_file(fullpath_image_names_file, fullpath_image_names)
    with open(swapped_list_of_paths, 'w') as myfile:
        for item in fullpath_image_names:
            item_name = pu.swap_for(item, '/', '+')
            myfile.write(item_name + '\n')


def make_image_data_cuhk2():
    """
    Creates 5 text files, one for each partition in CUHK02:
    id_all_file.txt                 contains the all unique IDs in order
    unique_id_file.txt              the set of id_all_file.txt (so no duplicates)
    short_image_names_file.txt      contains all the image names, without the full path
    fullpath_image_names_file.txt   short_image_names_file.txt with full path
    swapped_list_of_paths.txt       fullpath_image_names_file.txt but with '+' instead of '/'
    """
    top_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK2'

    subdirs = os.listdir(top_path)

    for a_dir in subdirs:
        folder_path = os.path.join(top_path, a_dir, 'all')

        id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
        unique_id = sorted(set(id_all))
        short_image_names = sorted(os.listdir(folder_path))
        fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])

        project_data_storage = os.path.join('../data/CUHK02', a_dir)
        if not os.path.exists(project_data_storage):
            os.mkdir(project_data_storage)

        id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
        unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
        short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
        fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')
        swapped_list_of_paths = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')

        pu.write_to_file(id_all_file, id_all)
        pu.write_to_file(unique_id_file, unique_id)
        pu.write_to_file(short_image_names_file, short_image_names)
        pu.write_to_file(fullpath_image_names_file, fullpath_image_names)
        with open(swapped_list_of_paths, 'w') as myfile:
            for item in fullpath_image_names:
                item_name = pu.swap_for(item, '/', '+')
                myfile.write(item_name + '\n')


def make_positive_combinations(fullpath, unique_ids, num, smallest_id_group, type='rank', augment_equal=False):
    """
    Makes positive pairs for images (or sequences in the case of video data)
    :param fullpath:            list of fullpath names
    :param unique_ids:          list of unique IDs
    :param num:                 int number of image per unique ID (sets a boundary)
    :param smallest_id_group:   int size of the smallest set of images per ID
    :param type:                string 'train' or 'rank'. indicates making combos for training or ranking
    :param augment_equal:       bool. indicates to augment with pairs of the same images
    :return:                    a list of matching pairs with label 'item_a,item_b,1'
    """
    if num > smallest_id_group:
        num = smallest_id_group
    combo_list = []
    num_ids = len(unique_ids)

    if num == 3:
        for item in range(num_ids):
            thing = str(fullpath[num * item] + ',' + fullpath[num * item + 1] + ',1\n')
            combo_list.append(thing)
            thing = str(fullpath[num * item + 1] + ',' + fullpath[num * item + 2] + ',1\n')
            combo_list.append(thing)

            # make augmented data to increase the number of positives in the dataset
            if augment_equal == True:
                thing = str(fullpath[num * item] + ',' + fullpath[num * item] + ',1\n')
                combo_list.append(thing)
                thing = str(fullpath[num * item + 1] + ',' + fullpath[num * item + 1] + ',1\n')
                combo_list.append(thing)
                thing = str(fullpath[num * item + 2] + ',' + fullpath[num * item + 2] + ',1\n')
                combo_list.append(thing)

    elif num == 2:
        for item in range(num_ids):
            thing = str(fullpath[num * item] + ',' + fullpath[num * item + 1] + ',1\n')
            combo_list.append(thing)

            if type == 'train' and augment_equal == True:
                thing = str(fullpath[num * item] + ',' + fullpath[num * item] + ',1\n')
                combo_list.append(thing)
                thing = str(fullpath[num * item + 1] + ',' + fullpath[num * item + 1] + ',1\n')
                combo_list.append(thing)

    return combo_list


def pre_selection(the_list, unique_ids, all_ids, num, dataset_name):
    """ Prevents there from being a HUGE number of combinations of pairs by setting an upper bound on allowed images per
        unique ID
    :param the_list:        list containing full path to images of the set of IDs. an ID can have multiple images
    :param unique_ids:      list of unique IDs. every ID appears only once
    :param all_ids:         the_list, but then only the IDs
    :param num:             the number of allowed image per ID
    :param dataset_name     string name of the datasaet
    :return:                truncated selection of the_list
    """
    selection = []
    min_id_group_size = 10000000
    # keep track of which ids we ignore.
    ignore_id = []

    for the_id in unique_ids:
        # get the indices for the matching IDs
        # id_group = [i for i, x in enumerate(all_ids) if x == the_id]
        id_group = []
        for i, x in enumerate(all_ids):
            if x == the_id:
                id_group.append((i))

        # get the fullpaths for each matching ID at the indices
        # full_path_group = [the_list[i] for i in id_group]
        full_path_group = []
        for i in id_group:
            full_path_group.append(the_list[i])

        # update min_id_group_size
        # if dataset is large and has a lot of large id groups(>num) then ignore the id groups that are smaller than num
        if dataset_name in {'market', 'cuhk02', 'caviar', 'prid2011', 'ilids-vid'}:
            if len(id_group) >= num:
                # print('length id group: %d, num: %d' % (len(id_group), num))
                if min_id_group_size > len(id_group):
                    min_id_group_size = len(id_group)
                # if there are more matching ID images than allowed images,
                # only add the number of allowed matching ID images
                # use a random number to decide which images get popped
                for ble in range(num):
                    selection.append(full_path_group.pop(rd.randrange(0, len(full_path_group))))
            else:
                ignore_id.append(the_id)
        else:
            if min_id_group_size > len(id_group):
                min_id_group_size = len(id_group)
            if num > len(id_group):
                # if the number of allowed images is greater than the number of matching ID images,
                # add all the images of that ID to the selection
                sub_selection = [thing for thing in full_path_group]
                selection += sub_selection
            else:
                # if there are more matching ID images than allowed images,
                # only add the number of allowed matching ID images
                # use a random number to decide which images get popped
                for ble in range(num):
                    selection.append(full_path_group.pop(rd.randrange(0, len(full_path_group))))

    for value in ignore_id:
        index = unique_ids.index(value)
        unique_ids.pop(index)

    return selection, min_id_group_size, unique_ids


def make_positive_pairs_training(id_all_file, unique_id_file, swapped_list_of_paths, dataset_name):
    # load the image data from saved txt files
    train_ids = list(np.genfromtxt(unique_id_file, dtype=None))
    all_train_ids = list(np.genfromtxt(id_all_file, dtype=None))
    training_ids_pos = list(np.genfromtxt(swapped_list_of_paths, dtype=None))

    # -- Create combinations and store the positive matches for training
    # You could increase this but then you'll get a lot more data
    upper_bound = 3
    training_ids_pos, min_group_size_train, train_ids = pre_selection(training_ids_pos, train_ids, all_train_ids,
                                                                      upper_bound, dataset_name)
    training_ids_pos = make_positive_combinations(training_ids_pos, train_ids, upper_bound, min_group_size_train,
                                                  type='train')

    # shuffle so that each time we get different first occurences for when making negative pairs
    rd.shuffle(training_ids_pos)

    return training_ids_pos


def make_positive_pairs_ranking(id_all_file, unique_id_file, swapped_list_of_paths, dataset_name, ranking_number):
    # load the image data from saved txt files
    unique_id = list(np.genfromtxt(unique_id_file, dtype=None))
    id_all = list(np.genfromtxt(id_all_file, dtype=None))
    fullpath_image_names = list(np.genfromtxt(swapped_list_of_paths, dtype=None))

    # -- Next we want to randomly select a ranking set and a training set where an ID can only be in one of the sets.
    # select at random a subset for ranking by drawing indices from a uniform distribution
    start = rd.randrange(0, len(unique_id) - ranking_number)
    stop = start + ranking_number
    # get the matching start and stop indices to determine where to slice the list
    index_start = id_all.index(unique_id[start])
    index_stop = id_all.index(unique_id[stop])
    # slice the list to create a set for ranking and a set for training
    ranking_ids_pos = fullpath_image_names[index_start:index_stop]

    # get the chosen unique IDs and all IDs
    ranking_ids = unique_id[start:stop]
    all_ranking_ids = id_all[index_start:index_stop]

    # -- Create combinations and store the positive matches for ranking
    # select upper bound for images per ID. Has to be 2 for ranking.
    upper_bound = 2
    ranking_ids_pos, min_group_size_rank, ranking_ids = pre_selection(ranking_ids_pos, ranking_ids, all_ranking_ids,
                                                                      upper_bound, dataset_name)
    ranking_ids_pos = make_positive_combinations(ranking_ids_pos, ranking_ids, upper_bound, min_group_size_rank)

    # shuffle so that each time we get different first occurences for when making negative pairs
    rd.shuffle(ranking_ids_pos)

    return ranking_ids_pos


def make_positive_pairs(id_all_file, unique_id_file, swapped_list_of_paths, dataset_name, ranking_number):
    """
    Creates positive labeled pairs for training and ranking set.
    This needs to be done once at the beginning of the iteration.

    :param id_all_file:                 string of path to id_all_file.txt
    :param unique_id_file:              string of path to unique_id_file.txt
    :param swapped_list_of_paths:       string of path to swapped_list_of_paths.txt
    :param dataset_name:                string name of the dataset
    :param ranking_number:              int the ranking number
    :return:                            two lists, containing labeled pairs for training and ranking respectively
    """
    # load the image data from saved txt files
    unique_id = list(np.genfromtxt(unique_id_file, dtype=None))
    id_all = list(np.genfromtxt(id_all_file, dtype=None))
    fullpath_image_names = list(np.genfromtxt(swapped_list_of_paths, dtype=None))

    # -- Next we want to randomly select a ranking set and a training set where an ID can only be in one of the sets.
    # select at random a subset for ranking by drawing indices from a uniform distribution
    start = rd.randrange(0, len(unique_id) - ranking_number)
    stop = start + ranking_number
    # get the matching start and stop indices to determine where to slice the list
    index_start = id_all.index(unique_id[start])
    index_stop = id_all.index(unique_id[stop])
    # slice the list to create a set for ranking and a set for training
    ranking_ids_pos = fullpath_image_names[index_start:index_stop]

    # training_ids_pos = fullpath_image_names[0:index_start] + fullpath_image_names[index_start:]
    training_ids_pos = fullpath_image_names[0:index_start] + fullpath_image_names[index_stop:]

    # get the chosen unique IDs and all IDs
    train_ids = unique_id[0:start] + unique_id[stop:]
    ranking_ids = unique_id[start:stop]
    all_train_ids = id_all[0:index_start] + id_all[index_stop:]
    all_ranking_ids = id_all[index_start:index_stop]

    # -- Create combinations and store the positive matches for ranking
    # select upper bound for images per ID. Has to be 2 for ranking.
    upper_bound = 2
    ranking_ids_pos, min_group_size_rank, ranking_ids = pre_selection(ranking_ids_pos, ranking_ids, all_ranking_ids,
                                                                      upper_bound, dataset_name)
    ranking_ids_pos = make_positive_combinations(ranking_ids_pos, ranking_ids, upper_bound, min_group_size_rank)

    # -- Create combinations and store the positive matches for training
    # You could increase this but then you'll get a lot more data
    upper_bound = 3
    training_ids_pos, min_group_size_train, train_ids = pre_selection(training_ids_pos, train_ids, all_train_ids,
                                                                      upper_bound, dataset_name)

    training_ids_pos = make_positive_combinations(training_ids_pos, train_ids, upper_bound, min_group_size_train,
                                                  type='train')

    # shuffle so that each time we get different first occurences for when making negative pairs
    rd.shuffle(ranking_ids_pos)
    rd.shuffle(training_ids_pos)

    return ranking_ids_pos, training_ids_pos


def make_negative_pairs(pos_list, the_type):
    """
    Creates negative labeled pairs for training and ranking set.
    :param pos_list:    list of the positive pairs
    :param the_type:    string 'ranking' or 'training'
    :return:            if ranking, return list with both positive and negative ranking pairs
                        elif training, return list with negative pairs
    """
    # split the positive list into its first 2 columns
    list_0 = [pos_list[index].split(',')[0] for index in range(len(pos_list))]
    list_1 = [pos_list[index].split(',')[1] for index in range(len(pos_list))]

    if the_type == 'ranking':
        ranking_list = []
        # ranking_neg = []
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                num = 1 if img0 == img1 else 0
                line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
                ranking_list.append(line)
                # if num == 0:
                #     ranking_neg.append(line)
        return ranking_list

    elif the_type == 'training':
        training_neg = []
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                if img0 == img1:
                    pass
                else:
                    line = list_0[img0] + ',' + list_1[img1] + ',0\n'
                    training_neg.append(line)

        return training_neg


def make_pairs_image(adjustable, project_data_storage, fixed_path, do_ranking, do_training, name, ranking_variable):
    """
    Makes pairs for the specified image dataset.
    :param adjustable:              object of class ProjectVariable
    :param project_data_storage:    string path to dataset processed data
    :param fixed_path:              string path to fixed dataset
    :param do_ranking:              bool
    :param do_training:             bool
    :param name:                    string name of the dataset
    :param ranking_variable:        the variable from which to read the ranking number.
                                    can be adjustable.ranking_number_train or adjustable.ranking_number_test
    :return:                        3 lists containing labeled pairs, one list for ranking and two for training
    """
    if not os.path.exists(project_data_storage):
        os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    swapped_list_of_paths = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')

    if not os.path.exists(id_all_file):
        make_image_data_files(fixed_path, project_data_storage)

    if ranking_variable == 'half':
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(ranking_variable, int):
        ranking_number = ranking_variable
    else:
        ranking_number = None

    if do_ranking is True and do_training is True:
        ranking_pos, training_pos = make_positive_pairs(id_all_file, unique_id_file, swapped_list_of_paths,
                                                        name, ranking_number)

        ranking = make_negative_pairs(ranking_pos, 'ranking')
        training_neg = make_negative_pairs(training_pos, 'training')

        # save if specified
        if adjustable.save_inbetween and adjustable.iterations == 1:
            file_name = '%s_ranking_%s.txt' % (name, adjustable.use_gpu)
            file_name = os.path.join(pc.SAVE_LOCATION_RANKING_FILES, file_name)
            pu.write_to_file(file_name, ranking)

    elif do_ranking is False and do_training is False:
        print('Error: ranking and training cannot both be false')
        return
    elif do_ranking is False and do_training is True:
        # only train, only make the training files using all the data
        training_pos = make_positive_pairs_training(id_all_file, unique_id_file, swapped_list_of_paths, name)
        training_neg = make_negative_pairs(training_pos, 'training')

        ranking = None
    elif do_ranking is True and do_training is False:
        # only test, only make the ranking file
        # check first if there exists a ranking file of the correct name, load from it
        ranking_file = '../ranking_files/%s_ranking_%s.txt' % (name, adjustable.use_gpu)
        if os.path.exists(ranking_file):
            print('Loading ranking from file: `%s`' % ranking_file)
            ranking = list(np.genfromtxt(ranking_file, dtype=None))
        else:
            ranking_pos = make_positive_pairs_ranking(id_all_file, unique_id_file, swapped_list_of_paths, name,
                                                      ranking_number)
            ranking = make_negative_pairs(ranking_pos, 'ranking')

        training_pos = None
        training_neg = None

    else:
        print('Error: some kind of dark magic happened')
        return

    return ranking, training_pos, training_neg


def make_pairs_cuhk2(adjustable, do_ranking, do_training, ranking_variable):
    """
    Makes pairs for CUHK02.
    This is needed because CUHK02 has 5 subdirectories.
    --
    TODO: If someone has time, just merge together the separate subdirectories and use method `make_image_pairs()` like
    how I did on the other image datasets.
    --

    :param adjustable:      object of class ProjectVariable
    :return:                3 lists contianing labeled pairs, one list for ranking and two for training
    """
    top_project_data_storage = '../data/CUHK02'
    subdirs = ['P1', 'P2', 'P3', 'P4', 'P5']
    num_subdirs = len(subdirs)
    name = 'cuhk02'

    # DONE TODO: fix adjustable.ranking_number
    # check if ranking_number is alright else fix it
    if ranking_variable == 'half':
        adapted_ranking_number = pc.RANKING_DICT['cuhk02'] / len(subdirs)
    elif isinstance(ranking_variable, int):
        if ranking_variable >= num_subdirs:
            if ranking_variable % num_subdirs != 0:
                print(
                'Error: cuhk02 ranking number must be divisible by %d.\n Recommended: change ranking number from %d to %d'
                % (num_subdirs, ranking_variable, ranking_variable / num_subdirs))
                return

            else:
                adapted_ranking_number = ranking_variable / num_subdirs
        elif ranking_variable < len(subdirs):
            print('Error: cuhk02 ranking number must be at least %d.\n Recommended: change ranking number from %d to %d'
                  % (num_subdirs, ranking_variable, num_subdirs))
            return
        else:
            adapted_ranking_number = None
    else:
        adapted_ranking_number = None

    ranking_all = []
    training_pos_all = []
    training_neg_all = []

    # for each subdirectory make positive and negative pairs
    for a_dir in subdirs:
        # DONE TODO: fix adjustable.ranking_number
        if ranking_variable == 'half':
            adapted_ranking_number = pc.RANKING_CUHK02_PARTS[a_dir]

        project_data_storage = os.path.join(top_project_data_storage, a_dir)

        if not os.path.exists(project_data_storage):
            os.mkdir(project_data_storage)

        id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
        unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
        swapped_list_of_paths = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')

        if not os.path.exists(id_all_file):
            make_image_data_cuhk2()

        if do_ranking is True and do_training is True:
            ranking_pos, training_pos = make_positive_pairs(id_all_file, unique_id_file, swapped_list_of_paths,
                                                            'cuhk02',
                                                            adapted_ranking_number)

            ranking = make_negative_pairs(ranking_pos, 'ranking')
            training_neg = make_negative_pairs(training_pos, 'training')

            ranking_all += ranking
            training_pos_all += training_pos
            training_neg_all += training_neg
        elif do_ranking is False and do_training is False:
            print('Error: ranking and training cannot both be false')
            return
        elif do_ranking is False and do_training is True:
            # only train, only make the training files using all the data
            training_pos = make_positive_pairs_training(id_all_file, unique_id_file, swapped_list_of_paths, name)
            training_neg = make_negative_pairs(training_pos, 'training')
            training_pos_all += training_pos
            training_neg_all += training_neg
            rank_all = None
        elif do_ranking is True and do_training is False:
            # only test, only make the ranking file
            # check first if there exists a ranking file of the correct name, load from it
            ranking_file = '../ranking_files/%s_ranking_%s.txt' % (name, adjustable.use_gpu)
            if not os.path.exists(ranking_file):
                ranking_pos = make_positive_pairs_ranking(id_all_file, unique_id_file, swapped_list_of_paths, name,
                                                          adapted_ranking_number)
                ranking = make_negative_pairs(ranking_pos, 'ranking')
                ranking_all += ranking
            else:
                rank_all = None
            training_pos_all = None
            training_neg_all = None
        else:
            print('Error: voodoo')
            return

    # fix the ranking
    if do_ranking is True and do_training is False:
        # only test, only make the ranking file
        # check first if there exists a ranking file of the correct name, load from it
        ranking_file = '../ranking_files/%s_ranking_%s.txt' % (name, adjustable.use_gpu)
        if os.path.exists(ranking_file):
            print('Loading ranking from file: `%s`' % ranking_file)
            rank_all = list(np.genfromtxt(ranking_file, dtype=None))
        else:
            # merge together the ranking pairs in the proper way: such that we have N IDs and N^2 pairs where N pairs
            # are positive
            rank_all = []
            for item in ranking_all:
                the_label = int(item.strip().split(',')[-1])
                if the_label == 1:
                    rank_all.append(item)

            list_0 = [rank_all[index].split(',')[0] for index in range(len(rank_all))]
            list_1 = [rank_all[index].split(',')[1] for index in range(len(rank_all))]

            rank_all = []
            for img0 in range(len(list_0)):
                for img1 in range(len(list_1)):
                    num = 1 if img0 == img1 else 0
                    line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
                    rank_all.append(line)

    elif do_ranking is True and do_training is True:
        # merge together the ranking pairs in the proper way: such that we have N IDs and N^2 pairs where N pairs
        # are positive
        rank_all = []
        for item in ranking_all:
            the_label = int(item.strip().split(',')[-1])
            if the_label == 1:
                rank_all.append(item)

        list_0 = [rank_all[index].split(',')[0] for index in range(len(rank_all))]
        list_1 = [rank_all[index].split(',')[1] for index in range(len(rank_all))]

        rank_all = []
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                num = 1 if img0 == img1 else 0
                line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
                rank_all.append(line)

        # save if specified
        if adjustable.save_inbetween and adjustable.iterations == 1:
            file_name = '%s_ranking_%s.txt' % (name, adjustable.use_gpu)
            file_name = os.path.join(pc.SAVE_LOCATION_RANKING_FILES, file_name)
            pu.write_to_file(file_name, rank_all)
    else:
        rank_all = None

    return rank_all, training_pos_all, training_neg_all


def read_plot_from_hdf5(swapped_list, h5_path):
    """
    Check if the data was loaded correctly by plotting it. To see plots, run in debug mode with
    breaking point at plt.imshow(thing).

    :param swapped_list:    string of path to swapped_list_of_paths.txt
    :param h5_path:         string of path to h5 file
    """
    hdf5_file = h5py.File(h5_path, 'r')
    list_of_paths = list(np.genfromtxt(swapped_list, dtype=None))
    for i in range(10):
        thing = hdf5_file[list_of_paths[i]][:]
        plt.imshow(thing)


def get_sequence_length_video_dataset(name):
    """
    Print the minimum sequence length and the maximum sequence length of the dataset,
    :param name:    string name of the video dataset: 'ilids-vid' or 'prid2011'
    """
    number_of_persons = [0, 0]
    min_seq_len = 100000000
    max_seq_len = 0

    ind = 0

    path = '/home/gabi/Documents/datasets/%s' % name
    cams = os.listdir(path)
    print(cams)
    for cam in cams:
        cam_path = os.path.join(path, cam)
        persons = os.listdir(cam_path)
        number_of_persons[ind] = len(persons)
        ind += 1
        for person in persons:
            person_path = os.path.join(cam_path, person)
            images = os.listdir(person_path)
            number_images = len(images)
            if number_images < min_seq_len:
                min_seq_len = number_images
            if number_images > max_seq_len:
                max_seq_len = number_images

    print('dataset: %s\nnumber of persons: %s\nmin seq length: %d\nmax seq length: %d'
          % (name, str(number_of_persons), min_seq_len, max_seq_len))


def make_video_data_files(name):
    """
    Creates 4 text files, and saves them in the dataset directory:
    id_all.txt                      contains the all unique IDs in order
    unique_id.txt                   the set of id_all.txt (so no duplicates)
    fullpath_sequence_names.txt     full path to sequences
    swapped_fullpath_names.txt      fullpath_sequence_names.txt but with '+' instead of '/'

    :param name:                    string name of the video dataset: 'ilids-vid' or 'prid2011'
    """
    path = '/home/gabi/Documents/datasets/%s-fixed' % name

    data_folder = '../data/%s' % name
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    list_unique_sequences = []
    list_id_all = []
    list_id_as_int = []
    list_swapped_unique = []

    cams = sorted(os.listdir(path))

    for cam in cams:
        cam_path = os.path.join(path, cam)
        persons = os.listdir(cam_path)
        for person in persons:
            person_path = os.path.join(cam_path, person)
            sequences = os.listdir(person_path)

            for sequence in sequences:
                sequence_path = os.path.join(person_path, sequence)
                # fullpath
                list_unique_sequences.append(sequence_path)
                # all_id
                the_id = sequence_path.strip().split('/')[-2].split('_')[-1]
                list_id_all.append(the_id)
                list_id_as_int.append(int(the_id))
                # swapped
                swapped = pu.swap_for(sequence_path, '/', '+')
                list_swapped_unique.append(swapped)

    zipped = zip(list_id_as_int, list_id_all, list_unique_sequences, list_swapped_unique)
    zipped.sort()
    list_id_as_int, list_id_all, list_unique_sequences, list_swapped_unique = zip(*zipped)

    # look for the IDs that only have 1 sequence. these won't be included in the end dataset because
    # we cannot make pairs with them
    loners = []

    for item in range(len(list_id_all)):
        # look at the id before and after it
        before = item - 1
        after = item + 1
        if not item == 0 and not item == (len(list_id_all) - 1):
            if not (list_id_all[item] == list_id_all[before] or list_id_all[item] == list_id_all[after]):
                loners.append(item)
        elif item == 0:
            if not list_id_all[item] == list_id_all[after]:
                loners.append(item)
        elif item == (len(list_id_all) - 1):
            if not list_id_all[item] == list_id_all[before]:
                loners.append(item)

    fullpath_unique_sequences = '../data/%s/fullpath_sequence_names.txt' % name
    id_all = '../data/%s/id_all.txt' % name
    swapped_unique = '../data/%s/swapped_fullpath_names.txt' % name

    lists = [list_unique_sequences, list_id_all, list_swapped_unique]
    files = [fullpath_unique_sequences, id_all, swapped_unique]

    numm = len(files)

    for i in range(numm):

        list_thing = lists[i]
        file_thing = files[i]

        with open(file_thing, 'w') as my_file:
            num = len(list_thing)
            for item in range(num):
                if item not in loners:
                    my_file.write('%s\n' % list_thing[item])

    list_id_all = list(np.genfromtxt(id_all, dtype=None))
    unique_id = list(set(list_id_all))
    unique_id.sort()

    id_unique = '../data/%s/unique_id.txt' % name

    with open(id_unique, 'w') as my_file:
        for item in range(len(unique_id)):
            my_file.write('%s\n' % unique_id[item])


# DONE TODO: adjustable.datasets
def make_pairs_video(adjustable, project_data_storage, fixed_path, do_ranking, do_training, name, ranking_variable):
    """
    Makes pairs for the specified video dataset.
    :param adjustable:              object of class ProjectVariable
    :param project_data_storage:    string path to dataset processed data
    :param fixed_path:              string path to fixed dataset
    :param do_ranking:              bool
    :param do_training:             bool
    :param name:                    string name of the dataset
    :param ranking_variable:        the variable from which to read the ranking number.
                                    can be adjustable.ranking_number_train or adjustable.ranking_number_test
    :return:                        3 lists containing labeled pairs, one list for ranking and two for training
    """
    if not os.path.exists(project_data_storage):
        os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id.txt')
    swapped_fullpath_file = os.path.join(project_data_storage, 'swapped_fullpath_names.txt')

    if not os.path.exists(id_all_file):
        make_video_data_files(fixed_path)

    # DONE TODO: fix adjustable.ranking_number
    if ranking_variable == 'half':
        # TODO: implement this for video data
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(ranking_variable, int):
        ranking_number = ranking_variable
    else:
        ranking_number = None

    if do_ranking is True and do_training is True:
        ranking_pos, training_pos = make_positive_pairs(id_all_file, unique_id_file, swapped_fullpath_file,
                                                        name, ranking_number)

        ranking = make_negative_pairs(ranking_pos, 'ranking')
        training_neg = make_negative_pairs(training_pos, 'training')

        # save if specified
        if adjustable.save_inbetween and adjustable.iterations == 1:
            file_name = '%s_ranking_%s.txt' % (name, adjustable.use_gpu)
            file_name = os.path.join(pc.SAVE_LOCATION_RANKING_FILES, file_name)
            pu.write_to_file(file_name, ranking)

    elif do_ranking is False and do_training is False:
        print('Error: ranking and training cannot both be false')
        return
    elif do_ranking is False and do_training is True:
        # only train, only make the training files using all the data
        training_pos = make_positive_pairs_training(id_all_file, unique_id_file, swapped_fullpath_file, name)
        training_neg = make_negative_pairs(training_pos, 'training')

        ranking = None
    elif do_ranking is True and do_training is False:
        # only test, only make the ranking file
        # check first if there exists a ranking file of the correct name, load from it
        ranking_file = '../ranking_files/%s_ranking_%s.txt' % (name, adjustable.use_gpu)
        if os.path.exists(ranking_file):
            print('Loading ranking from file: `%s`' % ranking_file)
            ranking = list(np.genfromtxt(ranking_file, dtype=None))
        else:
            ranking_pos = make_positive_pairs_ranking(id_all_file, unique_id_file, swapped_fullpath_file, name,
                                                      ranking_number)
            ranking = make_negative_pairs(ranking_pos, 'ranking')

        training_pos = None
        training_neg = None

    else:
        print('Error: some kind of dark magic happened')
        return

    return ranking, training_pos, training_neg


def make_human_data(fixed_path, label, storage_path):
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    all_keys = [os.path.join(fixed_path, item) for item in os.listdir(fixed_path)]
    all_swapped_keys = [pu.swap_for(item, '/', '+') for item in all_keys]

    fullpath = os.path.join(storage_path, 'fullpath.txt')
    swapped = os.path.join(storage_path, 'swapped.txt')

    len_files = len(all_keys)

    if os.path.exists(fullpath):
        action = 'a'
    else:
        action = 'w'

    with open(fullpath, action) as fp:
        with open(swapped, action) as sw:
            for item in range(len_files):
                fp.write(all_keys[item] + ',%d' % label + '\n')
                sw.write(all_swapped_keys[item] + ',%d' % label + '\n')


def make_inria_data():
    make_human_data('/home/gabi/Documents/datasets/INRIAPerson/fixed-pos', 1, '../data/INRIA')
    make_human_data('/home/gabi/Documents/datasets/INRIAPerson/fixed-neg', 0, '../data/INRIA')


def find_matching_angles():
    # viper
    # path_a = '/home/gabi/Documents/datasets/VIPeR/cam_a'
    # path_b = '/home/gabi/Documents/datasets/VIPeR/cam_b'
    # grid
    path_a = '/home/gabi/Documents/datasets/GRID/probe'
    path_b = '/home/gabi/Documents/datasets/GRID/valid-gallery'

    list_a = os.listdir(path_a)
    list_b = os.listdir(path_b)

    list_a.sort()
    list_b.sort()

    len_data = len(list_a)

    count = 0

    for item in range(len_data):
        # viper
        # angle_a = list_a[item].split('.')[0].split('_')[-1]
        # angle_b = list_b[item].split('.')[0].split('_')[-1]
        num_a = list_a[item].split('.')[0].split('_')[1]
        num_b = list_b[item].split('.')[0].split('_')[1]

        id_a = list_a[item].split('.')[0].split('_')[0]
        id_b = list_b[item].split('.')[0].split('_')[0]

        # viper
        # if angle_a == angle_b:
        if id_a == id_b:
            if num_a in ['1', '2'] and num_b in ['1', '2']:
                print(list_a[item], list_b[item])
                count += 1

    print('Number of matching angles: %d' % count)


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
    if list_of_datasets is None:
        return None
    else:
        h5_data = []

        if isinstance(list_of_datasets, str):
            h5_data.append(get_dataset(list_of_datasets))
        else:
            for dataset in list_of_datasets:
                h5_data.append(get_dataset(dataset))

        return h5_data


def create_training_and_ranking_set(name, adjustable, ranking_variable, do_ranking=True, do_training=True):
    """ Do this at the beginning of each iteration
    """
    if name == 'viper':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.VIPER_DATA_STORAGE, pc.VIPER_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'cuhk01':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.CUHK01_DATA_STORAGE, pc.CUHK01_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'cuhk02':
        ranking, training_pos, training_neg = make_pairs_cuhk2(adjustable, do_ranking, do_training, ranking_variable)
    elif name == 'market':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.MARKET_DATA_STORAGE, pc.MARKET_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'caviar':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.CAVIAR_DATA_STORAGE, pc.CAVIAR_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'grid':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.GRID_DATA_STORAGE, pc.GRID_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'prid450':
        ranking, training_pos, training_neg = make_pairs_image(adjustable, pc.PRID450_DATA_STORAGE, pc.PRID450_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'ilids-vid':
        ranking, training_pos, training_neg = make_pairs_video(adjustable, pc.ILIDS_DATA_STORAGE, pc.ILIDS_FIXED,
                                                               do_ranking, do_training, name, ranking_variable)
    elif name == 'prid2011':
        ranking, training_pos, training_neg = make_pairs_video(adjustable, pc.PRID2011_DATA_STORAGE,
                                                               pc.PRID2011_FIXED, do_ranking, do_training, name,
                                                               ranking_variable)
    else:
        ranking, training_pos, training_neg = None, None, None

    return ranking, training_pos, training_neg


def merge_datasets(adjustable, list_training_pos, list_training_neg):
    """ Merges specified datasets by shuffling together the positive and negative training instances.
        There will be many more negative instances than positive instances. This method needs to be excecuted
        once, right after 'create_trainin_and_ranking_set()'.

    :param adjustable:              object of class ProjectVariable created in 'running_experiments.py'
    :param list_training_pos:       A list of all training positive instances of the different datasets
    :param list_training_neg:       A list of all training negative instances of the different datasets
    :return:                        Two lists composed of the specified compositions of the data for negative
                                    and positive instances.
    """
    check = 0
    if list_training_pos == [] or list_training_pos is None:
        print('Warning: No positive training list given')
        check += 1
    if list_training_neg == [] or list_training_neg is None:
        print('Warning: No negative training list given')
        check += 1
    if check == 2:
        if adjustable.only_test is True:
            return None, None
        else:
            print('Error: No training list given. I quit!')
            return

    if len(list_training_pos) == len(list_training_neg):
        number_of_datasets = len(list_training_neg)
    else:
        print('Error: the number of datasets is inconsistent')
        return

    if adjustable.only_test == True:
        # only test, nothing to do
        merged_training_pos = None
        merged_training_neg = None
    else:
        # train
        merged_training_pos = []
        merged_training_neg = []

        if number_of_datasets == 0:
            print('Error: no training datasets have been specified')
            return
        elif number_of_datasets == 1:
            # can be train + test on only 1 dataset
            # can be only train on 1 dataset
            # mixing doesn't matter
            merged_training_pos = list_training_pos[0]
            merged_training_neg = list_training_neg[0]
            random.shuffle(merged_training_pos)
            random.shuffle(merged_training_neg)
        else:
            # can be train + test on multiple datasets
            # can be only train on multiple datasets
            # mixing does matter
            if adjustable.mix == True:
                # shuffle the data with each other
                # here we need to know if we only train or train+test
                if adjustable.dataset_test is None:
                    # only train, shuffle the data
                    # choice: don't balance
                    for index in range(number_of_datasets):
                        merged_training_pos += list_training_pos[index]
                        merged_training_neg += list_training_neg[index]
                    random.shuffle(merged_training_pos)
                    random.shuffle(merged_training_neg)
                else:
                    if adjustable.mix_with_test == True:
                        # mix with the test
                        for index in range(number_of_datasets):
                            merged_training_pos += list_training_pos[index]
                            merged_training_neg += list_training_neg[index]
                        random.shuffle(merged_training_pos)
                        random.shuffle(merged_training_neg)
                    else:
                        # don't mix with the test (which is at the end)
                        for index in range(number_of_datasets - 1):
                            merged_training_pos += list_training_pos[index]
                            merged_training_neg += list_training_neg[index]
                        random.shuffle(merged_training_pos)
                        random.shuffle(merged_training_neg)

                        # note: make this a 2 dimensional list because the order matters
                        merged_training_pos = [merged_training_pos]
                        merged_training_neg = [merged_training_neg]

                        merged_training_pos.append(list_training_pos[-1])
                        merged_training_neg.append(list_training_neg[-1])
            else:
                # train in order.
                # number of datasets don't matter
                merged_training_pos = list_training_pos
                merged_training_neg = list_training_neg
                # for index in range(number_of_datasets):
                #     # note: make this a 2 dimensional list because the order matters
                #     merged_training_pos.append(list_training_pos[index])
                #     merged_training_neg.append(list_training_neg[index])

    return merged_training_pos, merged_training_neg


def get_dataset_to_map(name, data_list, data_names):
    """
    Get the dataset
    :param name:            string, name of the dataset folder in ../data
    :param data_list:       list with h5py object(s) containing the data
    :param data_names:      list of strings containing path of the physical location of the h5py object on disk
    :return:                h5py object containing a dataset
    """

    # get hdf dataset filename, used only for pairs of images
    # split on '/' get second to last item
    # match with name
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

    # get the index at which the correct path to the h5_object is stored
    index_in_data_names_for_dataset = data_names.index(dataset)
    # fetch the h5_object given the index at which it is stored
    h5_object = data_list[index_in_data_names_for_dataset]

    return h5_object


def create_key_dataset_mapping(key_list, h5_dataset_list):
    """ Creates a mapping from the keys to the datasets.
    :param key_list:            list of keys in form of tuples with a label "img1,img2,1"
    :param h5_dataset_list:     list of the h5 datasets to search in
    :return:                    dictionary, a mapping from the keys to the datasets
    """
    key_dataset_mapping = {}

    if len(h5_dataset_list) == 1:

        for key in key_list:
            key_1 = key.split(',')[0]
            key_2 = key.split(',')[1]

            key_dataset_mapping[key_1] = h5_dataset_list[0]
            key_dataset_mapping[key_2] = h5_dataset_list[0]

    else:
        # get the physical location storing the h5 datasets
        h5_filenames = []
        for item in h5_dataset_list:
            # split to obtain the dataset folder
            the_filename = item.file.filename.split('/')[-2]
            the_filename = str(the_filename)
            h5_filenames.append(the_filename)

        for key in key_list:
            key_1 = key.split(',')[0]
            key_2 = key.split(',')[1]

            # split the key to get the dataset folder
            folder_key_1 = key_1.split('+')[-2]
            folder_key_2 = key_2.split('+')[-2]

            # get the h5 object containing the dataset for key_n
            dataset_key_1 = get_dataset_to_map(folder_key_1, h5_dataset_list, h5_filenames)
            dataset_key_2 = get_dataset_to_map(folder_key_2, h5_dataset_list, h5_filenames)

            key_dataset_mapping[key_1] = dataset_key_1
            key_dataset_mapping[key_2] = dataset_key_2

    return key_dataset_mapping


def grab_em_by_the_keys(key_list, training_h5, testing_h5):
    # def grab_em_by_the_keys(key_list, h5_dataset_list):
    """ Returns a training set
    :param key_list:                list of keys
    :param training_h5:             list of string with paths to h5 datasets
    :param testing_h5:              list of string with paths to h5 datasets
    :return:
    """

    h5_dataset_list = []

    if training_h5 is not None:
        for index in range(len(training_h5)):
            h5_dataset_list.append(training_h5[index])

    if testing_h5 is not None:
        for index in range(len(testing_h5)):
            h5_dataset_list.append(testing_h5[index])

    # create mapping from keys to dataset
    key_dataset_mapping = create_key_dataset_mapping(key_list, h5_dataset_list)

    ################################################################################################################
    #   isolate the different keys and values
    ################################################################################################################
    all_key_1 = []
    all_key_2 = []
    for item in key_list:
        all_key_1.append(item.split(',')[0])
        all_key_2.append(item.split(',')[1])

    ################################################################################################################
    #   get the values from the h5 file
    ################################################################################################################
    values_key_1 = []
    values_key_2 = []
    for index in range(len(key_list)):
        dataset_h5_object = key_dataset_mapping[all_key_1[index]]
        the_image = dataset_h5_object[all_key_1[index]][:]
        values_key_1.append(the_image)

        dataset_h5_object = key_dataset_mapping[all_key_2[index]]
        the_image = dataset_h5_object[all_key_2[index]][:]
        values_key_2.append(the_image)

    return np.asarray((values_key_1, values_key_2))


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
        rank_ordered_partitions = [
            this_ranking[item * pc.RANKING_DICT['cuhk02'] + item].strip().split(',')[0].split('+')[-3]
            for item in range(pc.RANKING_DICT['cuhk02'])]
        rank_ordered_ids = [
            pu.my_join(
                list(this_ranking[item * pc.RANKING_DICT['cuhk02'] + item].strip().split(',')[0].split('+')[-1])[0:4])
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
            index = randint(0, len(unique_ids) - 1)
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
        if adjustable.ranking_number_test == 'half':
            ranking_number = pc.RANKING_DICT[name_dataset]
        elif isinstance(adjustable.ranking_number_test, int):
            ranking_number = adjustable.ranking_number_test
        else:
            print("ranking_number_test must be 'half' or an int")
            return

            # print('ranking number: %s' % str(ranking_number))

            # rank_ordered_ids = [
            # dp.my_join(list(this_ranking[item * ranking_number + item].strip().split(',')[0].split('+')[-1])[0:4])
            # for item in range(ranking_number)]
        rank_ordered_ids = []

        # for thingy in this_ranking:
        #     print(thingy)

        for item in range(ranking_number):
            a = list(this_ranking[item * ranking_number + item].strip().split(',')[0].split('+')[-1])
            # print('a: %s' % str(a))
            b = pu.my_join(a[0:4])
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
            index = randint(0, len(unique_ids) - 1)
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
        thingy.append(h5data[key][:, :, 0:3])

    thingy = np.asarray(thingy)

    return thingy
