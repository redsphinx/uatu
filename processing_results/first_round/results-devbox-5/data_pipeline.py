"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

File description:

File has multiple functions:

1) Transforms raw images and video to the specified format that we need to feed the network.
2) Creates pairs of images and videos and stored as text data
3) Saves image and video data to HDF5
"""
import numpy as np
import os
import random as rd
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
from shutil import copyfile
from PIL import Image

import project_constants as pc


def fix_viper():
    """
    Assuming the VIPeR data is already downloaded and extracted, put all images in a single folder and pad the
    width with zeros.
    """
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'

    if not os.path.exists(padded_folder_path):
        os.mkdir(padded_folder_path)

    cams = ['cam_a', 'cam_b']
    for folder in cams:
        cam_path = os.path.join(original_folder_path, str(folder))
        padded_cam_path = padded_folder_path
        num = 'a' if folder == 'cam_a' else 'b'
        for the_file in os.listdir(cam_path):
            img = Image.open(os.path.join(cam_path, the_file))
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))

            img_width, img_height = img.size
            new_img_width, new_img_height = new_img.size
            padding_width = (new_img_width-img_width)/2
            padding_height = (new_img_height-img_height)/2

            new_img.paste(img, box=(padding_width, padding_height))

            filename = the_file.split('_')[0] + '_' + str(num) + '.bmp'
            filename = os.path.join(padded_cam_path, filename)
            new_img.save(filename)


def fix_cuhk01():
    """
    Assuming the CUHK01 data is already downloaded and extracted, put all images in a single folder and resizes
    the images from 160x60 to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK1'
    new_path = os.path.dirname(folder_path)

    list_images = os.listdir(folder_path)
    name_folder = folder_path.split('/')[-1]
    new_folder_path = os.path.join(new_path, 'cropped_' + str(name_folder))
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for image_path in list_images:
        img = Image.open(os.path.join(folder_path, image_path))
        img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
        img.save(os.path.join(new_folder_path, image_path))


def fix_cuhk02():
    """
    Assuming the CUHK02 data is already downloaded and extracted, put all images in a single folder and resizes
    the images from 160x60 to 128x64. Notice the weird layout of the folder structure. We leave the dataset partitioned
    in 5 parts.
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK2'
    cropped_folder_path = os.path.join(os.path.dirname(folder_path), 'cropped_CUHK2')
    if not os.path.exists(cropped_folder_path):
        os.mkdir(cropped_folder_path)

    subdirs = os.listdir(folder_path)

    for a_dir in subdirs:
        if not os.path.exists(os.path.join(cropped_folder_path, a_dir)):
            os.makedirs(os.path.join(cropped_folder_path, a_dir, 'all'))

    cameras = ['cam1', 'cam2']

    for a_dir in subdirs:
        for cam in cameras:
            original_images_path = os.path.join(folder_path, a_dir, cam)
            cropped_images_path = os.path.join(cropped_folder_path, a_dir, 'all')
            images = [a_file for a_file in os.listdir(original_images_path) if a_file.endswith('.png')]
            for ind in range(len(images)):
                image = os.path.join(original_images_path, images[ind])
                cropped_image = os.path.join(cropped_images_path, images[ind])
                img = Image.open(image)
                img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
                img.save(cropped_image)


def standardize(all_images, folder_path, fixed_folder_path, the_mod=None):
    """
    Assuming the images have been downloaded and extracted.
    Makes the images in the CAVIAR4REID, GRID and PRID450 in the correct size of 128x64
    :param all_images:          list of image names
    :param folder_path:         string, the directory of the extracted images
    :param fixed_folder_path:   string, the directory of the standardized images
    :param the_mod:             string, modifier to add to the image name, so `image_a` where `the_mod = '_a'`
    """

    def modify(name, a_mod):
        return name.split('.')[0].split('_')[-1] + a_mod + name.split('.')[-1]

    for image in all_images:
        original_image_path = os.path.join(folder_path, image)

        if the_mod is not None:
            modified_image_path = os.path.join(fixed_folder_path, modify(image, the_mod))
        else:
            modified_image_path = os.path.join(fixed_folder_path, image)
        the_image = Image.open(original_image_path)
        image_width, image_height = the_image.size

        if image_width < pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 1
        elif image_width > pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 2

        elif image_width < pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 3
        elif image_width > pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 4

        elif image_width < pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 1
        elif image_width > pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 2

        elif image_width == pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 2
        elif image_width == pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 1

        elif image_width == pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 5
        else:
            case = None

        # if dimensions are bigger than WIDTH, HEIGHT then resize
        # if dimensions are smaller then pad with zeros
        if case == 2:
            the_image = the_image.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
            the_image.save(modified_image_path)
        elif case == 1 or case == 3 or case == 4:
            if case == 3:
                the_image = the_image.resize((image_width, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
            elif case == 4:
                the_image = the_image.resize((pc.IMAGE_WIDTH, image_height), Image.ANTIALIAS)
            padding_width = (pc.IMAGE_WIDTH - the_image.size[0]) / 2
            padding_height = (pc.IMAGE_HEIGHT - the_image.size[1]) / 2
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))
            new_img.paste(the_image, box=(padding_width, padding_height))
            new_img.save(modified_image_path)
        elif case == 5:
            the_image.save(modified_image_path)


def fix_caviar():
    """
    Assuming the CAVIAR4REID data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/CAVIAR4REID/original'
    fixed_folder_path = os.path.join(os.path.dirname(folder_path), 'fixed_caviar')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    all_images = os.listdir(folder_path)
    standardize(all_images, folder_path, fixed_folder_path)


def fix_grid():
    """
    Assuming the GRID data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/GRID'
    probe = os.path.join(folder_path, 'probe')
    gallery = os.path.join(folder_path, 'gallery')

    probe_list = os.listdir(probe)
    gallery_list = os.listdir(gallery)

    # trim the gallery list and remove items with '0000' in the path, these are identities that do not belong to a pair
    proper_gallery_list = [item for item in gallery_list if not item[0:4] == '0000']

    fixed_folder_path = os.path.join(os.path.dirname(probe), 'fixed_grid')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    # standardize will put probe and gallery in the same fixed folder
    standardize(probe_list, probe, fixed_folder_path)
    standardize(proper_gallery_list, gallery, fixed_folder_path)


def fix_prid450():
    """
    Assuming the PRID450 data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/PRID450'
    cam_a = os.path.join(folder_path, 'cam_a')
    cam_b = os.path.join(folder_path, 'cam_b')

    cam_a_list = os.listdir(cam_a)
    cam_b_list = os.listdir(cam_b)

    # trim the dataset to contain only RGB color images
    proper_cam_a_list = [item for item in cam_a_list if item.split('_')[0] == 'img']
    proper_cam_b_list = [item for item in cam_b_list if item.split('_')[0] == 'img']

    fixed_folder_path = os.path.join(os.path.dirname(cam_a), 'fixed_prid')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    # standardize will put probe and gallery in the same fixed folder
    standardize(proper_cam_a_list, cam_a, fixed_folder_path, '_a.')
    standardize(proper_cam_b_list, cam_b, fixed_folder_path, '_b.')


def write_to_file(filepath, data):
    """
    Writes data to file
    :param filepath:    string path to file
    :param data:        a list with the data to write. list must be one-dimensional
    """
    with open(filepath, 'w') as myfile:
        for i in range(len(data)):
            myfile.write(str(data[i]) + '\n')


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

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)
    with open(swapped_list_of_paths, 'w') as myfile:
        for item in fullpath_image_names:
            item_name = swap_for(item, '/', '+')
            myfile.write(item_name + '\n')


def unique_id_and_all_images_cuhk2():
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

        write_to_file(id_all_file, id_all)
        write_to_file(unique_id_file, unique_id)
        write_to_file(short_image_names_file, short_image_names)
        write_to_file(fullpath_image_names_file, fullpath_image_names)
        with open(swapped_list_of_paths, 'w') as myfile:
            for item in fullpath_image_names:
                item_name = swap_for(item, '/', '+')
                myfile.write(item_name + '\n')


# def create_positive_combinations(fullpath, unique_ids, num, smallest_id_group):
#     """
#     Makes positive pairs for images (or sequences in the case of video data)
#     :param fullpath:            list of fullpath names
#     :param unique_ids:          list of unique IDs
#     :param num:                 int number of image per unique ID (sets a boundary)
#     :param smallest_id_group:   int size of the smallest set of images per ID
#     :return:                    a list of matching pairs with label 'item_a,item_b,1'
#     """
#     if num > smallest_id_group:
#         num = smallest_id_group
#     combo_list = []
#     num_ids = len(unique_ids)
#
#     if num == 3:
#         for item in range(num_ids):
#             thing = str(fullpath[num * item] + ',' + fullpath[num * item + 1] + ',1\n')
#             combo_list.append(thing)
#             thing = str(fullpath[num * item + 1] + ',' + fullpath[num * item + 2] + ',1\n')
#             combo_list.append(thing)
#     elif num == 2:
#         for item in range(num_ids):
#             thing = str(fullpath[num*item] + ',' + fullpath[num*item+1] + ',1\n')
#             combo_list.append(thing)
#     return combo_list


def create_positive_combinations(fullpath, unique_ids, num, smallest_id_group, type='rank', augment_equal=False):
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
    training_ids_pos = create_positive_combinations(training_ids_pos, train_ids, upper_bound, min_group_size_train,
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
    ranking_ids_pos = create_positive_combinations(ranking_ids_pos, ranking_ids, upper_bound, min_group_size_rank)

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
    start = rd.randrange(0, len(unique_id)-ranking_number)
    stop = start + ranking_number
    # get the matching start and stop indices to determine where to slice the list
    index_start = id_all.index(unique_id[start])
    index_stop = id_all.index(unique_id[stop])
    # slice the list to create a set for ranking and a set for training
    ranking_ids_pos = fullpath_image_names[index_start:index_stop]
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
    ranking_ids_pos = create_positive_combinations(ranking_ids_pos, ranking_ids, upper_bound, min_group_size_rank)

    # -- Create combinations and store the positive matches for training
    # You could increase this but then you'll get a lot more data
    upper_bound = 3
    training_ids_pos, min_group_size_train, train_ids = pre_selection(training_ids_pos, train_ids, all_train_ids,
                                                                      upper_bound, dataset_name)

    training_ids_pos = create_positive_combinations(training_ids_pos, train_ids, upper_bound, min_group_size_train,
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

    # DONE TODO: fix adjustable.ranking_number
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
            write_to_file(file_name, ranking)

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
                print('Error: cuhk02 ranking number must be divisible by %d.\n Recommended: change ranking number from %d to %d'
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
            unique_id_and_all_images_cuhk2()

        if do_ranking is True and do_training is True:
            ranking_pos, training_pos = make_positive_pairs(id_all_file, unique_id_file, swapped_list_of_paths, 'cuhk02',
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
            write_to_file(file_name, rank_all)
    else:
        rank_all = None

    return rank_all, training_pos_all, training_neg_all


def my_join(list_strings):
    """
    Makes 1 string from a list of strings
    :param list_strings:    list of strings
    :return:                string concatenated from all strings in the list_strings in order
    """
    awesome_string = ''
    for item in list_strings:
        awesome_string += item
    return awesome_string


def swap_for(the_thing, a, b):
    """
    Takes a string and swaps each character 'a' for character 'b', where the characters are specified
    :param the_thing:   full string
    :param a:           string
    :param b:           string
    :return:            string with swapped characters
    """
    the_thing = list(the_thing)
    for item in range(len(the_thing)):
        if the_thing[item] == a:
            the_thing[item] = b

    the_thing = my_join(the_thing)

    return str(the_thing)


def save_image_data_as_hdf5(file_list_of_paths, h5_path):
    """
    Saves image data as HDF5 (h5) file.
    :param file_list_of_paths:  string of path to 'fullpath_image_names_file.txt'
    :param h5_path:             string path to save location of h5 file
    """
    list_of_paths = list(np.genfromtxt(file_list_of_paths, dtype=None).tolist())
    swapped_list_of_paths = list(np.genfromtxt(os.path.join(os.path.dirname(file_list_of_paths),
                                                            'swapped_list_of_paths.txt')))

    action = 'a' if os.path.exists(h5_path) else 'w'

    with h5py.File(h5_path, action) as myfile:
        for item in range(len(list_of_paths)):
            myfile.create_dataset(name=swapped_list_of_paths[item], data=ndimage.imread(list_of_paths[item]))


def save_all_image_datasets_as_hdf5():
    """
    Saves all image datasets as HDF5 (h5) files.
    Run this method when you don't have the image data h5 files
    """
    save_image_data_as_hdf5('../data/GRID/fullpath_image_names_file.txt', '../data/GRID/grid.h5')
    print('saved grid')

    save_image_data_as_hdf5('../data/prid450/fullpath_image_names_file.txt', '../data/prid450/prid450.h5')
    print('saved prid450')

    save_image_data_as_hdf5('../data/caviar/fullpath_image_names_file.txt', '../data/caviar/caviar.h5')
    print('saved caviar')

    save_image_data_as_hdf5('../data/VIPER/fullpath_image_names_file.txt', '../data/VIPER/viper.h5')
    print('saved viper')

    subdirs = [item for item in os.listdir('../data/CUHK02') if not item.split('.')[-1] == 'txt']
    cuhk2_path = '../data/CUHK02'
    for a_dir in subdirs:
        the_full = os.path.join(cuhk2_path, a_dir, 'fullpath_image_names_file.txt')
        the_h5 = os.path.join(cuhk2_path, 'cuhk02.h5')
        save_image_data_as_hdf5(the_full, the_h5)
    print('saved cuhk02')

    save_image_data_as_hdf5('../data/CUHK/fullpath_image_names_file.txt', '../data/CUHK/cuhk01.h5')
    print('saved cuhk01')

    save_image_data_as_hdf5('../data/market/fullpath_image_names_file.txt', '../data/market/market.h5')
    print('saved market')


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


def fix_video_dataset(name, min_seq):
    """
    Make all sequences of same length and store them in a new folder
    :param name:        name of the video dataset: 'ilids-vid' or 'prid2011'
    :param min_seq:     minimal sequence length
    """
    old_path = '/home/gabi/Documents/datasets/%s' % name

    # make new directory
    new_path = '/home/gabi/Documents/datasets/%s-fixed' % name
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    # get the cameras
    cams = os.listdir(old_path)

    for cam in cams:
        if cam == 'cam1':
            cam_new = 'cam_a'
        elif cam == 'cam2':
            cam_new = 'cam_b'
        else:
            cam_new = cam
        new_cam_path = os.path.join(new_path, cam_new)
        if not os.path.exists(new_cam_path):
            os.mkdir(new_cam_path)
        old_cam_path = os.path.join(old_path, cam)
        persons = os.listdir(old_cam_path)

        # list the persons
        for person in persons:
            if len(person.split('_')) == 1:
                new_person = list(person)[-3:]
                new_person = my_join(new_person)
                new_person = int(new_person)
                new_person = 'person_%04d' % new_person
            else:
                new_person = person

            old_person_path = os.path.join(old_cam_path, person)
            images = sorted(os.listdir(old_person_path))

            # number of images in sequence
            number_images = len(images)

            # only continue if sequence has more than min_seq number of frames
            if number_images >= min_seq:
                new_person_path = os.path.join(new_cam_path, new_person)
                if not os.path.exists(new_person_path):
                    os.mkdir(new_person_path)

                possible_sequence_cuts = number_images / min_seq

                # depending on how many cuts we can make
                for sequence in range(possible_sequence_cuts):
                    sequence_path = os.path.join(new_person_path, 'sequence_%03d' % sequence)
                    if not os.path.exists(sequence_path):
                        os.mkdir(sequence_path)

                    sample_images = images[sequence * min_seq: min_seq + sequence * min_seq]
                    number_sample_images = len(sample_images)

                    for s_i in range(number_sample_images):
                        old_image = os.path.join(old_person_path, sample_images[s_i])
                        name_s_i = '%03d.png' % s_i
                        new_image = os.path.join(sequence_path, name_s_i)

                        # copy file
                        copyfile(old_image, new_image)
            else:
                print(old_person_path, number_images)


def fix_prid2011():
    """ turn into
        prid2011-fixed / cam_x / person_xxx / sequence_xxx / xxx.png
    """
    fix_video_dataset('prid2011', 20)


def fix_ilids():
    """ turn into
        ilids-vid-fixed / cam_x / person_xxx / sequence_xxx / xxx.png
    """
    fix_video_dataset('ilids-vid', 22)


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
                swapped = swap_for(sequence_path, '/', '+')
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
            write_to_file(file_name, ranking)

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


def save_video_as_hdf5(swapped_list, original_list,  h5_path):
    """
    Saves video data as HDF5 (h5) file
    :param swapped_list:    string path to swapped_fullpath_names.txt
    :param original_list:   string path to fullpath_sequence_names.txt
    :param h5_path:         string path to save location of h5 file
    """
    og_list_path = list(np.genfromtxt(original_list, dtype=None))
    list_of_paths = list(np.genfromtxt(swapped_list, dtype=None))

    with h5py.File(h5_path, 'w') as myfile:
        for item in range(len(list_of_paths)):
            # load all images in the sequence
            images = os.listdir(og_list_path[item])
            images.sort()
            len_images = len(images)
            image_arr = np.zeros((len_images, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
            for i in range(len_images):
                image_path = os.path.join(og_list_path[item], images[i])
                image_arr[i] = ndimage.imread(image_path)

            myfile.create_dataset(name=list_of_paths[item], data=image_arr)


def save_all_video_data_as_h5():
    """
    Saves all video datasets as HDF5 (h5) files.
    Run this method when you don't have the video data h5 files
    """
    names = ['ilids-vid', 'prid2011']

    for name in names:
        swapped_list = '../data/%s/swapped_fullpath_names.txt' % name
        og_list = '../data/%s/fullpath_sequence_names.txt' % name
        h5_path = '../data/%s/%s.h5' % (name, name)
        save_video_as_hdf5(swapped_list, og_list, h5_path)


def fix_inria():
    """
    Crops INRIA dataset images around center point and saves in new folder.
    Assuming the dataset is already downloaded.
    Since we are using these images to pre-train we will use all the images, so no specific testing/training split
    """
    # 2416 images
    path_train_positive = '/home/gabi/Documents/datasets/INRIAPerson/train_64x128_H96/pos'
    # 1218 images
    path_train_negative = '/home/gabi/Documents/datasets/INRIAPerson/train_64x128_H96/neg'
    # 1132 images
    path_test_positive = '/home/gabi/Documents/datasets/INRIAPerson/test_64x128_H96/pos'
    # 453 images
    path_test_negative = '/home/gabi/Documents/datasets/INRIAPerson/test_64x128_H96/neg'

    parent_path = '/home/gabi/Documents/datasets/INRIAPerson'
    fixed_positive = os.path.join(parent_path, 'fixed-pos')
    fixed_negative = os.path.join(parent_path, 'fixed-neg')

    if not os.path.exists(fixed_positive):
        os.mkdir(fixed_positive)
    if not os.path.exists(fixed_negative):
        os.mkdir(fixed_negative)

    # for positive images, we crop around the center
    for path in [path_train_positive, path_test_positive]:
        list_images = os.listdir(path)
        for image in list_images:
            old_image_path = os.path.join(path, image)
            new_image_path = os.path.join(fixed_positive, image)
            image_data = Image.open(old_image_path)
            width, height = image_data.size

            center_x = width / 2
            center_y = height / 2
            crop_x = center_x - pc.IMAGE_WIDTH / 2
            crop_y = center_y - pc.IMAGE_HEIGHT / 2

            cropped_image = image_data.crop((crop_x, crop_y, crop_x + pc.IMAGE_WIDTH, crop_y + pc.IMAGE_HEIGHT))
            cropped_image.save(new_image_path)

    # for negative images, we crop and do data augmentation by flipping the image horizontally
    for path in [path_train_negative, path_test_negative]:
        list_images = os.listdir(path)
        for image in list_images:
            old_image_path = os.path.join(path, image)
            new_image_path = os.path.join(fixed_negative, image)
            image_bare, suffix = image.split('.')
            flipped_path = os.path.join(fixed_negative, image_bare + '_flipped.' + suffix)
            image_data = Image.open(old_image_path)
            width, height = image_data.size

            center_x = width / 2
            center_y = height / 2
            crop_x = center_x - pc.IMAGE_WIDTH / 2
            crop_y = center_y - pc.IMAGE_HEIGHT / 2

            cropped_image = image_data.crop((crop_x, crop_y, crop_x + pc.IMAGE_WIDTH, crop_y + pc.IMAGE_HEIGHT))
            cropped_image.save(new_image_path)

            flipped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(flipped_path)


def make_human_data(fixed_path, label, storage_path):
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    all_keys = [os.path.join(fixed_path, item) for item in os.listdir(fixed_path)]
    all_swapped_keys = [swap_for(item, '/', '+') for item in all_keys]

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


def save_inria_data_as_hdf5():
    fullpath = '../data/INRIA/fullpath.txt'
    swapped = '../data/INRIA/swapped.txt'

    parent_dir = os.path.dirname(fullpath)
    h5_path = os.path.join(parent_dir, 'inria.h5')

    if not os.path.exists(fullpath):
        make_inria_data()

    fullpath_list = list(np.genfromtxt(fullpath, dtype=None))
    swapped_list = list(np.genfromtxt(swapped, dtype=None))

    len_files = len(fullpath_list)

    fullpath_list = [fullpath_list[item].split(',')[0] for item in range(len_files)]
    swapped_list = [swapped_list[item].split(',')[0] for item in range(len_files)]

    with h5py.File(h5_path, 'w') as myfile:
        for item in range(len_files):
            myfile.create_dataset(name=swapped_list[item], data=ndimage.imread(fullpath_list[item]))


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