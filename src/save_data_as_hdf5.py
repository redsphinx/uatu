"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

File contains methods save the data as hdf5 file
"""

import numpy as np
import os
import h5py
from scipy import ndimage
import project_constants as pc
import data_pipeline as dp


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


def save_prid2011_450_as_hdf5():
    name = 'prid2011_450'

    swapped_list = '../data/%s/swapped_fullpath_names.txt' % name
    og_list = '../data/%s/fullpath_sequence_names.txt' % name
    h5_path = '../data/%s/%s.h5' % (name, name)
    save_video_as_hdf5(swapped_list, og_list, h5_path)
    print('saved prid2011_450')


def save_inria_data_as_hdf5():
    fullpath = '../data/INRIA/fullpath.txt'
    swapped = '../data/INRIA/swapped.txt'

    parent_dir = os.path.dirname(fullpath)
    h5_path = os.path.join(parent_dir, 'inria.h5')

    if not os.path.exists(fullpath):
        dp.make_inria_data()

    fullpath_list = list(np.genfromtxt(fullpath, dtype=None))
    swapped_list = list(np.genfromtxt(swapped, dtype=None))

    len_files = len(fullpath_list)

    fullpath_list = [fullpath_list[item].split(',')[0] for item in range(len_files)]
    swapped_list = [swapped_list[item].split(',')[0] for item in range(len_files)]

    with h5py.File(h5_path, 'w') as myfile:
        for item in range(len_files):
            myfile.create_dataset(name=swapped_list[item], data=ndimage.imread(fullpath_list[item]))


