"""
Contains methods for making INRIA and NICTA datasets usable.
This was used when we first trained a CNN to recognize humans and then we transfer the learned weights to the SCNN
"""

import os
from PIL import Image
import time
import numpy as np

import project_constants as pc


# unused
def crop_INRIA_images(folder_path, width, height):
    """ crop images in center
    """
    num = 1
    if folder_path.endswith('/'):
        num = 2

    parts = folder_path.split('/')
    new_path = ''
    for i in range(0, len(parts)-num):
        new_path = os.path.join(new_path, parts[i])

    list_images = os.listdir(folder_path)
    name_folder = folder_path.split('/')[-num]
    new_folder_path = os.path.join(new_path, 'cropped_' + str(name_folder))
    new_folder_path = '/' + new_folder_path
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for image_path in list_images:
        img = Image.open(os.path.join(folder_path, image_path))
        img_width, img_height = img.size

        center_x = img_width / 2
        center_y = img_height / 2
        start_x = center_x - width / 2
        start_y = center_y - height / 2
        img2 = img.crop((start_x, start_y, start_x + width, start_y + height))
        img2.save(os.path.join(new_folder_path, image_path))


# unused
def fix_NICTA(name):
    original_folder_path = '/home/gabi/Documents/datasets/NICTAPedestrians/' + str(name) + '/64x80'
    padded_path = '/home/gabi/Documents/datasets/NICTAPedestrians/padded_' + str(name)

    # assuming they don't exist yet
    if not os.path.exists(padded_path):
        os.mkdir(padded_path)

    folder_list_level_1 = os.listdir(original_folder_path)
    for item_level_1 in folder_list_level_1:
        path_1 = os.path.join(original_folder_path, item_level_1)
        folder_list_level_2 = os.listdir(path_1)
        for item_level_2 in folder_list_level_2:
            path_2 = os.path.join(path_1, item_level_2)
            image_list = os.listdir(path_2)

            pad_path_1 = os.path.join(padded_path, item_level_1, item_level_2)

            if not os.path.exists(pad_path_1):
                os.makedirs(pad_path_1)

            for image in image_list:
                image_path = os.path.join(path_2, image)

                img = Image.open(image_path)
                new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))

                img_width, img_height = img.size
                new_img_width, new_img_height = new_img.size
                padding_width = (new_img_width - img_width) / 2
                padding_height = (new_img_height - img_height) / 2

                new_img.paste(img, box=(padding_width, padding_height))

                name = image.split('.')[0]
                filename = os.path.join(pad_path_1, str(name)+'.jpg')
                new_img.save(filename)


# unused
def create_pos_neg_nicta():
    start = time.time()
    print('Create pos neg list from NICTA pedestrian')
    base_path_pos = '/home/gabi/Documents/datasets/NICTAPedestrians/padded_positives'
    base_path_neg = '/home/gabi/Documents/datasets/NICTAPedestrians/padded_negatives'
    data_list_path = '/home/gabi/PycharmProjects/uatu/data/NICTA'

    positives_list = os.path.join(data_list_path, 'positives.txt')
    negatives_list = os.path.join(data_list_path, 'negatives.txt')

    def make_list(data_list, base_path, lab):
        count = 0
        if os.path.exists(data_list):
            print('%s exists already. aborting.' % data_list)
        else:
            with open(data_list, 'wr') as myFile:
                for dir_1 in os.listdir(base_path):
                    dir_1_path = os.path.join(base_path, dir_1)
                    for dir_2 in os.listdir(dir_1_path):
                        dir_2_path = os.path.join(dir_1_path, dir_2)
                        for image in os.listdir(dir_2_path):
                            image_path = os.path.join(dir_2_path, image)
                            myFile.write(str(image_path) + ',%d\n' % lab)
                            count += 1
            print('Created list. Size: %d' % count)

    make_list(positives_list, base_path_pos, 1)
    make_list(negatives_list, base_path_neg, 0)

    total_time = time.time() - start
    print('total time: %0.2f' % (total_time))


# unused
def create_pos_neg_inria():
    start = time.time()
    print('Create pos neg list from INRIA humans')
    original_data_path = '/home/gabi/Documents/datasets/INRIAPerson'
    data_list_path = '/home/gabi/PycharmProjects/uatu/data/INRIA'
    sub_dirs_pos = ['train_64x128_H96/real_cropped_images_pos', 'test_64x128_H96/real_cropped_images_pos']
    sub_dirs_neg = ['train_64x128_H96/real_cropped_images_neg', 'test_64x128_H96/real_cropped_images_neg']

    positives_list = os.path.join(data_list_path, 'positives.txt')
    negatives_list = os.path.join(data_list_path, 'negatives.txt')

    def make_list(data_list, sub_dirs, lab):
        count = 0
        if os.path.exists(data_list):
            print('%s exists already. aborting.' % data_list)
        else:
            with open(data_list, 'wr') as myFile:
                for dir_1 in sub_dirs:
                    dir_1_path = os.path.join(original_data_path, dir_1)
                    for image in os.listdir(dir_1_path):
                        image_path = os.path.join(dir_1_path, image)
                        myFile.write(str(image_path) + ',%d\n' % lab)
                        count += 1
            print('Created list. Size: %d' % count)

    make_list(positives_list, sub_dirs_pos, 1)
    make_list(negatives_list, sub_dirs_neg, 0)

    total_time = time.time() - start
    print('total time: %0.2f' % (total_time))


# unused
def create_pos_cbcl():
    start = time.time()
    print('Create pos list from CBCL pedestrian')
    original_data_path = '/home/gabi/Documents/datasets/CBCL_PEDESTRIAN_DATABASE/images'
    data_list_path = '/home/gabi/PycharmProjects/uatu/data/CBCL'

    if not os.path.exists(data_list_path):
        os.mkdir(data_list_path)

    positives_list = positives_list = os.path.join(data_list_path, 'positives.txt')

    if os.path.exists(positives_list):
        print('%s exists already. aborting.' % positives_list)
    else:
        count = 0
        with open(positives_list, 'wr') as myFile:
            for image in os.listdir(original_data_path):
                image_path = os.path.join(original_data_path, image)
                myFile.write(str(image_path) + ',1\n')
                count += 1
        print('Created list. Size: %d' % count)

    total_time = time.time() - start
    print('total time: %0.2f' % (total_time))


# unused
def merge_pedestrian_sets(save=False):
    data_location = '/home/gabi/PycharmProjects/uatu/data'
    pos = 'positives.txt'
    neg = 'negatives.txt'
    cbcl_pos = np.genfromtxt(os.path.join(data_location, 'CBCL', pos), dtype=None).tolist()
    inria_pos = np.genfromtxt(os.path.join(data_location, 'INRIA', pos), dtype=None).tolist()
    inria_neg = np.genfromtxt(os.path.join(data_location, 'INRIA', neg), dtype=None).tolist()
    nicta_pos = np.genfromtxt(os.path.join(data_location, 'NICTA', pos), dtype=None).tolist()
    nicta_neg = np.genfromtxt(os.path.join(data_location, 'NICTA', neg), dtype=None).tolist()

    pos_list = cbcl_pos + inria_pos + nicta_pos
    neg_list = inria_neg + nicta_neg

    if save:
        all_pos_list = os.path.join(data_location, 'all_positives.txt')
        all_neg_list = os.path.join(data_location, 'all_negatives.txt')

        with open(all_pos_list, 'wr') as myFile:
            for line in pos_list:
                myFile.write(str(line) + '\n')

        with open(all_neg_list, 'wr') as myFile:
            for line in neg_list:
                myFile.write(str(line) + '\n')

    return pos_list, neg_list

