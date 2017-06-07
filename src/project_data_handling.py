"""
Handles everything that has to do with turning raw images
into a list of positive and negative pairs
raw_data_handling?
"""
import numpy as np
import project_constants as pc
from PIL import Image
import os
from itertools import combinations
import time
import random as rd
import h5py
from scipy import ndimage
import matplotlib.pyplot as plt

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


def fix_viper():
    """ image has to be 64x128, this adds padding
    """
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    # cam_a_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_a'
    # cam_b_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_b'


    # assuming they don't exist yet
    os.mkdir(padded_folder_path)
    # os.mkdir(cam_a_p)
    # os.mkdir(cam_b_p)

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
    # it throws an error but it does the job


def make_pairs_viper_old():
    """ make matching and non-matching pairs
    """
    padded_folder_path = pc.LOCATION_RAW_VIPER
    pairings_pos_name = '../data/VIPER/positives.txt'
    pairings_neg_name = '../data/VIPER/negatives.txt'
    ranking_pos_name =  '../data/VIPER/ranking_pos.txt'
    ranking_neg_name =  '../data/VIPER/ranking_neg.txt'

    list_ids = os.listdir(os.path.join(padded_folder_path, 'cam_a'))

    ranking_ids = list_ids[0:pc.RANKING_NUMBER]
    list_ids = list_ids[pc.RANKING_NUMBER:]

    ranking_combos = combinations(ranking_ids, 2)

    with open(ranking_pos_name, 'wr') as myFile:
        for id in ranking_ids:
            path_1 = os.path.join(padded_folder_path, 'cam_a', id)
            path_2 = os.path.join(padded_folder_path, 'cam_b', id)
            myFile.write(str(path_1 + ',' + path_2 + ',1\n'))

    with open(ranking_neg_name, 'wr') as myFile:
        for comb in ranking_combos:
            a = comb[0]
            b = comb[1]
            if comb[0] == comb[1]:
                pass
            else:
                path_1 = os.path.join(padded_folder_path, 'cam_a', comb[0])
                path_2 = os.path.join(padded_folder_path, 'cam_b', comb[1])
                myFile.write(str(path_1 + ',' + path_2 + ',0\n'))

    combos = combinations(list_ids, 2)

    with open(pairings_pos_name, 'wr') as myFile:
        for id in list_ids:
            path_1 = os.path.join(padded_folder_path, 'cam_a', id)
            path_2 = os.path.join(padded_folder_path, 'cam_b', id)
            myFile.write(str(path_1 + ',' + path_2 + ',1\n'))

    with open(pairings_neg_name, 'wr') as myFile:
        for comb in combos:
            a = comb[0]
            b = comb[1]
            if comb[0] == comb[1]:
                pass
            else:
                path_1 = os.path.join(padded_folder_path, 'cam_a', comb[0])
                path_2 = os.path.join(padded_folder_path, 'cam_b', comb[1])
                myFile.write(str(path_1 + ',' + path_2 + ',0\n'))


def fix_cuhk1():
    """ crops images to 128x64
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK1'
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
        print('asdf')
        os.makedirs(new_folder_path)

    for image_path in list_images:
        img = Image.open(os.path.join(folder_path, image_path))

        img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
        img.save(os.path.join(new_folder_path, image_path))


def make_pairs_cuhk1_old():
    """ makes positive and negative pairs
    """
    def match(one, two):
        return list(one)[0:4] == list(two)[0:4]

    images_path = pc.LOCATION_RAW_CUHK01
    pairings_pos_name = '../data/CUHK/positives.txt'
    pairings_neg_name = '../data/CUHK/negatives.txt'
    ranking_pos_name =  '../data/CUHK/ranking_pos.txt'
    ranking_neg_name =  '../data/CUHK/ranking_neg.txt'

    list_ids = sorted(os.listdir(images_path))

    ranking_ids = list_ids[0:pc.RANKING_NUMBER*4]
    list_ids = list_ids[pc.RANKING_NUMBER*4:]

    ranking_combos = combinations(ranking_ids, 2)

    with open(ranking_pos_name, 'wr') as rankFilePos:
        with open(ranking_neg_name, 'wr') as rankFileNeg:
            for comb in ranking_combos:
                pic_1 = os.path.join(images_path, comb[0])
                pic_2 = os.path.join(images_path, comb[1])
                if match(comb[0], comb[1]):
                    if comb[0] == comb[1]:
                        pass
                    else:
                        rankFilePos.write(str(pic_1 + ',' + pic_2 + ',1\n'))
                else:
                    rankFileNeg.write(str(pic_1 + ',' + pic_2 + ',0\n'))

    combos = combinations(list_ids, 2)

    with open(pairings_pos_name, 'wr') as posFile:
        with open(pairings_neg_name, 'wr') as negFile:
            for comb in combos:
                pic_1 = os.path.join(images_path, comb[0])
                pic_2 = os.path.join(images_path, comb[1])
                if match(comb[0], comb[1]):
                    if comb[0] == comb[1]:
                        pass
                    else:
                        posFile.write(str(pic_1 + ',' + pic_2 + ',1\n'))
                else:
                    negFile.write(str(pic_1 + ',' + pic_2 + ',0\n'))


def fix_cuhk2():
    # note: in the later pipeline, treat CUHK02 as 5 different datasets
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK2'
    cropped_folder_path = os.path.join(os.path.dirname(folder_path), 'cropped_CUHK2')
    if not os.path.exists(cropped_folder_path): os.mkdir(cropped_folder_path)

    subdirs = os.listdir(folder_path)

    for dir in subdirs:
        if not os.path.exists(os.path.join(cropped_folder_path, dir)):
            os.mkdir(os.path.join(cropped_folder_path, dir))
            os.mkdir(os.path.join(cropped_folder_path, dir, 'all'))

    cameras = ['cam1', 'cam2']

    for dir in subdirs:
        for cam in cameras:
            original_images_path = os.path.join(folder_path, dir, cam)
            cropped_images_path = os.path.join(cropped_folder_path, dir, 'all')
            images = [file for file in os.listdir(original_images_path) if file.endswith('.png')]
            for ind in range(len(images)):
                image = os.path.join(original_images_path, images[ind])
                cropped_image = os.path.join(cropped_images_path, images[ind])
                img = Image.open(image)
                img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
                img.save(cropped_image)



def standardize(all_images, folder_path, fixed_folder_path, the_mod=None):
    def modify(name, the_mod):
        return name.split('.')[0].split('_')[-1] + the_mod + name.split('.')[-1]

    for image in all_images:
        original_image_path = os.path.join(folder_path, image)

        if not the_mod == None:
            modified_image_path = os.path.join(fixed_folder_path, modify(image, the_mod))
        else:
            modified_image_path = os.path.join(fixed_folder_path, image)
        the_image = Image.open(original_image_path)
        image_width, image_height = the_image.size

        case = None
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
    folder_path = '/home/gabi/Documents/datasets/CAVIAR4REID/original'
    fixed_folder_path = os.path.join(os.path.dirname(folder_path), 'fixed_caviar')
    if not os.path.exists(fixed_folder_path): os.mkdir(fixed_folder_path)

    all_images = os.listdir(folder_path)
    standardize(all_images, folder_path, fixed_folder_path)


def fix_grid():
    folder_path = '/home/gabi/Documents/datasets/GRID'
    probe = os.path.join(folder_path, 'probe')
    gallery = os.path.join(folder_path, 'gallery')

    probe_list = os.listdir(probe)
    gallery_list = os.listdir(gallery)

    proper_gallery_list = [item for item in gallery_list if not item[0:4] == '0000' ]

    fixed_folder_path = os.path.join(os.path.dirname(probe), 'fixed_grid')
    if not os.path.exists(fixed_folder_path): os.mkdir(fixed_folder_path)

    standardize(probe_list, probe, fixed_folder_path)
    standardize(proper_gallery_list, gallery, fixed_folder_path)


def fix_prid450():
    folder_path = '/home/gabi/Documents/datasets/PRID450'
    cam_a = os.path.join(folder_path, 'cam_a')
    cam_b = os.path.join(folder_path, 'cam_b')

    cam_a_list = os.listdir(cam_a)
    cam_b_list = os.listdir(cam_b)

    proper_cam_a_list = [item for item in cam_a_list if item.split('_')[0] == 'img']
    proper_cam_b_list = [item for item in cam_b_list if item.split('_')[0] == 'img']

    fixed_folder_path = os.path.join(os.path.dirname(cam_a), 'fixed_prid')
    if not os.path.exists(fixed_folder_path): os.mkdir(fixed_folder_path)

    standardize(proper_cam_a_list, cam_a, fixed_folder_path, '_a.')
    standardize(proper_cam_b_list, cam_b, fixed_folder_path, '_b.')


def make_pairs_cuhk2_old():
    # note:treat CUHK02 as 5 different datasets since it's partitioned into 5 datasets and the imagenames are not unique
    # This shoulnd't affect training because the total number of positive pairs will still be the same
    folder_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK2'
    sub_dirs = os.listdir(folder_path)

    def match(one, two):
        one = one.split('/')[-1]
        two = two.split('/')[-1]
        return list(one)[0:4] == list(two)[0:4]

    def write_combo_to_file(ids, pos, neg):
        combos = combinations(ids, 2)
        with open(pos, 'a') as pos_file:
            with open(neg, 'a') as neg_file:
                for comb in combos:
                    pic_1 = comb[0]
                    pic_2 = comb[1]

                    if match(pic_1, pic_2):
                        if not pic_1 == pic_2:
                            pos_file.write(str(pic_1 + ',' + pic_2 + ',1\n'))
                    else:
                        neg_file.write(str(pic_1 + ',' + pic_2 + ',0\n'))

    project_data_storage = '../data/CUHK02'
    
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    pairing_pos_name = os.path.join(project_data_storage, 'positives.txt')
    pairing_neg_name = os.path.join(project_data_storage, 'negatives.txt')
    ranking_pos_name =  os.path.join(project_data_storage, 'ranking_pos.txt')
    ranking_neg_name =  os.path.join(project_data_storage, 'ranking_neg.txt')

    for dir in sub_dirs:
        identity_list = sorted(os.listdir(os.path.join(folder_path, dir, 'all')))
        fullpath_identity_list = [os.path.join(folder_path, dir, 'all', item) for item in identity_list]

        adapted_ranking_number = pc.RANKING_NUMBER / len(sub_dirs)
        ranking_ids = fullpath_identity_list[0:adapted_ranking_number * 4]
        pairing_ids = fullpath_identity_list[adapted_ranking_number * 4:]
        
        write_combo_to_file(ranking_ids, ranking_pos_name, ranking_neg_name)
        write_combo_to_file(pairing_ids, pairing_pos_name, pairing_neg_name)


def write_to_file(filepath, data):
    with open(filepath, 'w') as myfile:
        for i in range(len(data)):
            myfile.write(str(data[i]) + '\n')


def unique_id_and_all_images_viper():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])
    project_data_storage = '../data/VIPER'

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_cuhk1():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])
    project_data_storage = '../data/CUHK'

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_cuhk2():
    """ This only needs to be done once ever.
    """
    top_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK2'

    subdirs = os.listdir(top_path)

    for dir in subdirs:
        folder_path = os.path.join(top_path, dir, 'all')

        id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
        unique_id = sorted(set(id_all))
        short_image_names = sorted(os.listdir(folder_path))
        fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])

        project_data_storage = os.path.join('../data/CUHK02', dir)
        if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

        id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
        unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
        short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
        fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

        write_to_file(id_all_file, id_all)
        write_to_file(unique_id_file, unique_id)
        write_to_file(short_image_names_file, short_image_names)
        write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_market():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/market-1501/identities'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])
    project_data_storage = '../data/market'

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_caviar():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/CAVIAR4REID/fixed_caviar'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])

    project_data_storage = '../data/caviar'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_grid():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/GRID/fixed_grid'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])
    project_data_storage = '../data/GRID'

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def unique_id_and_all_images_prid450():
    """ This only needs to be done once ever.
    """
    folder_path = '/home/gabi/Documents/datasets/PRID450/fixed_prid'
    id_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_id = sorted(set(id_all))
    short_image_names = sorted(os.listdir(folder_path))
    fullpath_image_names = sorted([os.path.join(folder_path, item) for item in short_image_names])
    project_data_storage = '../data/prid450'

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    write_to_file(id_all_file, id_all)
    write_to_file(unique_id_file, unique_id)
    write_to_file(short_image_names_file, short_image_names)
    write_to_file(fullpath_image_names_file, fullpath_image_names)


def make_combos_1(ids):
    """ Given a list of strings, create combinations.
        A combination is valid if the IDs match and if the image is not identical
    """
    def match(one, two):
        one = one.split('/')[-1]
        two = two.split('/')[-1]
        return list(one)[0:4] == list(two)[0:4]

    combos = list(combinations(ids, 2))

    combo_list = [str(comb[0] + ',' + comb[1] + ',1\n') for comb in combos if
                  (match(comb[0], comb[1]) and not comb[0] == comb[1])]

    return combo_list


def make_combos_2(fullpath, unique_ids, num, smallest_id_group, the_type):
    if num > smallest_id_group:
        num = smallest_id_group
    combo_list = []
    num_ids = len(unique_ids)
    # if the_type == 'training': num_ids = len(fullpath) / 3
    # elif the_type == 'ranking': num_ids = len(fullpath) / 2

    if num == 3:
        for item in range(num_ids):
            thing = str(fullpath[num*item] + ',' + fullpath[num*item+1] + ',1\n')
            combo_list.append(thing)
            thing = str(fullpath[num * item + 1] + ',' + fullpath[num * item + 2] + ',1\n')
            combo_list.append(thing)
    elif num == 2:
        for item in range(num_ids):
            a = fullpath[num*item]
            b = fullpath[num*item+1]
            thing = str(fullpath[num*item] + ',' + fullpath[num*item+1] + ',1\n')
            combo_list.append(thing)
    return combo_list


# FIXME market has min id group size of 2: this is an issue. add option to pop the groups of size 2 if dataset == market

def pre_selection(the_list, unique_ids, all_ids, num, dataset_name):
    """ Prevents there from being a HUGE number of combinations of pairs by setting an upper bound on allowed images per
        unique ID
    :param the_list:        list containing full path to images of the set of IDs. an ID can have multiple images
    :param unique_ids:      list of unique IDs. every ID appears only once
    :param all_ids:         the_list, but then only the IDs
    :param num:             the number of allowed image per ID
    :return:                truncated selection of the_list
    """
    selection = []
    min_id_group_size = 100000
    # keep track of which ids we ignore.
    ignore_id = []

    for id in unique_ids:
        # print('id: %s' % str(id))
        # get the indices for the matching IDs
        id_group = [i for i, x in enumerate(all_ids) if x == id]
        # get the fullpaths for each matching ID at the indices
        full_path_group = [the_list[i] for i in id_group]
        # update min_id_group_size
        # if the dataset is large and has a lot of large id groups(>num) then ignore the id groups that are smaller than num
        if dataset_name == 'market' or dataset_name == 'cuhk02' or dataset_name == 'caviar':
            if len(id_group) >= num:
                # print('length id group: %d, num: %d' % (len(id_group), num))
                if min_id_group_size > len(id_group):
                    min_id_group_size = len(id_group)
                # if there are more matching ID images than allowed images,
                # only add the number of allowed matching ID images
                # use a random number to decide which images get popped to make sure you don't choose the same images always
                for ble in range(num):
                    selection.append(full_path_group.pop(rd.randrange(0, len(full_path_group))))
            else:
                # to_pop_index = unique_ids.index(id)
                #
                # print('index: %s, to pop value: %d' % (str(id), to_pop_index))
                ignore_id.append(id)
        else:
            if min_id_group_size > len(id_group): min_id_group_size = len(id_group)
            if num > len(id_group):
                # if the number of allowed images is greater than the number of matching ID images,
                # add all the images of that ID to the selection
                sub_selection = [thing for thing in full_path_group]
                selection += sub_selection
            else:
                # if there are more matching ID images than allowed images,
                # only add the number of allowed matching ID images
                # use a random number to decide which images get popped to make sure you don't choose the same images always
                for ble in range(num):
                    selection.append(full_path_group.pop(rd.randrange(0, len(full_path_group))))

    # print('length unique ids before: %d' % (len(unique_ids)))

    # print('amount to pop: %d' % (len(ignore_id)))

    for value in ignore_id:
        index = unique_ids.index(value)
        # print('index: %d, value: %s' % (index, str(value)))
        popped = unique_ids.pop(index)
        # print('popped: %d' % popped)

    # print('length unique ids after: %d' % (len(unique_ids)))

    return selection, min_id_group_size, unique_ids


#note:swapped
def make_all_positives(id_all_file, unique_id_file, short_image_names_file, fullpath_image_names_file, dataset_name,
               ranking_number=pc.RANKING_NUMBER):
    """ This needs to be done once at the beginning of the iteration.
    """
    def match(one, two):
        return list(one)[0:4] == list(two)[0:4]
    
    def make_unique(the_list):
        unique = []
        seen = []
        for item in range(len(the_list)):
            i1 = the_list[item].split(',')[0].split('+')[-1][0:4]
            i2 = the_list[item].split(',')[1].split('+')[-1][0:4]
            if i1 == i2:
                if i1 not in seen:
                    seen.append(i1)
                    unique.append(the_list[item])
        return unique

    # create a list with unique identities
    unique_id = np.genfromtxt(unique_id_file, dtype=None).tolist()
    # select at random a subset for ranking by drawing indices from a uniform distribution
    start = rd.randrange(0, len(unique_id)-ranking_number)
    stop = start + ranking_number
    # we will need a list of the training id for later
    train_ids = unique_id[0:start] + unique_id[stop:]
    # we will need a list of the ranking id for later
    ranking_ids = unique_id[start:stop]
    # load the list with all the identities
    id_all = np.genfromtxt(id_all_file, dtype=None).tolist()
    # determine where to slice the list depending on the start and stop indices
    index_start = id_all.index(unique_id[start])
    index_stop = id_all.index(unique_id[stop])
    # we will need a list of the training id for later
    all_train_ids = id_all[0:index_start] + id_all[index_stop:]
    # we will need a list of the ranking id for later
    all_ranking_ids = id_all[index_start:index_stop]
    # load the list with all the imagepaths
    fullpath_image_names = np.genfromtxt(fullpath_image_names_file, dtype=None).tolist()
    # slice the lists to create a set for ranking and a set for training
    ranking_ids_pos = fullpath_image_names[index_start:index_stop]
    training_ids_pos = fullpath_image_names[0:index_start] + fullpath_image_names[index_start:]
    # if we use all the data we have more than 300 million combinations. going through all of these will take about 80 mins
    # instead for each unique id we select 2 images matching this id and discard the rest of the images for that id.
    # then we have 4.4 million combos instead of 300 million which will take no more than 2 minutes. this is acceptable
    upper_bound = 3
    # create combinations and store the positive matches
    ranking_ids_pos, min_group_size_rank, ranking_ids = pre_selection(ranking_ids_pos, ranking_ids, all_ranking_ids, num=2, dataset_name=dataset_name)
    ranking_ids_pos = make_combos_2(ranking_ids_pos, ranking_ids, 2, min_group_size_rank, 'ranking')
    training_ids_pos, min_group_size_train, train_ids = pre_selection(training_ids_pos, train_ids, all_train_ids, upper_bound, dataset_name)
    training_ids_pos = make_combos_2(training_ids_pos, train_ids, upper_bound, min_group_size_train, 'training')
    # shuffle so that each time we get different first occurences
    rd.shuffle(ranking_ids_pos)
    rd.shuffle(training_ids_pos)
    # from the shuffled original list, create a unique list by only selecting first id occurences
    # ranking_ids_pos = make_unique(ranking_ids_pos)
    # training_ids_pos = make_unique(training_ids_pos)

    return ranking_ids_pos, training_ids_pos


def make_all_negatives(pos_list, the_type):
    list_0 = [pos_list[index].split(',')[0] for index in range(len(pos_list))]
    list_1 = [pos_list[index].split(',')[1] for index in range(len(pos_list))]

    if the_type == 'ranking':
        ranking_list = []
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                num = 1 if img0 == img1 else 0
                line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
                ranking_list.append(line)
        return ranking_list

    elif the_type == 'training':
        training_pos = []
        training_neg = []
        for img0 in range(len(list_0)):
            for img1 in range(len(list_1)):
                if img0 == img1:
                    line = list_0[img0] + ',' + list_1[img1] + ',1\n'
                    training_pos.append(line)
                else:
                    line = list_0[img0] + ',' + list_1[img1] + ',0\n'
                    training_neg.append(line)
        return training_pos, training_neg

# note: swapped
def make_pairs_viper():
    start = time.time()
    project_data_storage = '../data/VIPER'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_viper()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'viper')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg


# note: swapped
def make_pairs_cuhk1():
    start = time.time()
    project_data_storage = '../data/CUHK'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_cuhk1()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'cuhk01')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg


def merge_ranking_files(rank_list):
    # for cuhk2
    rank_list_pos = []
    for item in rank_list:
        the_label = int(item.strip().split(',')[-1])
        if the_label == 1:
            rank_list_pos.append(item)

    list_0 = [rank_list_pos[index].split(',')[0] for index in range(len(rank_list_pos))]
    list_1 = [rank_list_pos[index].split(',')[1] for index in range(len(rank_list_pos))]

    rank_list_pos = []
    for img0 in range(len(list_0)):
        for img1 in range(len(list_1)):
            num = 1 if img0 == img1 else 0
            line = list_0[img0] + ',' + list_1[img1] + ',%d\n' % num
            rank_list_pos.append(line)

    return rank_list_pos


# note: swapped
def make_pairs_cuhk2():
    start = time.time()
    top_project_data_storage = '../data/CUHK02'
    original_data_location = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK2'

    subdirs = os.listdir(original_data_location)
    adapted_ranking_number = pc.RANKING_NUMBER / len(subdirs)

    ranking_all = []
    training_pos_all = []
    training_neg_all = []

    for dir in subdirs:
        project_data_storage = os.path.join(top_project_data_storage, dir)

        if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

        id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
        unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
        short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
        fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
        # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

        if not os.path.exists(id_all_file):
            unique_id_and_all_images_cuhk2()

        ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                       fullpath_image_names_file, ranking_number=adapted_ranking_number, dataset_name='cuhk02')

        ranking = make_all_negatives(ranking_pos, 'ranking')
        training_pos, training_neg = make_all_negatives(training_pos, 'training')

        ranking_all += ranking
        training_pos_all += training_pos
        training_neg_all += training_neg

    ranking_all = merge_ranking_files(ranking_all)

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)
    return ranking_all, training_pos_all, training_neg_all


#note:swapped
def make_pairs_market():
    start = time.time()
    project_data_storage = '../data/market'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_market()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'market')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg

#note:swapped
def make_pairs_caviar():
    start = time.time()
    project_data_storage = '../data/caviar'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_caviar()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'caviar')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg


#note:swapped
def make_pairs_grid():
    start = time.time()
    project_data_storage = '../data/GRID'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_grid()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'grid')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg


#note:swapped
def make_pairs_prid450():
    start = time.time()
    project_data_storage = '../data/prid450'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)

    id_all_file = os.path.join(project_data_storage, 'id_all_file.txt')
    unique_id_file = os.path.join(project_data_storage, 'unique_id_file.txt')
    short_image_names_file = os.path.join(project_data_storage, 'short_image_names_file.txt')
    fullpath_image_names_file = os.path.join(project_data_storage, 'swapped_list_of_paths.txt')
    # fullpath_image_names_file = os.path.join(project_data_storage, 'fullpath_image_names_file.txt')

    if not os.path.exists(id_all_file):
        unique_id_and_all_images_prid450()

    ranking_pos, training_pos = make_all_positives(id_all_file, unique_id_file, short_image_names_file,
                                                   fullpath_image_names_file, 'prid450')

    ranking = make_all_negatives(ranking_pos, 'ranking')
    training_pos, training_neg = make_all_negatives(training_pos, 'training')

    total_time = time.time() - start
    print('total_time   %0.2f seconds' % total_time)

    return ranking, training_pos, training_neg


# unused
def merge_reid_sets_old(save=False):
    """ merges the mentioned datasets
    """
    data_location = '../data'
    pos = 'positives.txt'
    neg = 'negatives.txt'
    viper_pos = np.genfromtxt(os.path.join(data_location, 'VIPER', pos), dtype=None).tolist()
    viper_neg = np.genfromtxt(os.path.join(data_location, 'VIPER', neg), dtype=None).tolist()
    cuhk_pos = np.genfromtxt(os.path.join(data_location, 'CUHK', pos), dtype=None).tolist()
    cuhk_neg = np.genfromtxt(os.path.join(data_location, 'CUHK', neg), dtype=None).tolist()

    pos_list = viper_pos + cuhk_pos
    neg_list = viper_neg + cuhk_neg

    if save:
        all_pos_list = os.path.join(data_location, 'reid_all_positives.txt')
        all_neg_list = os.path.join(data_location, 'reid_all_negatives.txt')

        with open(all_pos_list, 'wr') as myFile:
            for line in pos_list:
                myFile.write(str(line) + '\n')

        with open(all_neg_list, 'wr') as myFile:
            for line in neg_list:
                myFile.write(str(line) + '\n')

    return pos_list, neg_list


def my_join(list_strings):
    awesome_string = ''
    for item in list_strings:
        awesome_string += item
    return awesome_string


def swap_for(the_thing, a, b):
    the_thing = list(the_thing)
    for item in range(len(the_thing)):
        if the_thing[item] == a:
            the_thing[item] = b

    the_thing = my_join(the_thing)

    return str(the_thing)


def save_as_hdf5(file_list_of_paths, h5_path):
    list_of_paths = np.genfromtxt(file_list_of_paths, dtype=None).tolist()
    swapped_file_list_of_paths = os.path.join(os.path.dirname(file_list_of_paths), 'swapped_list_of_paths.txt')

    # with h5py.File(h5_path, 'a') as myfile:

    action = 'a' if os.path.exists(h5_path) else 'w'

    with h5py.File(h5_path, action) as myfile:
        with open(swapped_file_list_of_paths, 'w') as my_other_file:
            for item in list_of_paths:
                # swap the '/' for '+' or else listing the hdf5 keys will be a problem later
                item_name = swap_for(item, '/', '+')
                my_other_file.write(item_name + '\n')
                data = myfile.create_dataset(name=item_name, data=ndimage.imread(item))


def save_all_datasets_as_hdf5():
    save_as_hdf5('../data/GRID/fullpath_image_names_file.txt', '../data/GRID/grid.h5')
    print('saved grid')

    save_as_hdf5('../data/prid450/fullpath_image_names_file.txt', '../data/prid450/prid450.h5')
    print('saved prid450')

    save_as_hdf5('../data/caviar/fullpath_image_names_file.txt', '../data/caviar/caviar.h5')
    print('saved caviar')

    save_as_hdf5('../data/VIPER/fullpath_image_names_file.txt', '../data/VIPER/viper.h5')
    print('saved viper')

    subdirs = [item for item in os.listdir('../data/CUHK02') if not item.split('.')[-1] == 'txt']
    cuhk2_path = '../data/CUHK02'
    for dir in subdirs:
        the_full = os.path.join(cuhk2_path, dir, 'fullpath_image_names_file.txt')
        the_h5 = os.path.join(cuhk2_path, 'cuhk02.h5')
        save_as_hdf5(the_full, the_h5)
    print('saved cuhk02')

    save_as_hdf5('../data/CUHK/fullpath_image_names_file.txt', '../data/CUHK/cuhk01.h5')
    print('saved cuhk01')

    save_as_hdf5('../data/market/fullpath_image_names_file.txt', '../data/market/market.h5')
    print('saved market')



def read_plot_from_hdf5(file_list_of_paths, h5_path):
    hdf5_file = h5py.File(h5_path, 'r')

    a = hdf5_file.keys()

    list_of_paths = np.genfromtxt(file_list_of_paths, dtype=None).tolist()
    for i in range(10):
        thing = hdf5_file[list_of_paths[i]][:]
        plt.imshow(thing)

# save_all_datasets_as_hdf5()

# read_plot_from_hdf5('/home/gabi/PycharmProjects/uatu/data/VIPER/fullpath_image_names_file.txt', '/home/gabi/PycharmProjects/uatu/data/VIPER/viper.h5')