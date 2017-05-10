"""
Handles everything that has to do with turning raw images into a list of positive and negative pairs
"""
import numpy as np
import project_constants as pc
from PIL import Image
import os
from itertools import combinations
import time


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
    cam_a_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_a'
    cam_b_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_b'


    # assuming they don't exist yet
    os.mkdir(padded_folder_path)
    os.mkdir(cam_a_p)
    os.mkdir(cam_b_p)

    cams = ['cam_a', 'cam_b']
    for folder in cams:
        cam_path = os.path.join(original_folder_path, str(folder))
        padded_cam_path = os.path.join(padded_folder_path, str(folder))
        for the_file in os.listdir(cam_path):
            img = Image.open(os.path.join(cam_path, the_file))
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))

            img_width, img_height = img.size
            new_img_width, new_img_height = new_img.size
            padding_width = (new_img_width-img_width)/2
            padding_height = (new_img_height-img_height)/2

            new_img.paste(img, box=(padding_width, padding_height))

            filename = the_file.split('_')[0] + '.bmp'
            filename = os.path.join(padded_cam_path, filename)
            new_img.save(filename)
    # it throws an error but it does the job


def make_pairs_viper():
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


def make_pairs_cuhk1():
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


def make_pairs_cuhk2():
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


def make_pairs_market():
    # don't need to fix market
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

    folder_path = '/home/gabi/Documents/datasets/market-1501/identities'
    identities_all = sorted([item.split('/')[-1][0:4] for item in os.listdir(folder_path)])
    unique_identities = sorted(set(identities_all))
    identities_image_name = sorted(os.listdir(folder_path))
    fullpath_identities = sorted([os.path.join(folder_path, item) for item in identities_image_name])

    project_data_storage = '../data/market'
    if not os.path.exists(project_data_storage): os.mkdir(project_data_storage)
    pairing_pos_name = os.path.join(project_data_storage, 'positives.txt')
    pairing_neg_name = os.path.join(project_data_storage, 'negatives.txt')
    ranking_pos_name = os.path.join(project_data_storage, 'ranking_pos.txt')
    ranking_neg_name = os.path.join(project_data_storage, 'ranking_neg.txt')

    index = identities_all.index(unique_identities[pc.RANKING_NUMBER])
    ranking_ids = fullpath_identities[0:index]
    pairing_ids = fullpath_identities[index:]

    write_combo_to_file(ranking_ids, ranking_pos_name, ranking_neg_name)
    write_combo_to_file(pairing_ids, pairing_pos_name, pairing_neg_name)

# FIXME do this
make_pairs_market()


def merge_reid_sets(save=False):
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

# merge_reid_sets(save=True)