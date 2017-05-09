import numpy as np
import project_constants as pc
import dynamic_data_loading as ddl
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


# recursively transform list into tuple
def tupconv(lst):
    tuplst = []
    for x in lst:
        if isinstance(x, np.ndarray):
            tuplst.append(tupconv(x))
        elif isinstance(x, list):
            tuplst.append(tupconv(x))
        else:
            tuplst.append(x)
    return tuple(tuplst)


# used to calculate error
def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


# crop images in center
def crop_INRIA_images(folder_path, width, height):
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


def get_wrong_predictions():
    folder = 'wrong_predictions'

    paths = np.genfromtxt('test_images.csv', dtype=None).tolist()
    ans = np.genfromtxt('wrong_predictions.txt', dtype=None).tolist()

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)

    for line in range(0, len(ans)):
        step = ans[line].split(',')[1]
        if step == 'testing':
            target = ans[line].split(',')[3]
            prediction = ans[line].split(',')[5]
            if not target == prediction:
                thing = paths[line].split('/')[-1]
                copyfile(paths[line], os.path.join(folder, thing))


# image has to be 64x128, this adds padding
def fix_viper():
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    cam_a_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_a'
    cam_b_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_b'


    # # assuming they don't exist yet
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


# make matching and non-matching pairs
def make_pairs_viper():
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


def flip_labels(labels):
    labels = map(int, labels)
    new_labels = []

    for item in range(0, len(labels)):
        if labels[item] == 1:
            new_labels.append(0)
        else:
            new_labels.append(1)

    return new_labels

def threshold_predictions(predictions):
    new_predictions = np.zeros((len(predictions), 2))
    for item in range(0, len(predictions)):
        new_predictions[item, np.argmax(predictions[item])] = 1

    return new_predictions


def calculate_accuracy(predictions, labels):
    predictions = threshold_predictions(predictions)
    good = 0.0
    total = len(predictions) * 1.0

    if len(np.shape(labels)) > 1:
        for pred in range(0, len(predictions)):
            if predictions[pred][0] == labels[pred][0]:
                good += 1
    else:
        for pred in range(0, len(predictions)):
            if predictions[pred] == labels[pred]:
                good += 1

    acc = good / total
    return acc


# FIXME make it euclidean distance compatible
def make_confusion_matrix(predictions, labels):
    if len(np.shape(labels)) > 1:
        predictions = threshold_predictions(predictions)
        tp, fp, tn, fn = 0, 0, 0, 0
        for lab in range(0, len(labels)):
            if labels[lab][0] == 0:
                if predictions[lab][0] == 0:
                    tp += 1
                else:
                    fn += 1
            elif labels[lab][0] == 1:
                if predictions[lab][0] == 1:
                    tn += 1
                else:
                    fp += 1
        pass
    else:
        predictions = threshold_predictions(predictions)
        tp, fp, tn, fn = 0, 0, 0, 0
        for lab in range(0, len(labels)):
            if labels[lab] == 1:
                if predictions[lab] == 1:
                    tp += 1  # t=1, p=1
                else:
                    fn += 1  # t=1, p=0
            elif labels[lab] == 0:
                if predictions[lab] == 0:
                    tn += 1
                else:
                    fp += 1

    return [tp, fp, tn, fn]


def print_confusion_matrix(name, confusion_matrix):
    print('%s \n True positive  = %0.2f \n False positive = %0.2f \n True negative  = %0.2f \n False negative = %0.2f \n'
          %(name, confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]))


def enter_in_log(experiment_name, file_name, data_names, matrix_means, matrix_std, ranking_means, ranking_std, total_time):
    decimals = '.2f'
    if not os.path.exists(pc.LOG_FILE_PATH):
        with open(pc.LOG_FILE_PATH, 'w') as my_file:
            print('new log file made')

    with open(pc.LOG_FILE_PATH, 'a') as log_file:
        date = str(time.strftime("%d/%m/%Y")) + "   " + str(time.strftime("%H:%M:%S"))
        log_file.write('\n')
        log_file.write('name_of_experiment:         %s\n' % experiment_name)
        log_file.write('file_name:                  %s\n' % file_name)
        log_file.write('date:                       %s\n' % date)
        log_file.write('duration:                   %f\n' % total_time)
        for dataset in range(len(data_names)):
            name = data_names[dataset]
            log_file.write('%s mean tp,fp,tn,fn:    %s\n' % (name, str(reduce_float_length(matrix_means[dataset].tolist(), decimals))))
            log_file.write('%s std tp,fp,tn,fn:     %s\n' % (name, str(reduce_float_length(matrix_std[dataset].tolist(), decimals))))
            log_file.write('%s mean ranking:        %s\n' % (name, str(reduce_float_length(ranking_means[dataset].tolist(), decimals))))
            log_file.write('%s std ranking:         %s\n' % (name, str(reduce_float_length(ranking_std[dataset].tolist(), decimals))))
        log_file.write('\n')


def fix_cuhk1():
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


def merge_reid_sets(save=False):
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


# FIXME make comptible with euclidean distance
def calculate_CMC(predictions):
    ranking_matrix_abs = np.zeros((pc.RANKING_NUMBER, pc.RANKING_NUMBER))
    predictions = np.reshape(predictions[:, 1], (pc.RANKING_NUMBER, pc.RANKING_NUMBER))
    tallies = np.zeros(pc.RANKING_NUMBER)
    final_ranking = []

    for row in range(len(predictions)):
        ranking_matrix_abs[row] = [i[0] for i in sorted(enumerate(predictions[row]), key=lambda x: x[1],
                                                        reverse=True)]
        list_form = ranking_matrix_abs[row].tolist()
        num = list_form.index(row)
        tallies[num] += 1

    for tally in range(len(tallies)):
        percentage = sum(tallies[0:tally + 1]) * 1.0 / sum(tallies) * 1.0
        final_ranking.append(float('%0.2f' % percentage))
    return final_ranking


def reduce_float_length(a_list, decimals):
    for i in range(len(a_list)):
        a_list[i] = float(format(a_list[i], decimals))
    return a_list


# FIXME make a method that returns true or false if gpu is in use or not respectively
def gpu_in_use(gpu_number):
    return True


def assign_experiments():
    import running_experiments as re
    list_of_experiments = []
    list_of_experiments = ['experishit']
    # for i in range(57, 69 + 1):
    #     list_of_experiments.append('experiment_%d' % i)
    number_of_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])
    for gpu in range(number_of_gpus):
        if gpu_in_use(gpu) == False:
            the_experiment = getattr(re, list_of_experiments.pop(0))
            the_experiment(gpu)

assign_experiments()