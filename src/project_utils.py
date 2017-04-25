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


def load_INRIA():
    print('Loading INRIA person dataset')
    original_data_path = '/home/gabi/Documents/datasets/INRIAPerson'
    data_list_path = '/home/gabi/PycharmProjects/uatu/data/INRIA'

    if not os.path.exists(data_list_path):
        os.mkdir(data_list_path)

        def load_data_into_list(train_or_validate):
            labels = ['pos', 'neg']
            procedure = 'train_64x128_H96' if train_or_validate == 'train' else 'test_64x128_H96'
            data_list_file = os.path.join(data_list_path, '%s.txt' % train_or_validate)
            with open(data_list_file, 'wr') as the_myFile:
                for label in labels:
                    storage_path = os.path.join(original_data_path, procedure, 'real_cropped_images_' + label)
                    all_items = os.listdir(storage_path)
                    for the_item in all_items:
                        lab = 1 if label == 'pos' else 0
                        file_name = os.path.join(storage_path, the_item)
                        the_myFile.write(file_name + ',%d\n' % lab)

        load_data_into_list('train')
        load_data_into_list('validate')

        # create test dataset
        test_list_path = os.path.join(data_list_path, 'test.txt')
        validation_list_path = os.path.join(data_list_path, 'validate.txt')
        validation_list = np.genfromtxt(validation_list_path, dtype=None).tolist()

        rd.shuffle(validation_list)
        validation_data = validation_list[0:np.shape(validation_list)[0] / 2]
        test_data = validation_list[np.shape(validation_list)[0] / 2:np.shape(validation_list)[0]]

        with open(test_list_path, 'wr') as myFile:
            for item in test_data:
                myFile.write(item + '\n')

        with open(validation_list_path, 'wr') as myFile:
            for item in validation_data:
                myFile.write(item + '\n')

    def load_INRIA_data_in_array(data):
        data_array = np.zeros(shape=(len(data), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
        for image in range(0, len(data)):
            name = data[image].split(',')[0]
            data_array[image] = ndimage.imread(name)[:, :, 0:3]
        return data_array

    train_data = np.genfromtxt(os.path.join(data_list_path, 'train.txt'), dtype=None).tolist()
    rd.shuffle(train_data)
    train_data_array = load_INRIA_data_in_array(train_data)
    train_labels = np.asarray([train_data[row].split(',')[1] for row in range(0, len(train_data))])

    test_data = np.genfromtxt(os.path.join(data_list_path, 'test.txt'), dtype=None).tolist()
    rd.shuffle(test_data)
    test_data_array = load_INRIA_data_in_array(test_data)
    test_labels = np.asarray([test_data[row].split(',')[1] for row in range(0, len(test_data))])

    validation_data = np.genfromtxt(os.path.join(data_list_path, 'validate.txt'), dtype=None).tolist()
    rd.shuffle(validation_data)
    validation_data_array = load_INRIA_data_in_array(validation_data)
    validation_labels = np.asarray([validation_data[row].split(',')[1] for row in range(0, len(validation_data))])

    return train_data_array, train_labels, validation_data_array, validation_labels, test_data_array, test_labels


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
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    # store_folder_path = '/home/gabi/PycharmProjects/uatu/data/VIPeR_CUHK'
    pairings_pos_name = '/home/gabi/PycharmProjects/uatu/data/VIPER/positives.txt'
    pairings_neg_name = '/home/gabi/PycharmProjects/uatu/data/VIPER/negatives.txt'

    list_ids = os.listdir(os.path.join(padded_folder_path, 'cam_a'))
    combos = combinations(list_ids, 2)

    with open(pairings_pos_name, 'wr') as myFile:
        for id in list_ids:
            path_1 = os.path.join(padded_folder_path, id)
            path_2 = os.path.join(padded_folder_path, id)
            myFile.write(str(path_1 + ',' + path_2 + ',1\n'))

    with open(pairings_neg_name, 'wr') as myFile:
        for comb in combos:
            a = comb[0]
            b = comb[1]
            if comb[0] == comb[1]:
                pass
            else:
                path_1 = os.path.join(padded_folder_path, comb[0])
                path_2 = os.path.join(padded_folder_path, comb[1])
                myFile.write(str(path_1 + ',' + path_2 + ',0\n'))


def make_labels_viper(data_file):
    data = np.reshape(data_file, (len(data_file), 3))
    labels = data[:, -1]
    return labels


# takes a image name pair file and loads the images into an ndarray
def load_viper_data_in_array(data):
    data_array = np.zeros(shape=(len(data), pc.NUM_CAMERAS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    for pair in range(0, len(data)):
        for image in range(0,2):
            if image == 0:
                cam = 'cam_a'
            else:
                cam = 'cam_b'

            # change the specific path to your data here
            path = os.path.join('/home/gabi/Documents/datasets/VIPeR/padded', cam, data[pair][image])
            data_array[pair][image] = ndimage.imread(path)

            ## uncomment to display images
            # thing1 = data_array[pair][image]
            # img = Image.fromarray(thing1.astype('uint8')).convert('RGB')
            # img.show()
    return data_array


# loads the viper dataset for use in a person re-id setting in a siamese network
def load_viper(val_pos, test_pos):
    path_validation = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK, 'validation_data_viper.txt')
    path_test = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK,'test_data_viper.txt')
    path_train = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK,'train_data_viper.txt')

    # if validation file doesn't exist assume the other files don't exist either
    if not os.path.exists(path_validation):
        print('creating viper data')
        positive_combo_list = np.genfromtxt('/home/gabi/Documents/datasets/VIPeR/padded/pairings_pos.txt', dtype=None).tolist()
        negative_combo_list = np.genfromtxt('/home/gabi/Documents/datasets/VIPeR/padded/pairings_neg.txt', dtype=None).tolist()

        random.shuffle(negative_combo_list)
        random.shuffle(positive_combo_list)
        negative_combo_list_ = []

        for pick in range(0, len(positive_combo_list)):
            negative_combo_list_.append(negative_combo_list[pick])

        all_list = positive_combo_list + negative_combo_list_
        random.shuffle(all_list)

        test_data, all_list = ddl.make_specific_balanced_set(all_list, positives_percentage=test_pos, set_size=200)
        validation_data, all_list = ddl.make_specific_balanced_set(all_list, positives_percentage=val_pos, set_size=200)
        train_data = all_list

        random.shuffle(test_data)
        random.shuffle(validation_data)
        random.shuffle(train_data)

        print('test: ' + str(ddl.analyze_data_set(test_data)))
        print('validation: ' + str(ddl.analyze_data_set(validation_data)))
        print('train: ' + str(ddl.analyze_data_set(train_data)))

        validation_data_text = path_validation
        with open(validation_data_text, 'wr') as my_file:
            for line in range(0, len(validation_data)):
                my_file.write(str(validation_data[line]) + '\n')

        test_data_text = path_test
        with open(test_data_text, 'wr') as my_file:
            for line in range(0, len(test_data)):
                my_file.write(str(test_data[line]) + '\n')

        train_data_text = path_train
        with open(train_data_text, 'wr') as my_file:
            for line in range(0, len(train_data)):
                my_file.write(str(train_data[line]) + '\n')

        del train_data, test_data, validation_data, positive_combo_list, negative_combo_list, all_list

    print('loading viper data from files')

    train_data = list(csv.reader(np.genfromtxt(path_train, dtype=None)))
    validation_data = list(csv.reader(np.genfromtxt(path_validation, dtype=None)))
    test_data = list(csv.reader(np.genfromtxt(path_test, dtype=None)))

    train_labels = make_labels_viper(train_data)
    validation_labels = make_labels_viper(validation_data)
    test_labels = make_labels_viper(test_data)

    train_data_array = load_viper_data_in_array(train_data)
    validation_data_array = load_viper_data_in_array(validation_data)
    test_data_array = load_viper_data_in_array(test_data)

    return [train_data_array, train_labels, validation_data_array, validation_labels,
            test_data_array, test_labels]


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


def make_confusion_matrix(predictions, labels):
    predictions = threshold_predictions(predictions)
    tp, fp, tn, fn = 0, 0, 0, 0

    if len(np.shape(labels)) > 1:
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


def enter_in_log(experiment_name, file_name, super_main_iterations, test_confusion_matrix, dataset_name, total_time):
# def enter_in_log(name):
    if not os.path.exists(pc.LOG_FILE_PATH):
        with open(pc.LOG_FILE_PATH, 'w') as my_file:
            print('new log file made')


    with open(pc.LOG_FILE_PATH, 'a') as log_file:
        date = str(time.strftime("%d/%m/%Y")) + "   " + str(time.strftime("%H:%M:%S"))
        accuracy = (test_confusion_matrix[0] + test_confusion_matrix[2])*1.0 / (sum(test_confusion_matrix)*1.0)
        confusion_matrix = str(test_confusion_matrix)
        log_file.write('\n')
        log_file.write('name_of_experiment:         %s\n' % experiment_name)
        log_file.write('file_name:                  %s\n' % file_name)
        log_file.write('date:                       %s\n' % date)
        log_file.write('duration:                   %f\n' % total_time)
        log_file.write('data_set:                   %s\n' % dataset_name)
        log_file.write('iterations:                 %d\n' % super_main_iterations)
        log_file.write('start_learning_rate:        %f\n' % pc.START_LEARNING_RATE)
        log_file.write('batch_size:                 %d\n' % pc.BATCH_SIZE)
        log_file.write('similarity_metric           %s\n' % pc.SIMILARITY_METRIC)
        log_file.write('decay_rate:                 %f\n' % pc.DECAY_RATE)
        log_file.write('momentum:                   %f\n' % pc.MOMENTUM)
        log_file.write('epochs:                     %d\n' % pc.NUM_EPOCHS)
        log_file.write('number_of_cameras:          %d\n' % pc.NUM_CAMERAS)
        log_file.write('number_of_siamese_heads:    %d\n' % pc.NUM_SIAMESE_HEADS)
        log_file.write('dropout:                    %f\n' % pc.DROPOUT)
        log_file.write('transfer_learning:          %s\n' % pc.TRANSFER_LEARNING)
        log_file.write('train_cnn:                  %s\n' % pc.TRAIN_CNN)
        log_file.write('mean_tp_fp_tn_fn:           %s\n' % confusion_matrix)
        log_file.write('mean_accuracy:              %f\n' % accuracy)

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

    images_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images'
    pairings_pos_name = '/home/gabi/PycharmProjects/uatu/data/CUHK/positives.txt'
    pairings_neg_name = '/home/gabi/PycharmProjects/uatu/data/CUHK/negatives.txt'

    list_ids = os.listdir(images_path)
    combos = combinations(list_ids, 2)

    with open(pairings_pos_name, 'wr') as posFile:
        with open(pairings_neg_name, 'wr') as negFile:
            for comb in combos:
                if match(comb[0], comb[1]):
                    if comb[0] == comb[1]:
                        pass
                    else:
                        posFile.write(str(comb[0] + ',' + comb[1] + ',1\n'))
                else:
                    negFile.write(str(comb[0] + ',' + comb[1] + ',0\n'))


def make_labels_cuhk1(data_file):
    data = np.reshape(data_file, (len(data_file), 3))
    labels = data[:, -1]
    return labels


def load_cuhk1_data_in_array(data):
    data_array = np.zeros(shape=(len(data), pc.NUM_SIAMESE_HEADS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    for pair in range(0, len(data)):
        print(pair)
        for image in range(0, 2):

            # change the specific path to your data here
            path = os.path.join('/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images', data[pair][image])
            data_array[pair][image] = ndimage.imread(path)
    return data_array


def load_cuhk1(val_pos, test_pos):
    path_validation = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK,'validation_data_cuhk1.txt')
    path_test = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK,'test_data_cuhk1.txt')
    path_train = os.path.join(pc.SAVE_LOCATION_VIPER_CUHK,'train_data_cuhk1.txt')


    if not os.path.exists(path_validation):
        print('creating cuhk1 data')
        positive_combo_list = np.genfromtxt('/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/pairings_pos.txt',
                                            dtype=None).tolist()
        negative_combo_list = np.genfromtxt('/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/pairings_neg.txt',
                                            dtype=None).tolist()

        print('shuffling')
        random.shuffle(negative_combo_list)
        random.shuffle(positive_combo_list)
        negative_combo_list_ = []

        for pick in range(0, len(positive_combo_list)):
            negative_combo_list_.append(negative_combo_list[pick])

        all_list = positive_combo_list + negative_combo_list_
        random.shuffle(all_list)

        test_data, all_list = ddl.make_specific_balanced_set(all_list, positives_percentage=test_pos, set_size=400)
        validation_data, all_list = ddl.make_specific_balanced_set(all_list, positives_percentage=val_pos, set_size=400)
        train_data = all_list

        random.shuffle(test_data)
        random.shuffle(validation_data)
        random.shuffle(train_data)

        print('test: ' + str(ddl.analyze_data_set(test_data)))
        print('validation: ' + str(ddl.analyze_data_set(validation_data)))
        print('train: ' + str(ddl.analyze_data_set(train_data)))

        print('writing cuhk1 names to file')

        validation_data_text = path_validation
        with open(validation_data_text, 'wr') as my_file:
            for line in range(0, len(validation_data)):
                my_file.write(str(validation_data[line]) + '\n')

        test_data_text = path_test
        with open(test_data_text, 'wr') as my_file:
            for line in range(0, len(test_data)):
                my_file.write(str(test_data[line]) + '\n')

        train_data_text = path_train
        with open(train_data_text, 'wr') as my_file:
            for line in range(0, len(train_data)):
                my_file.write(str(train_data[line]) + '\n')

        del train_data, test_data, validation_data, positive_combo_list, negative_combo_list, all_list

    # if validation file doesn't exist assume the other files don't exist either

    print('loading cuhk1 data from files')

    train_data = list(csv.reader(np.genfromtxt(path_train, dtype=None)))
    validation_data = list(csv.reader(np.genfromtxt(path_validation, dtype=None)))
    test_data = list(csv.reader(np.genfromtxt(path_test, dtype=None)))

    train_labels = make_labels_cuhk1(train_data)
    validation_labels = make_labels_cuhk1(validation_data)
    test_labels = make_labels_cuhk1(test_data)

    train_data_array = load_cuhk1_data_in_array(train_data)
    validation_data_array = load_cuhk1_data_in_array(validation_data)
    test_data_array = load_cuhk1_data_in_array(test_data)

    return [train_data_array, train_labels, validation_data_array, validation_labels,
            test_data_array, test_labels]


def load_viper_cuhk1():
    train_data_v, train_labels_v, validation_data_v, validation_labels_v, test_data_v, test_labels_v = load_viper(val_pos=0.3, test_pos=0.1)
    train_data_c, train_labels_c, validation_data_c, validation_labels_c, test_data_c, test_labels_c = load_cuhk1(val_pos=0.3, test_pos=0.1)

    print('asf')
    # test
    test_labels = np.zeros(len(test_labels_v) + len(test_labels_c))
    test_labels[0:len(test_labels_v)] = test_labels_v
    test_labels[len(test_labels_v):] = test_labels_c

    test_data_array = np.zeros(shape=(len(test_data_v) + len(test_data_c), pc.NUM_CAMERAS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    test_data_array[0:len(test_data_v)] = test_data_v
    test_data_array[len(test_data_v):] = test_data_c

    test = list(zip(test_labels, test_data_array))
    rd.shuffle(test)
    test_labels, test_data_array = zip(*test)

    # train
    train_labels = np.zeros(len(train_labels_v) + len(train_labels_c))
    train_labels[0:len(train_labels_v)] = train_labels_v
    train_labels[len(train_labels_v):] = train_labels_c

    train_data_array = np.zeros(
        shape=(len(train_data_v) + len(train_data_c), pc.NUM_CAMERAS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    train_data_array[0:len(train_data_v)] = train_data_v
    train_data_array[len(train_data_v):] = train_data_c

    train = list(zip(train_labels, train_data_array))
    rd.shuffle(train)
    train_labels, train_data_array = zip(*train)
    
    # validation
    validation_labels = np.zeros(len(validation_labels_v) + len(validation_labels_c))
    validation_labels[0:len(validation_labels_v)] = validation_labels_v
    validation_labels[len(validation_labels_v):] = validation_labels_c

    validation_data_array = np.zeros(
        shape=(len(validation_data_v) + len(validation_data_c), pc.NUM_CAMERAS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    validation_data_array[0:len(validation_data_v)] = validation_data_v
    validation_data_array[len(validation_data_v):] = validation_data_c

    validation = list(zip(validation_labels, validation_data_array))
    rd.shuffle(validation)
    validation_labels, validation_data_array = zip(*validation)
    
    
    return [train_data_array, train_labels, validation_data_array, validation_labels, test_data_array, test_labels]


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


def load_NICTA():
    print('Loading NICTA pedestrian dataset')
    base_path = '/home/gabi/Documents/datasets/NICTAPedestrians/'
    data_list_path = '/home/gabi/PycharmProjects/uatu/data/NICTA'

    if not os.path.exists(data_list_path):
        os.mkdir(data_list_path)

        tr_pos_data_path_0 = os.path.join(base_path, 'padded_positives/NICTA_Pedestrian_Positive_Train_Set_A/00000000')
        tr_pos_data_path_1 = os.path.join(base_path, 'padded_positives/NICTA_Pedestrian_Positive_Train_Set_A/00000001')
        tr_pos_data_path_2 = os.path.join(base_path, 'padded_positives/NICTA_Pedestrian_Positive_Train_Set_A/00000002')

        tr_neg_data_path_0 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Train_Set_A/00000000')
        tr_neg_data_path_1 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Train_Set_A/00000001')
        tr_neg_data_path_2 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Train_Set_A/00000002')
        tr_neg_data_path_3 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Train_Set_A/00000003')
        tr_neg_data_path_4 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Train_Set_A/00000004')


        te_pos_data_path_0 = os.path.join(base_path, 'padded_positives/NICTA_Pedestrian_Positive_Valid_Set_A/00000000')
        te_pos_data_path_1 = os.path.join(base_path, 'padded_positives/NICTA_Pedestrian_Positive_Valid_Set_A/00000001')

        te_neg_data_path_0 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Valid_Set_A/00000000')
        te_neg_data_path_1 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Valid_Set_A/00000001')
        te_neg_data_path_2 = os.path.join(base_path, 'padded_negatives/NICTA_Pedestrian_Negative_Valid_Set_A/00000002')

        tr_pos_list = [tr_pos_data_path_0, tr_pos_data_path_1, tr_pos_data_path_2]
        tr_neg_list = [tr_neg_data_path_0, tr_neg_data_path_1, tr_neg_data_path_2, tr_neg_data_path_3, tr_neg_data_path_4]
        te_pos_list = [te_pos_data_path_0, te_pos_data_path_1]
        te_neg_list = [te_neg_data_path_0, te_neg_data_path_1, te_neg_data_path_2]

        list_of_list = [tr_pos_list, tr_neg_list, te_pos_list, te_neg_list]
        # tr_pos, tr_neg, te_pos, te_neg
        numbers = [2416, 4105, 1132, 2244]

        train_data_path = os.path.join(data_list_path, 'train.txt')
        test_data_path = os.path.join(data_list_path, 'test.txt')

        def load_data_into_list(train_or_validate, b, e):
            which_path = train_data_path if train_or_validate == 'train' else test_data_path
            with open(which_path, 'wr') as myFile:
                for item in range(b, e):
                    the_list = list_of_list[item]
                    counter = 0
                    max_files = numbers[item]
                    for sub_list in range(len(the_list)):
                        files_in_dir = os.listdir(the_list[sub_list])
                        for image in files_in_dir:
                            path = os.path.join(the_list[sub_list], image)
                            lab = 1 if item%2 == 0 else 0
                            if counter < max_files:
                                myFile.write(path + ',%d\n' %lab)
                                counter += 1
                            else:
                                break

        load_data_into_list('train', 0, 2)
        load_data_into_list('test', 2, 4)

        # create validation dataset
        test_list_path = os.path.join(data_list_path, 'test.txt')
        validation_list_path = os.path.join(data_list_path, 'validate.txt')
        test_list = np.genfromtxt(test_list_path, dtype=None).tolist()

        rd.shuffle(test_list)
        validation_data = test_list[0:len(test_list) / 2]
        test_data = test_list[len(test_list) / 2:len(test_list)]

        with open(test_list_path, 'wr') as myFile:
            for item in test_data:
                myFile.write(item + '\n')

        with open(validation_list_path, 'wr') as myFile:
            for item in validation_data:
                myFile.write(item + '\n')

    def load_NICTA_data_in_array(data):
        data_array = np.zeros(shape=(len(data), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
        for image in range(0, len(data)):
            name = data[image].split(',')[0]
            data_array[image] = ndimage.imread(name)[:, :, 0:3]
        return data_array

    train_data = np.genfromtxt(os.path.join(data_list_path, 'train.txt'), dtype=None).tolist()
    rd.shuffle(train_data)
    train_data_array = load_NICTA_data_in_array(train_data)
    train_labels = np.asarray([train_data[row].split(',')[1] for row in range(0, len(train_data))])

    test_data = np.genfromtxt(os.path.join(data_list_path, 'test.txt'), dtype=None).tolist()
    rd.shuffle(test_data)
    test_data_array = load_NICTA_data_in_array(test_data)
    test_labels = np.asarray([test_data[row].split(',')[1] for row in range(0, len(test_data))])

    validation_data = np.genfromtxt(os.path.join(data_list_path, 'validate.txt'), dtype=None).tolist()
    rd.shuffle(validation_data)
    validation_data_array = load_NICTA_data_in_array(validation_data)
    validation_labels = np.asarray([validation_data[row].split(',')[1] for row in range(0, len(validation_data))])

    return train_data_array, train_labels, validation_data_array, validation_labels, test_data_array, test_labels


def load_inria_nicta():
    train_data_v, train_labels_v, validation_data_v, validation_labels_v, test_data_v, test_labels_v = load_INRIA()
    train_data_c, train_labels_c, validation_data_c, validation_labels_c, test_data_c, test_labels_c = load_NICTA()

    # print('asf')
    # test
    test_labels = np.zeros(len(test_labels_v) + len(test_labels_c))
    test_labels[0:len(test_labels_v)] = test_labels_v
    test_labels[len(test_labels_v):] = test_labels_c

    test_data_array = np.zeros(
        shape=(len(test_data_v) + len(test_data_c), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    test_data_array[0:len(test_data_v)] = test_data_v
    test_data_array[len(test_data_v):] = test_data_c

    test = list(zip(test_labels, test_data_array))
    rd.shuffle(test)
    test_labels, test_data_array = zip(*test)

    # train
    train_labels = np.zeros(len(train_labels_v) + len(train_labels_c))
    train_labels[0:len(train_labels_v)] = train_labels_v
    train_labels[len(train_labels_v):] = train_labels_c

    train_data_array = np.zeros(
        shape=(len(train_data_v) + len(train_data_c), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    train_data_array[0:len(train_data_v)] = train_data_v
    train_data_array[len(train_data_v):] = train_data_c

    train = list(zip(train_labels, train_data_array))
    rd.shuffle(train)
    train_labels, train_data_array = zip(*train)

    # validation
    validation_labels = np.zeros(len(validation_labels_v) + len(validation_labels_c))
    validation_labels[0:len(validation_labels_v)] = validation_labels_v
    validation_labels[len(validation_labels_v):] = validation_labels_c

    validation_data_array = np.zeros(
        shape=(len(validation_data_v) + len(validation_data_c), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH,
               pc.NUM_CHANNELS))
    validation_data_array[0:len(validation_data_v)] = validation_data_v
    validation_data_array[len(validation_data_v):] = validation_data_c

    validation = list(zip(validation_labels, validation_data_array))
    rd.shuffle(validation)
    validation_labels, validation_data_array = zip(*validation)

    return [train_data_array, train_labels, validation_data_array, validation_labels, test_data_array, test_labels]


def initialize_cnn_data():
    [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = load_inria_nicta()

    train_data = np.asarray(train_data)
    validation_data = np.asarray(validation_data)
    test_data = np.asarray(test_data)

    train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
    validation_labels = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)
    print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))
    return [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]


def initialize_scn_data():
    [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = load_viper_cuhk1()

    train_data = np.asarray(train_data)
    validation_data = np.asarray(validation_data)
    test_data = np.asarray(test_data)

    train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
    validation_labels = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)
    print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))
    return [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]


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
    data_location = '/home/gabi/PycharmProjects/uatu/data'
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

a,b = merge_reid_sets(save=True)