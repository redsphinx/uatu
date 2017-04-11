import numpy as np
import project_constants as pc
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


# generate noise to test if siamese_cnn pipeline is working
def load_data():
    print('loading data')
    train = np.array([])
    validate = np.array([])
    for num in range(0, pc.AMOUNT_DATA):
        print(num)
        imarray = np.random.rand(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS) * 255
        im_t_1 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        im_t_2 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        train = np.append(train, [im_t_1, im_t_2])
        imarray = np.random.rand(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS) * 255
        im_v_1 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        im_v_2 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        validate = np.append(validate, [im_v_1, im_v_2])

    train = train.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    validate = validate.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    ans = [train, validate]
    return ans


# generate shitty labels to test if the siamese_cnn is working
def load_labels():
    a = np.array([])
    b = np.array([])
    for num in range(0, pc.AMOUNT_DATA):
        a = np.append(a, [1,0])
        b = np.append(b, [0,1])

    a = a.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES])
    b = b.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES])
    return [a, b]


# used to calculate error
def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


# crop images in center
def crop_images(folder_path, width, height):
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
        img_width, img_height = img.size


        center_x = img_width / 2
        center_y = img_height / 2
        start_x = center_x - width / 2
        start_y = center_y - height / 2
        img2 = img.crop((start_x, start_y, start_x + width, start_y + height))
        img2.save(os.path.join(new_folder_path, image_path))

    pass


def load_INRIA_data(path):

    print('loading data')
    # load into lists
    train_data_names_pos_ = np.genfromtxt(os.path.join(path, 'train_data_names_pos.csv'), dtype=None).tolist()
    train_data_names_neg_ = np.genfromtxt(os.path.join(path, 'train_data_names_neg.csv'), dtype=None).tolist()
    train_data_labels_pos_ = np.genfromtxt(os.path.join(path, 'train_data_labels_pos.csv'), dtype=None).tolist()
    train_data_labels_neg_ = np.genfromtxt(os.path.join(path, 'train_data_labels_neg.csv'), dtype=None).tolist()
    validation_data_names_pos_ = np.genfromtxt(os.path.join(path, 'validation_data_names_pos.csv'), dtype=None).tolist()
    validation_data_names_neg_ = np.genfromtxt(os.path.join(path, 'validation_data_names_neg.csv'), dtype=None).tolist()
    validation_data_labels_pos_ = np.genfromtxt(os.path.join(path, 'validation_data_labels_pos.csv'), dtype=None).tolist()
    validation_data_labels_neg_ = np.genfromtxt(os.path.join(path, 'validation_data_labels_neg.csv'), dtype=None).tolist()

    # TODO (optional) if images need cropping, use the 'crop_images()' function above separately

    # shuffle
    train_data_ = train_data_names_pos_ + train_data_names_neg_
    train_labels_ = train_data_labels_pos_ + train_data_labels_neg_

    validation_data_ = validation_data_names_pos_ + validation_data_names_neg_
    validation_labels_ = validation_data_labels_pos_ + validation_data_labels_neg_

    everything = list(zip(train_data_, train_labels_))
    rd.shuffle(everything)
    train_data_, train_labels_ = zip(*everything)

    everything = list(zip(validation_data_, validation_labels_))
    rd.shuffle(everything)
    validation_data_, validation_labels_ = zip(*everything)

    test_images_names = 'test_images.csv'
    with open(test_images_names, 'wr') as my_file:
        for line in range(len(validation_data_)/2, len(validation_data_)):
            my_file.write(str(validation_data_[line]) + '\n')

    test_images_labels = 'test_images_labels.csv'
    with open(test_images_labels, 'wr') as my_file:
        for line in range(len(validation_labels_ )/2, len(validation_labels_)):
            my_file.write(str(validation_labels_[line]) + '\n')

    # create empty arrays
    train_data_array = np.zeros(shape=(len(train_data_), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    train_labels_array = np.zeros(shape=(len(train_labels_), pc.NUM_CLASSES))
    validation_data_array = np.zeros(shape=(len(validation_data_), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    validation_labels_array = np.zeros(shape=(len(validation_labels_), pc.NUM_CLASSES))

    def ohe(old, new):
        for item in range(0, len(old)):
            new[item][old[item]] = 1
        return new

    for image in range(0, len(train_data_)):
        train_data_array[image] = ndimage.imread(train_data_[image])[:, :, 0:3]
    for image in range(0, len(validation_data_)):
        validation_data_array[image] = ndimage.imread(validation_data_[image])[:, :, 0:3]

    train_labels = ohe(train_labels_, train_labels_array)
    validation_labels = ohe(validation_labels_, validation_labels_array)

    # temp to test CNN
    train_labels = train_labels_
    validation_labels = validation_labels_

    return [train_data_array, train_labels, validation_data_array, validation_labels]
    pass


# load a specific dataset
def load_human_detection_data():
    # load data
    location_path = '/home/gabi/Documents/datasets/INRIAPerson'
    train_path = 'train_64x128_H96'
    validation_path = 'test_64x128_H96'

    data_list_folder = '/home/gabi/PycharmProjects/uatu/data/INRIA'

    if not os.path.exists(data_list_folder):
        print('folder does not exist, made it')
        os.makedirs(data_list_folder)
    if len(os.listdir(data_list_folder)) <= 0:
        print('folder empty, making files')
        # write list then load data
        train_data_path_pos = os.path.join(location_path, train_path, 'real_cropped_images_pos')
        train_data_names_pos = os.listdir(train_data_path_pos)
        with open(os.path.join(data_list_folder, 'train_data_names_pos.csv'), 'wr') as my_file:
            for item in train_data_names_pos:
                my_file.write(str(os.path.join(train_data_path_pos, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'train_data_labels_pos.csv'), 'wr') as my_file:
            for item in train_data_names_pos:
                my_file.write(str(1) + '\n')

        train_data_path_neg = os.path.join(location_path, train_path, 'real_cropped_images_neg')
        train_data_names_neg = os.listdir(train_data_path_neg)
        with open(os.path.join(data_list_folder, 'train_data_names_neg.csv'), 'wr') as my_file:
            for item in train_data_names_neg:
                my_file.write(str(os.path.join(train_data_path_neg, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'train_data_labels_neg.csv'), 'wr') as my_file:
            for item in train_data_names_neg:
                my_file.write(str(0) + '\n')

        validation_data_path_pos = os.path.join(location_path, validation_path, 'real_cropped_images_pos')
        validation_data_names_pos = os.listdir(validation_data_path_pos)
        with open(os.path.join(data_list_folder, 'validation_data_names_pos.csv'), 'wr') as my_file:
            for item in validation_data_names_pos:
                my_file.write(str(os.path.join(validation_data_path_pos, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'validation_data_labels_pos.csv'), 'wr') as my_file:
            for item in validation_data_names_pos:
                my_file.write(str(1) + '\n')

        validation_data_path_neg = os.path.join(location_path, validation_path, 'real_cropped_images_neg')
        validation_data_names_neg = os.listdir(validation_data_path_neg)
        with open(os.path.join(data_list_folder, 'validation_data_names_neg.csv'), 'wr') as my_file:
            for item in validation_data_names_neg:
                my_file.write(str(os.path.join(validation_data_path_neg, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'validation_data_labels_neg.csv'), 'wr') as my_file:
            for item in validation_data_names_neg:
                my_file.write(str(0) + '\n')

    data = load_INRIA_data(data_list_folder)
    return data


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
                bla = paths[line].split('/')
                thing = paths[line].split('/')[-1]
                copyfile(paths[line], os.path.join(folder, thing))


# image has to be 64x128, this adds padding
def fix_viper():
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    # cam_a_o = '/home/gabi/Documents/datasets/VIPeR/cam_a'
    # cam_b_o = '/home/gabi/Documents/datasets/VIPeR/cam_b'
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
        for file in os.listdir(cam_path):
            img = Image.open(os.path.join(cam_path, file))
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255,255,255))

            img_width, img_height = img.size
            new_img_width, new_img_height = new_img.size
            padding_width = (new_img_width-img_width)/2
            padding_height = (new_img_height-img_height)/2

            new_img.paste(img, box=(padding_width, padding_height))

            filename = file.split('_')[0] + '.bmp'
            filename = os.path.join(padded_cam_path, filename)
            new_img.save(filename)

    # it throws an error but it does the job

# make matching and non-matching pairs
def make_pairs_viper():
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    pairings_neg_name = 'pairings_neg.txt'
    pairings_pos_name = 'pairings_pos.txt'

    pairings_neg = open(os.path.join(padded_folder_path, pairings_neg_name), "wr")
    pairings_pos = open(os.path.join(padded_folder_path, pairings_pos_name), "wr")

    list_ids = os.listdir(os.path.join(padded_folder_path, 'cam_a'))
    combos = combinations(list_ids, 2)

    for comb in combos:
        a = comb[0]
        b = comb[1]
        if comb[0] == comb[1]:
            pass
        else:
            pairings_neg.write(str(comb[0] + ',' + comb[1] + ',0\n'))

    pairings_neg.close()

    for id in list_ids:
        pairings_pos.write(str(id + ',' + id + ',1\n'))

    pairings_pos.close()


def make_labels_viper(data_file):
    data = np.reshape(data_file, (len(data_file), 3))
    labels = data[:, -1]
    return labels


# takes a image name pair file and loads the images into an ndarray
def load_data_in_array(data):
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


def analyze_data_set(dataset):
    data_list = list(csv.reader(np.genfromtxt(dataset, dtype=None)))
    labels = np.asarray([data_list[row][2] for row in range(0, len(data_list))], dtype=int)
    positives_percentage = np.sum(labels) * 1.0 / len(labels)
    negatives_percentage = 1.0 - positives_percentage
    # print('positives: %0.2f  negatives: %0.2f' % (positives_percentage, negatives_percentage))
    return [positives_percentage, negatives_percentage]


def make_specific_balanced_set(dataset, positives_percentage, set_size):
    data_list = np.asarray(dataset)
    labels = np.asarray([dataset[row].split(',')[2] for row in range(0, len(dataset))])
    num_of_positives = positives_percentage * set_size
    test_data = []
    new_data_list = []
    count_pos = 0
    count_neg = 0
    for row in range(0, len(data_list)):
        if labels[row] == '1' and count_pos < num_of_positives:
            test_data.append(dataset[row])
            count_pos += 1
        elif labels[row] == '0' and count_neg < set_size - num_of_positives:
            test_data.append(dataset[row])
            count_neg += 1
        else:
            new_data_list.append(dataset[row])
    return test_data, new_data_list


# loads the viper dataset for use in a person re-id setting in a siamese network
def load_viper():
    path_validation = 'validation_data_viper.txt'
    path_test = 'test_data_viper.txt'
    path_train = 'train_data_viper.txt'

    # if validation file doesn't exist assume the other files don't exist either
    if os.path.exists(path_validation):
        print('loading viper data from files')

        train_data = list(csv.reader(np.genfromtxt(path_train, dtype=None)))
        validation_data = list(csv.reader(np.genfromtxt(path_validation, dtype=None)))
        test_data = list(csv.reader(np.genfromtxt(path_test, dtype=None)))

        train_labels = make_labels_viper(train_data)
        validation_labels = make_labels_viper(validation_data)
        test_labels = make_labels_viper(test_data)

        train_data_array = load_data_in_array(train_data)
        validation_data_array = load_data_in_array(validation_data)
        test_data_array = load_data_in_array(test_data)

        return [train_data_array, train_labels, validation_data_array, validation_labels,
                test_data_array, test_labels]


    else:
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

        test_data, all_list = make_specific_balanced_set(all_list, positives_percentage=0.1, set_size=50)
        validation_data, all_list = make_specific_balanced_set(all_list, positives_percentage=0.3, set_size=50)
        train_data = all_list

        random.shuffle(test_data)
        random.shuffle(validation_data)
        random.shuffle(train_data)

        print('test: ' + str(analyze_data_set(test_data)))
        print('validation: ' + str(analyze_data_set(validation_data)))
        print('train: ' + str(analyze_data_set(train_data)))

        validation_data_text = 'validation_data_viper.txt'
        with open(validation_data_text, 'wr') as my_file:
            for line in range(0, len(validation_data)):
                my_file.write(str(validation_data[line]) + '\n')

        test_data_text = 'test_data_viper.txt'
        with open(test_data_text, 'wr') as my_file:
            for line in range(0, len(test_data)):
                my_file.write(str(test_data[line]) + '\n')

        train_data_text = 'train_data_viper.txt'
        with open(train_data_text, 'wr') as my_file:
            for line in range(0, len(train_data)):
                my_file.write(str(train_data[line]) + '\n')

        load_viper()

    pass


def viper_from_raw():
    # add padding and save in new folder
    fix_viper()
    # make a text list with pairings in the new folder
    make_pairs_viper()


# load_viper()


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


def enter_in_log(experiment_name, file_name, super_main_iterations, test_confusion_matrix, dataset_name):
# def enter_in_log(name):
    if not os.path.exists(pc.LOG_FILE_PATH):
        with open(pc.LOG_FILE_PATH, 'w') as my_file:
            print('new log file made')


    with open(pc.LOG_FILE_PATH, 'a') as log_file:
        date = str(time.strftime("%d/%m/%Y")) + "   " + str(time.strftime("%H:%M:%S"))
        accuracy = (test_confusion_matrix[0] + test_confusion_matrix[2]) / pc.TEST_DATA_SIZE
        confusion_matrix = str(test_confusion_matrix)
        log_file.write('\n')
        log_file.write('name_of_experiment:         %s\n' %experiment_name)
        log_file.write('file_name:                  %s\n' %file_name)
        log_file.write('date:                       %s\n' %date)
        log_file.write('data_set:                   %s\n' %dataset_name)
        log_file.write('iterations:                 %d\n' %super_main_iterations)
        log_file.write('start_learning_rate:        %f\n' %pc.START_LEARNING_RATE)
        log_file.write('batch_size:                 %d\n' %pc.BATCH_SIZE)
        log_file.write('similarity_metric           %s\n' %pc.SIMILARITY_METRIC)
        log_file.write('decay_rate:                 %f\n' %pc.DECAY_RATE)
        log_file.write('momentum:                   %f\n' %pc.MOMENTUM)
        log_file.write('epochs:                     %d\n' %pc.NUM_EPOCHS)
        log_file.write('number_of_cameras:          %d\n' %pc.NUM_CAMERAS)
        log_file.write('number_of_siamese_heads:    %d\n' % pc.NUM_SIAMESE_HEADS)
        log_file.write('dropout:                    %f\n' %pc.DROPOUT)
        log_file.write('transfer_learning:          %s\n' %pc.TRANSFER_LEARNING)
        log_file.write('train_cnn:                  %s\n' %pc.TRAIN_CNN)
        log_file.write('mean_tp_fp_tn_fn:           %s\n' %confusion_matrix)
        log_file.write('mean_accuracy:              %f\n' %accuracy)

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


def match(one, two):
    return list(one)[0:4] == list(two)[0:4]


def make_pairs_cuhk1():
    folder_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/'
    images_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images'
    pairings_neg_name = 'pairings_neg.txt'
    pairings_pos_name = 'pairings_pos.txt'
    pairings_neg = open(os.path.join(folder_path, pairings_neg_name), "wr")
    pairings_pos = open(os.path.join(folder_path, pairings_pos_name), "wr")
    list_ids = os.listdir(images_path)
    combos = combinations(list_ids, 2)
    for comb in combos:
        if match(comb[0], comb[1]):
            if comb[0] == comb[1]:
                pass
            else:
                pairings_pos.write(str(comb[0] + ',' + comb[1] + ',1\n'))
        else:
            pairings_neg.write(str(comb[0] + ',' + comb[1] + ',0\n'))

    pairings_neg.close()
    pairings_pos.close()


def make_labels_cuhk1(data_file):
    data = np.reshape(data_file, (len(data_file), 3))
    labels = data[:, -1]
    return labels


def load_cuhk1_data_in_array(data):
    data_array = np.zeros(shape=(len(data), pc.NUM_SIAMESE_HEADS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    for pair in range(0, len(data)):
        print(pair)
        for image in range(0,2):

            # change the specific path to your data here
            path = os.path.join('/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images', data[pair][image])
            data_array[pair][image] = ndimage.imread(path)
    return data_array




def load_cuhk1():
    path_validation = 'validation_data_cuhk1.txt'
    path_test = 'test_data_cuhk1.txt'
    path_train = 'train_data_cuhk1.txt'

    # if validation file doesn't exist assume the other files don't exist either
    if os.path.exists(path_validation):
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


    else:
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

        test_data, all_list = make_specific_balanced_set(all_list, positives_percentage=0.1, set_size=100)
        validation_data, all_list = make_specific_balanced_set(all_list, positives_percentage=0.3, set_size=100)
        train_data = all_list

        random.shuffle(test_data)
        random.shuffle(validation_data)
        random.shuffle(train_data)

        print('test: ' + str(analyze_data_set(test_data)))
        print('validation: ' + str(analyze_data_set(validation_data)))
        print('train: ' + str(analyze_data_set(train_data)))

        print('writing cuhk1 names to file')

        validation_data_text = 'validation_data_cuhk1.txt'
        with open(validation_data_text, 'wr') as my_file:
            for line in range(0, len(validation_data)):
                my_file.write(str(validation_data[line]) + '\n')

        test_data_text = 'test_data_cuhk1.txt'
        with open(test_data_text, 'wr') as my_file:
            for line in range(0, len(test_data)):
                my_file.write(str(test_data[line]) + '\n')

        train_data_text = 'train_data_cuhk1.txt'
        with open(train_data_text, 'wr') as my_file:
            for line in range(0, len(train_data)):
                my_file.write(str(train_data[line]) + '\n')

        load_cuhk1()


def load_viper_cuhk1():
    # TODO fix the nonetype error
    train_data_v, train_labels_v, validation_data_v, validation_labels_v, test_data_v, test_labels_v = load_viper()
    train_data_c, train_labels_c, validation_data_c, validation_labels_c, test_data_c, test_labels_c = load_cuhk1()

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


