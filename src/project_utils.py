import numpy as np
import project_constants as pc
from PIL import Image
import os
import random as rd
from scipy import ndimage
from shutil import copyfile
import shutil
from itertools import combinations



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
    new_folder_path = os.path.join(new_path, 'real_cropped_images_' + str(name_folder))
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


# image has to be 64x128
def fix_viper():
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    # cam_a_o = '/home/gabi/Documents/datasets/VIPeR/cam_a'
    # cam_b_o = '/home/gabi/Documents/datasets/VIPeR/cam_b'
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'
    cam_a_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_a'
    cam_b_p = '/home/gabi/Documents/datasets/VIPeR/padded/cam_b'


    # # assuming they don't exist yet
    # os.mkdir(padded_folder_path)
    # os.mkdir(cam_a_p)
    # os.mkdir(cam_b_p)

    for folder in os.listdir(original_folder_path):
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
            new_img.save(os.path.join(padded_cam_path, file))

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


# loads the viper dataset for use in a person re-id setting in a siamese network
def load_viper():


    pass


make_pairs_viper()