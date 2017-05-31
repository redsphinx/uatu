from keras.models import load_model
import numpy as np
import dynamic_data_loading as ddl
import project_constants as pc
import project_data_handling as pd
import project_utils as pu
import os
import keras
from scipy import ndimage
from PIL import Image
from skimage.util import random_noise
from matplotlib.image import imsave
from itertools import combinations
import time
import matplotlib.pyplot as plt


def zoom(image):
    the_image = image
    image_width, image_height = the_image.size
    image_2 = the_image.crop((image_width*0.15, image_height*0.15, image_width*0.85, image_height*0.85))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def rotate(image):
    the_image = image
    image_2 = the_image.rotate(20)
    image_width, image_height = the_image.size
    image_2 = image_2.crop((image_width * 0.15, image_height * 0.15, image_width * 0.85, image_height * 0.85))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2

def noise(path):
    the_image = ndimage.imread(path)
    image_2 = random_noise(the_image)
    return image_2


def flip(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    return image_2


def create_and_save_augmented_images(keys, the_id, name):
    # augments data. saves data

    if name == 'cuhk02':
        path = '../data/CUHK02/augmented/%s' % the_id
    else:
        path = '../data/market/augmented/%s' % the_id
    # if not os.path.exists(path): os.mkdir(path)
    if not os.path.exists(path): os.makedirs(path)

    for item in keys:
        image = Image.open(item)
        name_bare = item.strip().split('/')[-1].split('.')[0]

        name_original = os.path.join(path, name_bare + '_original.png')
        image.save(name_original)

        # image_zoom = zoom(image)
        # name_zoom = os.path.join(path, name_bare + '_zoom.png')
        # image_zoom.save(name_zoom)
        #
        # image_rotate = rotate(image)
        # name_rotate = os.path.join(path, name_bare + '_rotate.png')
        # image_rotate.save(name_rotate)

        # image_noise = noise(item)
        # name_noise = os.path.join(path, name_bare + '_noise.png')
        # imsave(name_noise, image_noise)
        #
        # image_vertical_flip = flip(image)
        # name_flip = os.path.join(path, name_bare + '_flip.png')
        # image_vertical_flip.save(name_flip)


def load_augmented_images(list_augmented_images):
    combos = list(combinations(list_augmented_images, 2))

    data = np.zeros((len(combos), 2, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, 3))

    for comb in range(len(combos)):
        image_1 = ndimage.imread(combos[comb][0])
        image_2 = ndimage.imread(combos[comb][1])
        data[comb][0] = image_1[:, :, 0:3]
        data[comb][1] = image_2[:, :, 0:3]

    return data


def train_and_test(adjustable, name, this_ranking, model, h5_dataset):
    """ Separately prime the network for each ID in the ranking test:
        For each ID in the ranking test, train the network on a small set of images containing the probe and a couple of
        images belonging to that same ID, but not the same as the image in the gallery.
    """
    full_predictions = np.zeros((len(this_ranking), 2))

    for id in range(pc.RANKING_NUMBER):
        print('ID %d/%d' % (id, pc.RANKING_NUMBER))
        matching_index = id * pc.RANKING_NUMBER + id
        if name == 'cuhk02':
            partition = this_ranking[matching_index].strip().split(',')[0].split('+')[-3]
            folder_name = 'CUHK02'
        else:
            partition = None
            folder_name = 'market'
        image_1 = this_ranking[matching_index].strip().split(',')[0].split('+')[-1]
        image_2 = this_ranking[matching_index].strip().split(',')[1].split('+')[-1]
        seen = [image_1, image_2]
        the_id = pd.my_join(list(image_1)[0:4])

        list_related_keys = ddl.get_related_keys(name, partition, the_id, seen)
        path = '../data/%s/augmented/%s' % (folder_name, the_id)

        if os.path.exists(path):
            if len(os.listdir(path)) == 0:
                create_and_save_augmented_images(list_related_keys, the_id, name)
        else:
            create_and_save_augmented_images(list_related_keys, the_id, name)

        list_augmented_images = os.listdir(path)
        list_augmented_images = [os.path.join(path, item) for item in list_augmented_images]

        # training
        prime_train = load_augmented_images(list_augmented_images)
        training_instances = len(prime_train[:, 0])
        prime_labels = np.ones(training_instances, dtype=int)
        prime_labels = keras.utils.to_categorical(prime_labels, pc.NUM_CLASSES)

        weight_path = os.path.join('../model_weights', adjustable.load_weights_name)
        model.load_weights(weight_path)

        model.fit([prime_train[:, 0], prime_train[:, 1]], prime_labels,
                  batch_size=adjustable.batch_size,
                  epochs=adjustable.prime_epochs,
                  verbose=2)

        # testing
        part_ranking = this_ranking[id * pc.RANKING_NUMBER:(id + 1) * pc.RANKING_NUMBER]
        test_data = ddl.grab_em_by_the_keys(part_ranking, h5_dataset)

        test_data = np.asarray(test_data)

        part_prediction = model.predict([test_data[0], test_data[1]])
        full_predictions[id * pc.RANKING_NUMBER:(id + 1) * pc.RANKING_NUMBER] = part_prediction

    return full_predictions


def only_test(model, h5_dataset, this_ranking):
    """ Only runs testing
    """
    test_data = ddl.grab_em_by_the_keys(this_ranking, h5_dataset)
    test_data = np.asarray(test_data)
    part_prediction = model.predict([test_data[0], test_data[1]])
    full_predictions = part_prediction

    return full_predictions


def main(adjustable):
    start = time.time()
    cuhk02_ranking = list(np.genfromtxt('cuhk02_ranking.txt', dtype=None))
    market_ranking = list(np.genfromtxt('market_ranking.txt', dtype=None))

    all_ranking = [cuhk02_ranking, market_ranking]
    names = ['cuhk02', 'market']
    confusion_matrices = []
    ranking_matrices = []

    path = os.path.join('../model_weights', adjustable.load_model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu
    model = load_model(path)

    for item in range(len(all_ranking)):
        name = names[item]
        this_ranking = all_ranking[item]

        h5_dataset = ddl.load_datasets_from_h5([name])
        if adjustable.only_test:
            full_predictions = only_test(model, h5_dataset, this_ranking)
        else:
            full_predictions = train_and_test(adjustable, name, this_ranking, model, h5_dataset)

        # putting it all together
        final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in
                                range(len(this_ranking))]
        final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

        matrix = pu.make_confusion_matrix(full_predictions, final_testing_labels)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0
        confusion_matrices.append(matrix)

        ranking = pu.calculate_CMC(full_predictions)
        ranking_matrices.append(ranking)

        print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
              % (name, accuracy, precision, str(matrix), str(ranking)))

    stop = time.time()
    total_time = stop - start
    number_of_datasets = 2
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_std = np.zeros((number_of_datasets, pc.RANKING_NUMBER))

    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable.experiment_name, file_name, names, confusion_matrices, matrix_std, ranking_matrices,
                        ranking_std,
                        total_time)