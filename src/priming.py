# from tensorflow.contrib.keras import models
# import tensorflow.contrib.keras as keras

from keras import models, layers, optimizers, losses
import keras

import numpy as np
import dynamic_data_loading as ddl
import project_constants as pc
import data_pipeline as dp
import project_utils as pu
import os
from scipy import ndimage
from PIL import Image
from skimage.util import random_noise
from itertools import combinations
import time
import random

import siamese_cnn_image as scn

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
    elif name == 'market':
        path = '../data/market/augmented/%s' % the_id
    elif name == 'grid':
        path = '../data/GRID/augmented/%s' % the_id
    elif name == 'viper':
        path = '../data/VIPER/augmented/%s' % the_id
    elif name == 'prid450':
        path = '../data/prid450/augmented/%s' % the_id
    else:
        path = None

    if not os.path.exists(path):
        os.makedirs(path)

    for item in keys:
        image = Image.open(item)
        name_bare = item.strip().split('/')[-1].split('.')[0]

        name_original = os.path.join(path, name_bare + '_original.png')
        image.save(name_original)

        image_zoom = zoom(image)
        name_zoom = os.path.join(path, name_bare + '_zoom.png')
        image_zoom.save(name_zoom)

        image_rotate = rotate(image)
        name_rotate = os.path.join(path, name_bare + '_rotate.png')
        image_rotate.save(name_rotate)

        # image_noise = noise(item)
        # name_noise = os.path.join(path, name_bare + '_noise.png')
        # imsave(name_noise, image_noise)

        image_vertical_flip = flip(image)
        name_flip = os.path.join(path, name_bare + '_flip.png')
        image_vertical_flip.save(name_flip)


def is_match(comb):
    i1 = comb[0]
    i2 = comb[1]
    key1 = i1.strip().split('/')[2]
    key2 = i2.strip().split('/')[2]

    if key1 == key2:
        h1 = i1.strip().split('/')[-1].split('_')[0]
        h2 = i2.strip().split('/')[-1].split('_')[0]
        if h1 == h2:
            return 1
        else:
            return 0
    else:
        return 0


def load_augmented_images(list_augmented_images):
    combos = list(combinations(list_augmented_images, 2))

    random.shuffle(combos)

    data = np.zeros((len(combos), 2, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, 3))
    labels = []

    for comb in range(len(combos)):
        image_1 = ndimage.imread(combos[comb][0])
        image_2 = ndimage.imread(combos[comb][1])

        labels.append(is_match(combos[comb]))

        data[comb][0] = image_1[:, :, 0:3]
        data[comb][1] = image_2[:, :, 0:3]

    labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
    return data, labels


def train_and_test(adjustable, name, this_ranking, model, h5_dataset):
    """ Separately prime the network for each ID in the ranking test:
        For each ID in the ranking test, train the network on a small set of images containing the probe and a couple of
        images belonging to that same ID, but not the same as the image in the gallery.
    """
    full_predictions = np.zeros((len(this_ranking), 2))
    
    if adjustable.ranking_number == 'half':
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(adjustable.ranking_number, int):
        ranking_number = adjustable.ranking_number
    else:
        ranking_number = None


    for an_id in range(ranking_number):
        print('ID %d/%d' % (an_id, ranking_number))
        matching_pair_index = an_id * ranking_number + an_id
        if name == 'cuhk02':
            partition = this_ranking[matching_pair_index].strip().split(',')[0].split('+')[-3]
            folder_name = 'CUHK02'
        elif name == 'market':
            partition = None
            folder_name = 'market'
        elif name == 'grid':
            partition = None
            folder_name = 'GRID'
        elif name == 'viper':
            partition = None
            folder_name = 'VIPER'
        elif name == 'prid450':
            partition = None
            folder_name = 'prid450'
        else:
            partition = None
            folder_name = None

        image_1 = this_ranking[matching_pair_index].strip().split(',')[0].split('+')[-1]
        image_2 = this_ranking[matching_pair_index].strip().split(',')[1].split('+')[-1]
        seen = [image_1, image_2]
        the_id = dp.my_join(list(image_1)[0:4])

        list_related_keys = ddl.get_related_keys(adjustable, name, partition, seen, this_ranking, the_id)
        path = '../data/%s/augmented/%s' % (folder_name, the_id)

        if os.path.exists(path):
            if len(os.listdir(path)) == 0:
                create_and_save_augmented_images(list_related_keys, the_id, name)
        else:
            create_and_save_augmented_images(list_related_keys, the_id, name)

        list_augmented_images_short = os.listdir(path)
        list_augmented_images_long = [os.path.join(path, item) for item in list_augmented_images_short]

        # training
        prime_train, prime_labels = load_augmented_images(list_augmented_images_long)

        weight_path = os.path.join('../model_weights', adjustable.load_weights_name)
        model.load_weights(weight_path, by_name=True)

        model.fit([prime_train[:, 0], prime_train[:, 1]], prime_labels,
                  batch_size=adjustable.batch_size,
                  epochs=adjustable.prime_epochs,
                  verbose=2)

        # testing
        part_ranking = this_ranking[an_id * ranking_number:(an_id + 1) * ranking_number]
        test_data = ddl.grab_em_by_the_keys(part_ranking, h5_dataset)

        test_data = np.asarray(test_data)

        part_prediction = model.predict([test_data[0], test_data[1]])
        full_predictions[an_id * ranking_number:(an_id + 1) * ranking_number] = part_prediction

    return full_predictions


def only_test(model, h5_dataset, this_ranking):
    """ Only runs testing
    """
    test_data = ddl.grab_em_by_the_keys(this_ranking, h5_dataset)
    test_data = np.asarray(test_data)
    part_prediction = model.predict([test_data[0], test_data[1]])
    full_predictions = part_prediction

    return full_predictions


def main(adjustable, all_ranking, names, model):
    confusion_matrices = []
    ranking_matrices = []

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

        matrix = pu.make_confusion_matrix(adjustable, full_predictions, final_testing_labels)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0
        confusion_matrices.append(matrix)

        ranking = pu.calculate_CMC(adjustable, full_predictions)
        ranking_matrices.append(ranking)

        print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
              % (name, accuracy, precision, str(matrix), str(ranking)))

    return confusion_matrices, ranking_matrices


def super_main(adjustable):
    name = adjustable.datasets[0]
    print('name: %s' % name)
    start = time.time()

    if name == 'cuhk02':
        dataset_ranking = list(np.genfromtxt('../ranking_files/cuhk02_ranking_%s.txt' % adjustable.use_gpu, dtype=None))
    elif name == 'market':
        dataset_ranking = list(np.genfromtxt('../ranking_files/market_ranking_%s.txt' % adjustable.use_gpu, dtype=None))
    elif name == 'grid':
        dataset_ranking = list(np.genfromtxt('../ranking_files/grid_ranking_%s.txt' % adjustable.use_gpu, dtype=None))
    elif name == 'viper':
        dataset_ranking = list(np.genfromtxt('../ranking_files/viper_ranking_%s.txt' % adjustable.use_gpu, dtype=None))
    elif name == 'prid450':
        dataset_ranking = list(np.genfromtxt('../ranking_files/prid450_ranking_%s.txt' % adjustable.use_gpu, dtype=None))
    else:
        dataset_ranking = None
    # all_ranking = [cuhk02_ranking, market_ranking]
    # names = ['cuhk02', 'market']

    all_ranking = [dataset_ranking]
    names = [name]

    # if adjustable.only_test:
    #     viper_ranking = list(np.genfromtxt('viper_ranking.txt', dtype=None))
    #     grid_ranking = list(np.genfromtxt('grid_ranking.txt', dtype=None))
    #     caviar_ranking = list(np.genfromtxt('caviar_ranking.txt', dtype=None))
    #     prid450_ranking = list(np.genfromtxt('prid450_ranking.txt', dtype=None))
    #     all_ranking.append(viper_ranking)
    #     all_ranking.append(grid_ranking)
    #     all_ranking.append(caviar_ranking)
    #     all_ranking.append(prid450_ranking)
    #     names.append('viper')
    #     names.append('grid')
    #     names.append('caviar')
    #     names.append('prid450')

    # path = os.path.join('../model_weights', adjustable.load_model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu

    # FIXME: add this feature
    # TODO: add option to make a model from scratch, load weights and then compile. This way we can adjust the LR

    # model = models.load_model(path)

    '''
    '''

    if adjustable.load_model_name is not None and adjustable.load_weights_name is None:
        model = models.load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_model_name))
    elif adjustable.load_weights_name is not None and adjustable.load_model_name is None:
        model = scn.create_siamese_network(adjustable)

        the_path = os.path.join('../model_weights', adjustable.load_weights_name)
        model.load_weights(the_path, by_name=True)

        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
            model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])
        elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            rms = keras.optimizers.RMSprop()
            model.compile(loss=scn.contrastive_loss, optimizer=rms, metrics=[scn.absolute_distance_difference])
    else:
        model = None
    '''
    '''

    # FIXME: set only test to 1 dataset
    number_of_datasets = len(names)
    if adjustable.only_test:
        number_of_datasets = 6

    if adjustable.ranking_number == 'half':
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(adjustable.ranking_number, int):
        ranking_number = adjustable.ranking_number

    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, ranking_number))

    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)

        confusion, ranking = main(adjustable, all_ranking, names, model)

        confusion_matrices[iter] = confusion
        ranking_matrices[iter] = ranking

    stop = time.time()
    total_time = stop - start

    matrix_means = np.zeros((number_of_datasets, 4))
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_means = np.zeros((number_of_datasets, ranking_number))
    ranking_std = np.zeros((number_of_datasets, ranking_number))

    for dataset in range(number_of_datasets):
        matrices = np.zeros((adjustable.iterations, 4))
        rankings = np.zeros((adjustable.iterations, ranking_number))

        for iter in range(adjustable.iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]

        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)

    # note: TURN ON if you want to log results!!
    if adjustable.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable, adjustable.experiment_name, file_name, names, matrix_means, matrix_std, ranking_means,
                        ranking_std,
                        total_time)
