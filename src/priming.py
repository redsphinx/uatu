# from tensorflow.contrib.keras import models
# import tensorflow.contrib.keras as keras

from keras import models, layers, optimizers, losses
import keras
import numpy as np
# import dynamic_data_loading as ddl
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
from clr_callback import *
import shutil


def zoom(image):
    the_image = image
    image_2 = the_image.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def rotate(image):
    the_image = image
    image_2 = the_image.rotate(4)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2

def flip(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    return image_2

def flip_zoom(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def flip_rotate(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    image_2 = image_2.rotate(4)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


# unused
def noise(path):
    the_image = ndimage.imread(path)
    image_2 = random_noise(the_image)
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

        image_vertical_flip = flip(image)
        name_flip = os.path.join(path, name_bare + '_flip.png')
        image_vertical_flip.save(name_flip)

        image_flip_zoom = flip_zoom(image)
        name_flip_zoom = os.path.join(path, name_bare + '_flip_zoom.png')
        image_flip_zoom.save(name_flip_zoom)

        image_flip_rotate = flip_rotate(image)
        name_flip_rotate = os.path.join(path, name_bare + '_flip_rotate.png')
        image_flip_rotate.save(name_flip_rotate)

        # image_noise = noise(item)
        # name_noise = os.path.join(path, name_bare + '_noise.png')
        # imsave(name_noise, image_noise)




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

    match_count = 0
    mismatch_count = 0

    for comb in range(len(combos)):
        image_1 = ndimage.imread(combos[comb][0])
        image_2 = ndimage.imread(combos[comb][1])

        labels.append(is_match(combos[comb]))
        if is_match(combos[comb]):
            match_count += 1
        else:
            mismatch_count += 1

        data[comb][0] = image_1[:, :, 0:3]
        data[comb][1] = image_2[:, :, 0:3]

    print('match: ', match_count)
    print('mismatch: ', mismatch_count)

    labels = keras.utils.to_categorical(labels, pc.NUM_CLASSES)
    return data, labels


def train_and_test(adjustable, name, this_ranking, model, h5_dataset):
    """ Separately prime the network for each ID in the ranking test:
        For each ID in the ranking test, train the network on a small set of images containing the probe and a couple of
        images belonging to that same ID, but not the same as the image in the gallery.
    """
    full_predictions = np.zeros((len(this_ranking), 2))
    
    if adjustable.ranking_number_test == 'half':
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(adjustable.ranking_number_test, int):
        ranking_number = adjustable.ranking_number_test
    else:
        print("ranking_number_test must be 'half' or an int")
        return


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
        the_id = pu.my_join(list(image_1)[0:4])

        list_related_keys = dp.get_related_keys(adjustable, name, partition, seen, this_ranking, the_id)
        path = '../data/%s/augmented/%s' % (folder_name, the_id)


        if os.path.exists(path):
            # remove the folder
            shutil.rmtree(path, ignore_errors=True)

        create_and_save_augmented_images(list_related_keys, the_id, name)

        # if os.path.exists(path):
        #     if len(os.listdir(path)) == 0:
        #         create_and_save_augmented_images(list_related_keys, the_id, name)
        # else:
        #     create_and_save_augmented_images(list_related_keys, the_id, name)

        list_augmented_images_short = os.listdir(path)
        list_augmented_images_long = [os.path.join(path, item) for item in list_augmented_images_short]

        # training
        prime_train, prime_labels = load_augmented_images(list_augmented_images_long)

        # model is compiled in get_model
        weight_path = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, '%s_weights.h5' % adjustable.load_weights_name)
        model.load_weights(weight_path, by_name=True)

        if adjustable.use_cyclical_learning_rate:
            # choose step_size=8, because number of samples < batchsize, so per epoch there is 1 step
            # paper recommends multiplying steps/epoch by a number between 2-10. so we choose 8
            # clr = CyclicLR(step_size=(len(prime_labels) / adjustable.batch_size) * 8,
            clr=CyclicLR(step_size=8,
                           base_lr=adjustable.cl_min,
                           max_lr=adjustable.cl_max)

            model.fit([prime_train[:, 0], prime_train[:, 1]], prime_labels,
                      batch_size=adjustable.batch_size,
                      epochs=adjustable.prime_epochs,
                      callbacks=[clr],
                      verbose=0)

        else:
            model.fit([prime_train[:, 0], prime_train[:, 1]], prime_labels,
                      batch_size=adjustable.batch_size,
                      epochs=adjustable.prime_epochs,
                      verbose=0)


        # testing
        part_ranking = this_ranking[an_id * ranking_number:(an_id + 1) * ranking_number]
        test_data = dp.grab_em_by_the_keys(part_ranking, None, h5_dataset)

        test_data = np.asarray(test_data)

        part_prediction = model.predict([test_data[0], test_data[1]])
        full_predictions[an_id * ranking_number:(an_id + 1) * ranking_number] = part_prediction

    return full_predictions


def only_test(model, h5_dataset, this_ranking):
    """ Only runs testing
    """
    test_data = dp.grab_em_by_the_keys(this_ranking, None, h5_dataset)
    test_data = np.asarray(test_data)
    part_prediction = model.predict([test_data[0], test_data[1]])
    full_predictions = part_prediction

    return full_predictions


def main(adjustable, all_ranking, names, model):
    confusion_matrices = []
    ranking_matrices = []
    gregor_matrices = []

    for item in range(len(all_ranking)):
        name = names[item]
        this_ranking = all_ranking[item]

        h5_dataset = dp.load_datasets_from_h5([name])
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

        gregor_matrix = pu.make_gregor_matrix(adjustable, full_predictions, final_testing_labels)
        gregor_matrices.append(gregor_matrix)

        # detection_rate = (gregor_matrix[0] * 1.0 / (gregor_matrix[0] * 1.0 + gregor_matrix[3] * 1.0))
        detection_rate, false_alarm = pu.calculate_TPR_FPR(matrix)
        # false_alarm = (gregor_matrix[1] * 1.0 / (gregor_matrix[1] * 1.0 + gregor_matrix[2] * 1.0))


        ranking = pu.calculate_CMC(adjustable, full_predictions)
        ranking_matrices.append(ranking)

        print(
        '%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \nCMC: \n%s \nDetection rate: %s  False alarm: %s'
        % (name, accuracy, precision, str(matrix), str(ranking), str(detection_rate), str(false_alarm)))

    return confusion_matrices, ranking_matrices, gregor_matrices


def get_model(adjustable):
    """
    Returns a model depending on the specifications.
    1. Loads a saved model + weights IF model name is specified
    2. Creates the model from scratch, loads saved weights and compiles IF model name is not specified AND
                                                                                model weights is specified
    3. Creates the model from scratch and compiles IF nothing is indicated

    :param adjustable:      object of class ProjectVariable
    :return:                returns the model
    """
    if adjustable.optimizer == 'nadam':
        the_optimizer = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
    elif adjustable.optimizer == 'sgd':
        the_optimizer = keras.optimizers.SGD()
    elif adjustable == 'rms':
        the_optimizer = keras.optimizers.RMSprop()
    else:
        the_optimizer = None

    # case 1
    if adjustable.load_model_name is not None:
        model = models.load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, '%s_model.h5' % adjustable.load_model_name))

    else:
        # case 3
        model = scn.create_siamese_network(adjustable)

        # case 2
        if adjustable.load_weights_name is not None:
            the_path = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, '%s_weights.h5' % adjustable.load_weights_name)
            model.load_weights(the_path, by_name=True)

        # compile
        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            model.compile(loss=adjustable.loss_function, optimizer=the_optimizer, metrics=['accuracy'])
        elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            model.compile(loss=scn.contrastive_loss, optimizer=the_optimizer, metrics=[scn.absolute_distance_difference])

    return model


def super_main(adjustable, get_data=False):
    name = adjustable.dataset_test
    # name = adjustable.datasets[0]
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

    all_ranking = [dataset_ranking]
    names = [name]

    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu

    '''
    '''
    model = get_model(adjustable)
    # if adjustable.load_model_name is not None:
    #     model = models.load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_model_name))
    # elif adjustable.load_weights_name is not None and adjustable.load_model_name is None:
    #     model = scn.create_siamese_network(adjustable)
    # 
    #     the_path = os.path.join('../model_weights', adjustable.load_weights_name)
    #     model.load_weights(the_path, by_name=True)
    # 
    #     if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
    #         nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
    #         model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])
    #     elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
    #         rms = keras.optimizers.RMSprop()
    #         model.compile(loss=scn.contrastive_loss, optimizer=rms, metrics=[scn.absolute_distance_difference])
    # else:
    #     model = None
    '''
    '''

    # number_of_datasets = len(names)
    number_of_datasets = 1

    # if adjustable.ranking_number == 'half':
    #     ranking_number = pc.RANKING_DICT[name]
    # elif isinstance(adjustable.ranking_number, int):
    #     ranking_number = adjustable.ranking_number
    if adjustable.ranking_number_test == 'half':
        ranking_number = pc.RANKING_DICT[name]
    elif isinstance(adjustable.ranking_number_test, int):
        ranking_number = adjustable.ranking_number_test
    else:
        print("ranking_number_test must be 'half' or an int")
        return

    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, ranking_number))
    gregor_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))

    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)

        confusion, ranking, gregor = main(adjustable, all_ranking, names, model)

        confusion_matrices[iter] = confusion
        ranking_matrices[iter] = ranking
        gregor_matrices[iter] = gregor

    stop = time.time()
    total_time = stop - start

    matrix_means = np.zeros((number_of_datasets, 4))
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_means = np.zeros((number_of_datasets, ranking_number))
    ranking_std = np.zeros((number_of_datasets, ranking_number))
    gregor_matrix_means = np.zeros((number_of_datasets, 4))
    gregor_matrix_std = np.zeros((number_of_datasets, 4))

    for dataset in range(number_of_datasets):
        matrices = np.zeros((adjustable.iterations, 4))
        rankings = np.zeros((adjustable.iterations, ranking_number))
        g_matrices = np.zeros((adjustable.iterations, 4))

        for iter in range(adjustable.iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]
            g_matrices[iter] = gregor_matrices[iter][dataset]

        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)
        gregor_matrix_means[dataset] = np.mean(g_matrices, axis=0)
        gregor_matrix_std[dataset] = np.std(g_matrices, axis=0)

    matrix_means = matrix_means[0]
    matrix_std = matrix_std[0]
    ranking_means = ranking_means[0]
    ranking_std = ranking_std[0]
    gregor_matrix_means = gregor_matrix_means[0]
    gregor_matrix_std = gregor_matrix_std[0]

    if adjustable.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable, adjustable.experiment_name, file_name, names, matrix_means, matrix_std, ranking_means,
                        ranking_std,
                        total_time, gregor_matrix_means, gregor_matrix_std)

    if get_data == True:
        return ranking_means, matrix_means, total_time