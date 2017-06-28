# import keras
# from keras import models
# from keras import layers
# from keras import optimizers
# import keras.backend as K
# from keras import initializers

from tensorflow.contrib import keras
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import initializers
import tensorflow as tf

import dynamic_data_loading as ddl
import project_constants as pc
import project_utils as pu
import os
import numpy as np
import time
import h5py
from clr_callback import *
import random


def euclidean_distance(vects):
    """ Returns the euclidean distance between the 2 feature vectors
    """
    x, y = vects
    # diff = x - y
    # sq = K.square(x - y)
    # su = K.sum(K.square(x - y), axis=1, keepdims=True)
    # sqrt = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    # return K.sqrt(tf.to_float(K.sum(K.square(x - y))))



def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    y = 0 if image is similar
    y = 1 if image is different

    according to Tokukawa: https://github.com/fchollet/keras/issues/4980 it has to be:
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    instead of:
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    """
    margin = 1
    loss = K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    return loss


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def cosine_distance(vects):
    """ Returns the cosine distance between the 2 feature vectors
    """
    x, y = vects

    return K.sum(x * y, axis=1, keepdims=True) / K.sqrt(
        K.sum(K.square(x), axis=1, keepdims=True) * K.sum(K.square(y), axis=1, keepdims=True))
    # return K.sum(x * y) / K.sqrt(
    #         K.sum(K.square(x)) * K.sum(K.square(y)))


def cos_dist_output_shape(shapes):
    """ IDK what this does
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_cost_module(inputs, adjustable):
    """Implements the cost module of the siamese network.
    :param inputs:          list containing feature tensor from each siamese head
    :return:                some type of distance
    """
    # NOTE: 2017/08 `merge` will become deprecated
    def subtract(x):
        output = x[0] - x[1]
        return output

    def divide(x):
        output = x[0] / x[1]
        return output

    def absolute(x):
        output = abs(x[0] - x[1])
        return output

    def the_shape(shapes):
        shape1, shape2 = shapes
        a_shape = shape1
        return a_shape

    if adjustable.cost_module_type == 'neural_network':
        if adjustable.neural_distance == 'concatenate':
            features = layers.concatenate(inputs)
        elif adjustable.neural_distance == 'add':
            features = layers.add(inputs)
        elif adjustable.neural_distance == 'multiply':
            features = layers.multiply(inputs)
        elif adjustable.neural_distance == 'subtract':
            # features = layers.merge(inputs=inputs, mode=subtract, output_shape=the_shape)
            features = layers.Lambda(subtract)(inputs)
        elif adjustable.neural_distance == 'divide':
            # features = layers.merge(inputs=inputs, mode=divide, output_shape=the_shape)
            features = layers.Lambda(divide)(inputs)
        elif adjustable.neural_distance == 'absolute':
            # features = layers.merge(inputs=inputs, mode=absolute, output_shape=the_shape)
            features = layers.Lambda(absolute)(inputs)
        else:
            features = None
        dense_layer = layers.Dense(adjustable.neural_distance_layers[0], name='dense_1', trainable=adjustable.trainable_cost_module)(features)
        activation = layers.Activation(adjustable.activation_function)(dense_layer)
        dropout_layer = layers.Dropout(pc.DROPOUT)(activation)
        dense_layer = layers.Dense(adjustable.neural_distance_layers[1], name='dense_2', trainable=adjustable.trainable_cost_module)(dropout_layer)
        activation = layers.Activation(adjustable.activation_function)(dense_layer)
        dropout_layer = layers.Dropout(pc.DROPOUT)(activation)
        output_layer = layers.Dense(pc.NUM_CLASSES, name='ouput')(dropout_layer)
        softmax = layers.Activation('softmax')(output_layer)

        if not adjustable.weights_name == None:
            softmax.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

        return softmax

    elif adjustable.cost_module_type == 'euclidean':
        # distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        distance = layers.Lambda(euclidean_distance)(inputs)
        return distance

    elif adjustable.cost_module_type == 'euclidean_fc':
        distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        dense_layer = layers.Dense(1, name='dense_1')(distance)
        activation = layers.Activation(adjustable.activation_function)(dense_layer)
        output_layer = layers.Dense(pc.NUM_CLASSES, name='ouput')(activation)
        softmax = layers.Activation('softmax')(output_layer)
        return softmax

    elif adjustable.cost_module_type == 'cosine':
        distance = layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)(inputs)
        return distance

    # elif adjustable.cost_module_type == 'DHSL':
    #     ''' As proposed by Zhu et al. in https://arxiv.org/abs/1702.04858
    #     '''
    # elif adjustable.cost_module_type == 'kissme':




def add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name, first_layer=False):
    """One-liner for adding: pooling + activation + batchnorm
    :param model:       the model to add to
    :return:            the model with added activation and max pooling
    :param trainable:   boolean indicating if layer is trainable
    """

    if first_layer:
        if adjustable.pooling_type == 'avg_pooling':
            model.add(layers.AveragePooling2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
        else:  # max_pooling
            model.add(layers.MaxPool2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
    else:
        if adjustable.pooling_type == 'avg_pooling':
            model.add(layers.AveragePooling2D(pool_size=(adjustable.pooling_size[1][0], adjustable.pooling_size[1][1])))
        else:  # max_pooling
            model.add(layers.MaxPool2D(pool_size=(adjustable.pooling_size[1][0], adjustable.pooling_size[1][1])))

    model.add(layers.Activation(adjustable.activation_function))

    if use_batch_norm:
        model.add(layers.BatchNormalization(name=batch_norm_name, trainable=adjustable.trainable_bn))
    return model


def create_siamese_head(adjustable):
    """Implements 1 head of the siamese network.
    :return:                    a keras models.Sequential model
    """
    use_batch_norm = True if adjustable.head_type == 'batch_normalized' else False

    # convolutional unit 1
    model = models.Sequential()
    if use_batch_norm == True:
        model.add(layers.BatchNormalization(name='bn_1', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                                     trainable=adjustable.trainable_bn))
    model.add(layers.Conv2D(16 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_1',
                     trainable=adjustable.trainable_12))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_2', first_layer=True)
    # convolutional unit 2
    model.add(layers.Conv2D(32 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_2',
                     trainable=adjustable.trainable_12))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_3')
    # convolutional unit 3
    model.add(layers.Conv2D(64 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_3',
                     trainable=adjustable.trainable_34))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_4')
    # convolutional unit 4
    model.add(layers.Conv2D(128 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_4',
                     trainable=adjustable.trainable_34))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_5')
    # convolutional unit 5
    model.add(layers.Conv2D(256 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_5',
                     trainable=adjustable.trainable_56))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_6')
    # convolutional unit 6
    model.add(layers.Conv2D(512 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_6',
                     trainable=adjustable.trainable_56))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_7')
    if adjustable.pooling_size == [[2, 2], [2, 2]]:
        model.add(layers.Conv2D(1024 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_7',
                         trainable=adjustable.trainable_56))
        model.add(layers.Activation(adjustable.activation_function))
        if use_batch_norm == True:
            model.add(layers.BatchNormalization(name='bn_8', trainable=adjustable.trainable_bn))

    model.add(layers.Flatten(name='cnn_flat'))


    if not adjustable.weights_name == None:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

    return model


def create_siamese_network(adjustable):
    """Creates the siamese network.
    :return:    Keras models.Sequential model
    """
    input_a = layers.Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    input_b = layers.Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

    siamese_head = create_siamese_head(adjustable)

    processed_a = siamese_head(input_a)
    processed_b = siamese_head(input_b)

    distance = create_cost_module([processed_a, processed_b], adjustable)
    model = models.Model([input_a, input_b], distance)

    return model


# unused
def train_network(adjustable, model, step, validation_interval, train_data, train_labels, validation_data,
                  validation_labels):
    """Trains the siamese network.
    :param model:                   the model that needs to be trained
    :param step:                    the training step in an epoch
    :param validation_interval:     indicates after how many steps validation takes place
    :param train_data:              array containing the training data
    :param train_labels:            array containing the training labels
    :param validation_data:         array containing the validation data
    :param validation_labels:       array containing the validation labels
    """
    if adjustable.use_cyclical_learning_rate:
        clr = CyclicLR(step_size=(np.shape(train_data)[0] / adjustable.batch_size) * 8, base_lr=adjustable.cl_min, max_lr=adjustable.cl_max)
        if step % validation_interval == 0:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=adjustable.batch_size,
                      epochs=1,
                      validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                      verbose=2,
                      callbacks=[clr])
        else:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=adjustable.batch_size,
                      epochs=1,
                      verbose=0,
                      callbacks=[clr])
    else:
        if step % validation_interval == 0:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=adjustable.batch_size,
                      epochs=1,
                      validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                      verbose=2)
        else:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=adjustable.batch_size,
                      epochs=1,
                      verbose=0)



def train_network_light(adjustable, model, final_training_data, final_training_labels, h5_data_list):
    if adjustable.use_cyclical_learning_rate:

        clr = CyclicLR(step_size=(len(final_training_labels) / adjustable.batch_size) * 8, base_lr=adjustable.cl_min,
                       max_lr=adjustable.cl_max)

        train_data = ddl.grab_em_by_the_keys(final_training_data, h5_data_list)

        train_data = np.asarray(train_data)

        model.fit([train_data[0, :], train_data[1, :]], final_training_labels,
                  batch_size=adjustable.batch_size,
                  epochs=1,
                  validation_split=0.01,
                  verbose=2,
                  callbacks=[clr])
    else:
        train_data = ddl.grab_em_by_the_keys(final_training_data, h5_data_list)
        train_data = np.asarray(train_data)

        model.fit([train_data[0, :], train_data[1, :]], final_training_labels,
                  batch_size=adjustable.batch_size,
                  epochs=1,
                  validation_split=0.01,
                  verbose=2)


def absolute_distance_difference(y_true, y_pred):
    return abs(y_true - y_pred)

def main(adjustable, h5_data_list, all_ranking, merged_training_pos, merged_training_neg):
    """Runs a the whole training and testing phase
    :return:    array of dataset names, array containing the confusion matrix for each dataset, array containing the
                ranking for each dataset
    """

    if not adjustable.load_model_name == None:
        model = models.load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_model_name))
    elif not adjustable.load_weights_name == None:
        model = create_siamese_network(adjustable)

        the_path = os.path.join('../model_weights', adjustable.load_weights_name)
        model.load_weights(the_path, by_name=True)

        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
            model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])
        elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            rms = keras.optimizers.RMSprop()
            model.compile(loss=contrastive_loss, optimizer=rms, metrics=[absolute_distance_difference])

    else:
        model = create_siamese_network(adjustable)

        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
            model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])
        elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            rms = keras.optimizers.RMSprop()
            model.compile(loss=contrastive_loss, optimizer=rms, metrics=[absolute_distance_difference])

    for epoch in range(adjustable.epochs):
        print('epoch %d/%d' % (epoch, adjustable.epochs))
        # sample from the big set of negative training instances
        random.shuffle(merged_training_neg)
        training_neg_sample = merged_training_neg[0:len(merged_training_pos)]
        # now we have the final list of keys to the instances we use for training

        final_training_data = merged_training_pos + training_neg_sample

        random.shuffle(final_training_data)

        final_training_data = pu.sideways_shuffle(final_training_data)

        final_training_labels = [int(final_training_data[item].strip().split(',')[-1]) for item in
                                 range(len(final_training_data))]
        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            final_training_labels = keras.utils.to_categorical(final_training_labels, pc.NUM_CLASSES)

        train_network_light(adjustable, model, final_training_data, final_training_labels, h5_data_list)

        time_stamp = time.strftime('scnn_%d%m%Y_%H%M')
        # print('DONE WITH TRAINING')

        if adjustable.save_inbetween and adjustable.iterations == 1:
            if epoch+1 in adjustable.save_points:
                if adjustable.name_indication == 'epoch':
                    model_name = time_stamp + '_epoch_%s_model.h5' % str(epoch + 1)
                    weights_name = time_stamp + '_epoch_%s_weigths.h5' % str(epoch + 1)
                elif adjustable.name_indication == 'dataset_name' and len(adjustable.datasets) == 1:
                    # model_name = time_stamp + '_%s_model.h5' % adjustable.datasets[0]
                    # weights_name = time_stamp + '_%s_weights.h5' % adjustable.datasets[0]
                    model_name = '%s_model_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                    weights_name = '%s_weigths_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                else:
                    model_name = None
                    weights_name = None

                model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, model_name))
                model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name))
                print('MODEL SAVED')

    # if appropriate, skip the testing step
    confusion_matrices = []
    ranking_matrices = []
    names = []

    # for test_set in range(test_sets):
    for dataset in range(len(adjustable.datasets)):
        # name = test[test_set * 3]
        name = adjustable.datasets[dataset]
        names.append(name)
        this_ranking = all_ranking[dataset]
        # print('GRABBING BY THE KEYS')
        test_data = ddl.grab_em_by_the_keys(this_ranking, h5_data_list)
        test_data = np.asarray(test_data)

        # make a record of the ranking selection for each dataset
        # for priming
        if adjustable.save_inbetween and adjustable.iterations == 1:
            # file_name = '%s_ranking_%s.txt' % (name, adjustable.ranking_time_name)
            file_name = '%s_ranking_%s.txt' % (name, adjustable.use_gpu)
            file_name = os.path.join(pc.SAVE_LOCATION_RANKING_FILES, file_name)
            with open(file_name, 'w') as my_file:
                for item in this_ranking:
                    my_file.write(item)

        # print('final_testing_labels')
        final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in range(len(this_ranking))]


        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            print('more final testing labels')
            final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

        # print('predictions')
        predictions = model.predict([test_data[0, :], test_data[1, :]])
        # print predictions
        if adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            new_thing = zip(predictions, final_testing_labels)
            print(new_thing[0:50])
        # print('MAKING CONFUSION MATRIX')
        # matrix = pu.make_confusion_matrix(predictions, test_labels)
        matrix = pu.make_confusion_matrix(adjustable, predictions, final_testing_labels)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0
        confusion_matrices.append(matrix)

        ranking = pu.calculate_CMC(adjustable, predictions)
        ranking_matrices.append(ranking)

        print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
              % (name, accuracy, precision, str(matrix), str(ranking)))


    del model
    return names, confusion_matrices, ranking_matrices


def super_main(adjustable):
    """Runs main for a specified iterations. Useful for experiment running.
    Note: set iterations to 1 if you want to save weights
    """
    # load the datasets from h5
    # note: this will always be 1 dataset
    all_h5_datasets = ddl.load_datasets_from_h5(adjustable.datasets)

    if adjustable.ranking_number == 'half':
        the_dataset_name = adjustable.datasets[0]
        ranking_number = pc.RANKING_DICT[the_dataset_name]
    elif isinstance(adjustable.ranking_number, int):
        ranking_number = adjustable.ranking_number
    else:
        ranking_number = None

    the_dataset_name = adjustable.datasets[0]
    # select which GPU to use, necessary to start tf session
    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu
    # arrays for storing results
    number_of_datasets = len(adjustable.datasets)
    name = np.zeros(number_of_datasets)
    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, ranking_number))

    start = time.time()
    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)
        # lists for storing intermediate results
        all_ranking, all_training_pos, all_training_neg = [], [], []
        # create training and ranking set for all datasets
        ss = time.time()
        for name in range(len(adjustable.datasets)):
            ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(adjustable.datasets[name], adjustable)
            # labels have different meanings in `euclidean` case, 0 for match and 1 for mismatch
            if adjustable.cost_module_type == 'euclidean':
                ranking = pu.flip_labels(ranking)
                training_pos = pu.flip_labels(training_pos)
                training_neg = pu.flip_labels(training_neg)
            elif adjustable.cost_module_type == 'cosine':
                ranking = pu.zero_to_min_one_labels(ranking)
                training_pos = pu.zero_to_min_one_labels(training_pos)
                training_neg = pu.zero_to_min_one_labels(training_neg)

            # data gets appended in order
            all_ranking.append(ranking)
            all_training_pos.append(training_pos)
            all_training_neg.append(training_neg)
        # put all the training data together
        st = time.time()
        print('%0.2f mins' % ((st-ss)/60))
        merged_training_pos, merged_training_neg = ddl.merge_datasets(adjustable, all_training_pos, all_training_neg)
        # run main
        name, confusion_matrix, ranking_matrix = main(adjustable, all_h5_datasets, all_ranking, merged_training_pos,
                                                      merged_training_neg)
        # store results
        confusion_matrices[iter] = confusion_matrix
        ranking_matrices[iter] = ranking_matrix

    stop = time.time()
    total_time = stop - start

    matrix_means = np.zeros((number_of_datasets, 4))
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_means = np.zeros((number_of_datasets, ranking_number))
    ranking_std = np.zeros((number_of_datasets, ranking_number))
    # for each dataset, create confusion and ranking matrices
    for dataset in range(number_of_datasets):
        matrices = np.zeros((adjustable.iterations, 4))
        rankings = np.zeros((adjustable.iterations, ranking_number))

        for iter in range(adjustable.iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]
        # calculate the mean and std
        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)
    # log the results
    if adjustable.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable, adjustable.experiment_name, file_name, name, matrix_means, matrix_std, ranking_means, ranking_std,
                        total_time)
