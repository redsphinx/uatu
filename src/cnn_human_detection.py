"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

Defines a network to train on human datasets.

"""

import keras
from keras import models
from keras import layers
from keras import optimizers
import project_constants as pc
import project_utils as pu
import data_pipeline as dp
from clr_callback import *
import time
import os
import numpy as np
import random


def add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name, first_layer=False):
    """One-liner for adding: pooling + activation + batchnorm
    :param adjustable:
    :param model:           the model to add to
    :param use_batch_norm:
    :param batch_norm_name:
    :param first_layer:
    :return:                the model with added activation and max pooling
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


def create_cnn_model(adjustable):
    """Implements a convolutional neural network
    :param adjustable:          object of class ProjectVariable
    :return:                    a keras models.Sequential model
    """
    if adjustable.head_type == 'batch_normalized':
        use_batch_norm = True
    else:
        use_batch_norm = False

    # convolutional unit 1
    model = models.Sequential()
    if use_batch_norm:
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
        if use_batch_norm:
            model.add(layers.BatchNormalization(name='bn_8', trainable=adjustable.trainable_bn))

    model.add(layers.Flatten(name='cnn_flat'))

    model.add(layers.Dense(adjustable.neural_distance_layers[0], name='dense_1'))
    model.add(layers.Activation(adjustable.activation_function))
    model.add(layers.Dropout(pc.DROPOUT, name='dropout_1'))
    model.add(layers.Dense(adjustable.neural_distance_layers[1], name='dense_2'))
    model.add(layers.Activation(adjustable.activation_function))
    model.add(layers.Dropout(pc.DROPOUT, name='dropout_2'))
    model.add(layers.Dense(pc.NUM_CLASSES, name='output'))
    model.add(layers.Activation('softmax'))

    if adjustable.weights_name is not None:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

    return model


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
        the_path = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, '%s_model.h5' % adjustable.load_model_name)
        model = models.load_model(the_path)

    else:
        # case 3
        model = create_cnn_model(adjustable)

        # case 2
        if adjustable.load_weights_name is not None:
            the_path = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, '%s_weights.h5' % adjustable.load_weights_name)
            model.load_weights(the_path, by_name=True)

        # compile
        model.compile(loss=adjustable.loss_function, optimizer=the_optimizer, metrics=['accuracy'])

    return model


def main(adjustable, test_data, train_data, h5_dataset):
    """
    Runs the main loop
    :param adjustable:      object of class ProjectVariable
    :param test_data:       list of keys for test data
    :param train_data:      list of keys for train data
    :param h5_dataset:      hdf5 dataset file
    :returns:               confusion matrix
    """

    model = get_model(adjustable)

    len_train = len(train_data)

    labels_train = [train_data[item].split(',')[-1] for item in range(len_train)]
    labels_train = keras.utils.to_categorical(labels_train, 2)

    keys_train = [train_data[item].split(',')[0] for item in range(len_train)]

    # if using cyclical learning rate
    if adjustable.use_cyclical_learning_rate:
        clr = CyclicLR(step_size=(len(labels_train) / adjustable.batch_size) * 8, base_lr=adjustable.cl_min,
                       max_lr=adjustable.cl_max)
    else:
        clr = None

    model.fit(dp.get_human_data(keys_train, h5_dataset),
              labels_train,
              epochs=adjustable.epochs,
              validation_split=0.1,
              batch_size=adjustable.batch_size,
              verbose=2,
              callbacks=[clr])

    # maybe save
    if adjustable.cnn_save and adjustable.iterations == 1:
        name = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, 'cnn_model_%d_epochs.h5' % adjustable.epochs)
        model.save(name)
        name = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, 'cnn_weights_%d_epochs.h5' % adjustable.epochs)
        model.save_weights(name)

    # testing
    len_test = len(test_data)

    labels_test = [test_data[item].split(',')[-1] for item in range(len_test)]
    labels_test = keras.utils.to_categorical(labels_test, 2)

    keys_test = [train_data[item].split(',')[0] for item in range(len_test)]

    predictions = model.predict(dp.get_human_data(keys_test, h5_dataset))

    matrix = pu.make_confusion_matrix(adjustable, predictions, labels_test)
    accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
    if not matrix[0] == 0:
        precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
    else:
        precision = 0
    print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s'
          % (adjustable.datasets[0], accuracy, precision, str(matrix)))
    return matrix


def super_main(adjustable):
    """
    Runs the main loop a number of times and saves the results of the experiments.
    :param adjustable:      object of class ProjectVariable
    """

    h5_datasets = dp.load_datasets_from_h5(adjustable.datasets)
    if adjustable.datasets[0] == 'inria':
        data_list = list(np.genfromtxt('../data/INRIA/swapped.txt', dtype=None))
    else:
        data_list = None

    # to store the confusion matrices
    matrices = np.zeros((adjustable.iterations, 4))

    start = time.time()

    for iteration in range(adjustable.iterations):
        # for each iteration of an experiment, present the data in a different order
        random.shuffle(data_list)

        # let's say that we use % of the data to validate on
        test_ratio = 0.01
        test = data_list[0:int(test_ratio*len(data_list))]
        train = data_list[int(test_ratio*len(data_list)):]

        matrices[iteration] = main(adjustable, test, train, h5_datasets)

    stop = time.time()

    mean_matrix = np.mean(matrices, axis=0)
    std_matrix = np.std(matrices, axis=0)

    total_time = stop - start

    if adjustable.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log_cnn(adjustable, adjustable.experiment_name, file_name, adjustable.datasets[0], mean_matrix,
                            std_matrix, total_time)
