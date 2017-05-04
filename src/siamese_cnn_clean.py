import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda, BatchNormalization
from keras import optimizers
import dynamic_data_loading as ddl
from keras import backend as K
import project_constants as pc
import project_utils as pu
import os
import numpy as np
import time
import h5py
from clr_callback import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_cost_module(inputs, adjustable):
    """Implements the cost module of the siamese network.
    :param inputs:          list containing feature tensor from each siamese head
    :return:                some type of distance
    """
    if adjustable.cost_module_type == 'neural_network':
        if adjustable.neural_distance == 'concatenate':
            features = keras.layers.concatenate(inputs)
        elif adjustable.neural_distance == 'add':
            features = keras.layers.add(inputs)
        elif adjustable.neural_distance == 'multiply':
            features = keras.layers.multiply(inputs)
        else:
            features = None

        dense_layer = Dense(512)(features)
        activation = Activation('relu')(dense_layer)
        dropout_layer = Dropout(pc.DROPOUT)(activation)
        dense_layer = Dense(1024)(dropout_layer)
        activation = Activation('relu')(dense_layer)
        dropout_layer = Dropout(pc.DROPOUT)(activation)
        output_layer = Dense(pc.NUM_CLASSES)(dropout_layer)
        softmax = Activation('softmax')(output_layer)
        return softmax

    elif adjustable.cost_module_type == 'euclidean':
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        return distance

    else:
        return None


def add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name):
    """One-liner for adding activation and max pooling
    :param model:       the model to add to
    :return:            the model with added activation and max pooling
    :param trainable:   boolean indicating if layer is trainable
    """
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if use_batch_norm:
        model.add(BatchNormalization(name=batch_norm_name, trainable=adjustable.trainable))
    return model


def create_siamese_head(adjustable):
    """Implements 1 head of the siamese network.
    :return:                    a keras Sequential model
    """
    use_batch_norm = True if adjustable.head_type == 'batch_normalized' else False

    model = Sequential()
    model.add(Conv2D(16 * adjustable.numfil, kernel_size=(3, 3), padding='same',
                     input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                     name='conv_1',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_1')
    model.add(Conv2D(32 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_2', trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_2')
    model.add(Conv2D(64 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_3', trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_3')
    model.add(Conv2D(128 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_4', trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_4')
    model.add(Conv2D(256 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_5', trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_5')
    model.add(Conv2D(512 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_6', trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_6')
    model.add(Conv2D(1024 * adjustable.numfil, kernel_size=(3, 3), padding='same', name='conv_7', trainable=adjustable.trainable))
    model.add(Activation('relu'))
    if use_batch_norm == True:
        model.add(BatchNormalization(name='bn_7', trainable=adjustable.trainable))
    model.add(Flatten(name='cnn_flat'))

    if adjustable.transfer_weights == True:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.cnn_weights_name), by_name=True)

    return model


def create_siamese_network(adjustable):
    """Creates the siamese network.
    :return:    Keras Sequential model
    """
    input_a = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    input_b = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

    siamese_head = create_siamese_head(adjustable)

    processed_a = siamese_head(input_a)
    processed_b = siamese_head(input_b)

    distance = create_cost_module([processed_a, processed_b], adjustable)
    model = Model([input_a, input_b], distance)
    return model


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


def main(adjustable):
    """Runs a the whole training and testing phase
    :return:    array of dataset names, array containing the confusion matrix for each dataset, array containing the
                ranking for each dataset
    """
    total_data_list, validation, test = ddl.get_data_scnn(adjustable)
    total_data_list_pos, total_data_list_neg = total_data_list
    validation_data, validation_labels = validation

    model = create_siamese_network(adjustable)

    slice_size = 5000
    train_data_size = 2 * min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps_per_epoch = np.ceil(train_data_size * 1.0 / slice_size).astype(int)

    num_validations_per_epoch = 1  # note: must be at least 1
    if num_validations_per_epoch > num_steps_per_epoch: num_validations_per_epoch = num_steps_per_epoch
    validation_interval = np.floor(num_steps_per_epoch / num_validations_per_epoch).astype(int)

    if adjustable.cost_module_type == 'neural_network':
        nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
    elif adjustable.cost_module_type == 'euclidean':
        rms = keras.optimizers.RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)

    for epoch in range(adjustable.epochs):
        print('------EPOCH: %d' % epoch)
        slice_size_queue = ddl.make_slice_queue(train_data_size, slice_size)
        total_train_data_list = ddl.make_train_batches(total_data_list_pos, total_data_list_neg, data_type='images')
        for step in range(num_steps_per_epoch):
            train_data_list = total_train_data_list[step * slice_size: step * slice_size + slice_size_queue[step]]

            train_data, train_labels = ddl.load_in_array(adjustable, data_list=train_data_list,
                                                         heads=2,
                                                         data_type='images')

            train_network(adjustable, model=model, step=step, validation_interval=validation_interval, train_data=train_data,
                          train_labels=train_labels, validation_data=validation_data,
                          validation_labels=validation_labels)

    test_sets = len(test) / 3
    confusion_matrices = []
    ranking_matrices = []
    names = []

    for test_set in range(test_sets):
        name = test[test_set * 3]
        names.append(name)

        test_data = test[(test_set * 3) + 1]
        test_labels = test[(test_set * 3) + 2]

        predictions = model.predict([test_data[:, 0], test_data[:, 1]])

        # FIXME make make_confusion_matrix euclidean distance compatible
        matrix = pu.make_confusion_matrix(predictions, test_labels)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0
        confusion_matrices.append(matrix)

        # FIXME make calculate_CMC euclidean distance compatible
        ranking = pu.calculate_CMC(predictions)
        ranking_matrices.append(ranking)

        print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
              % (name, accuracy, precision, str(matrix), str(ranking)))

    if not adjustable.scnn_save_weights_name == None:
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.scnn_save_weights_name))

    del model
    return names, confusion_matrices, ranking_matrices


def super_main(adjustable):
    """Runs main for a specified iterations. Useful for experiment running.
    Note: set iterations to 1 if you want to save weights
    """
    number_of_datasets = 2
    name = np.zeros(number_of_datasets)
    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, pc.RANKING_NUMBER))

    start = time.time()
    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)

        name, confusion_matrix, ranking_matrix = main(adjustable)

        confusion_matrices[iter] = confusion_matrix
        ranking_matrices[iter] = ranking_matrix

    stop = time.time()
    total_time = stop - start

    matrix_means = np.zeros((number_of_datasets, 4))
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_means = np.zeros((number_of_datasets, pc.RANKING_NUMBER))
    ranking_std = np.zeros((number_of_datasets, pc.RANKING_NUMBER))

    for dataset in range(number_of_datasets):
        matrices = np.zeros((adjustable.iterations, 4))
        rankings = np.zeros((adjustable.iterations, pc.RANKING_NUMBER))

        for iter in range(adjustable.iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]

        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)

    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable.experiment_name, file_name, name, matrix_means, matrix_std, ranking_means, ranking_std,
                        total_time)
