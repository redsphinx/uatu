import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda, BatchNormalization, AveragePooling2D
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
import random

def euclidean_distance(vects):
    """ Returns the euclidean distance between the 2 feature vectors
    """
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# TODO find out what this does
def eucl_dist_output_shape(shapes):
    """ IDK what this does
    """
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
            # TODO: debug
        elif adjustable.neural_distance == 'subtract':
            features = inputs[0] - inputs[1]
        elif adjustable.neural_distance == 'divide':
            features = inputs[0] / inputs[1]
        elif adjustable.neural_distance == 'absolute':
            features = abs(inputs[0] - inputs[1])
        else:
            features = None

        dense_layer = Dense(adjustable.neural_distance_layers[0], name='dense_1')(features)
        activation = Activation(adjustable.activation_function)(dense_layer)
        dropout_layer = Dropout(pc.DROPOUT)(activation)
        dense_layer = Dense(adjustable.neural_distance_layers[1], name='dense_2')(dropout_layer)
        activation = Activation(adjustable.activation_function)(dense_layer)
        dropout_layer = Dropout(pc.DROPOUT)(activation)
        output_layer = Dense(pc.NUM_CLASSES, name='ouput')(dropout_layer)
        softmax = Activation('softmax')(output_layer)

        if not adjustable.weights_name == None:
            softmax.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

        return softmax

    elif adjustable.cost_module_type == 'euclidean':
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        return distance

    else:
        return None


def add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name, first_layer=False):
    """One-liner for adding activation and max pooling
    :param model:       the model to add to
    :return:            the model with added activation and max pooling
    :param trainable:   boolean indicating if layer is trainable
    """
    model.add(Activation(adjustable.activation_function))
    if first_layer:
        if adjustable.pooling_type == 'avg_pooling':
            model.add(AveragePooling2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
        else:  # max_pooling
            model.add(MaxPool2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
    else:
        model.add(MaxPool2D(pool_size=(adjustable.pooling_size[1][0], adjustable.pooling_size[1][1])))
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
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_1', first_layer=True)
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
    model.add(Activation(adjustable.activation_function))
    if use_batch_norm == True:
        model.add(BatchNormalization(name='bn_7', trainable=adjustable.trainable))
    model.add(Flatten(name='cnn_flat'))

    if not adjustable.weights_name == None:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

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


def main(adjustable, h5_data_list, all_ranking, merged_training_pos, merged_training_neg):
    """Runs a the whole training and testing phase
    :return:    array of dataset names, array containing the confusion matrix for each dataset, array containing the
                ranking for each dataset
    """
    model = create_siamese_network(adjustable)

    if adjustable.cost_module_type == 'neural_network':
        nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
        model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])
    elif adjustable.cost_module_type == 'euclidean':
        rms = keras.optimizers.RMSprop()
        # TODO: debug
        model.compile(loss=contrastive_loss, optimizer=rms)

    for epoch in range(adjustable.epochs):
        print('epoch %d/%d' % (epoch, adjustable.epochs))
        # sample from the big set of negative training instances
        random.shuffle(merged_training_neg)
        training_neg_sample = merged_training_neg[0:len(merged_training_pos)]
        # now we have the final list of keys to the instances we use for training

        final_training_data = merged_training_pos + training_neg_sample

        random.shuffle(final_training_data)

        # TODO: debug
        final_training_data = pu.sideways_shuffle(final_training_data)

        # TODO: debug
        final_training_labels = [int(final_training_data[item].strip().split(',')[-1]) for item in
                                 range(len(final_training_data))]
        if adjustable.cost_module_type == 'neural_network':
            final_training_labels = keras.utils.to_categorical(final_training_labels, pc.NUM_CLASSES)

        train_network_light(adjustable, model, final_training_data, final_training_labels, h5_data_list)

    confusion_matrices = []
    ranking_matrices = []
    names = []

    # for test_set in range(test_sets):
    for dataset in range(len(adjustable.datasets)):
        # name = test[test_set * 3]
        name = adjustable.datasets[dataset]
        names.append(name)
        this_ranking = all_ranking[dataset]
        test_data = ddl.grab_em_by_the_keys(this_ranking, h5_data_list)
        test_data = np.asarray(test_data)

        # make a record of the ranking selection for each dataset
        # for priming
        if not adjustable.scnn_save_weights_name == None:
            file_name = '%s_ranking.txt' % name
            with open(file_name, 'w') as my_file:
                for item in this_ranking:
                    my_file.write(item)

        final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in range(len(this_ranking))]
        final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

        predictions = model.predict([test_data[0, :], test_data[1, :]])

        # FIXME make make_confusion_matrix euclidean distance compatible
        # matrix = pu.make_confusion_matrix(predictions, test_labels)
        matrix = pu.make_confusion_matrix(predictions, final_testing_labels)
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

    if not adjustable.scnn_save_model_name == None:
        model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.scnn_save_model_name))

    del model
    return names, confusion_matrices, ranking_matrices


def super_main(adjustable):
    """Runs main for a specified iterations. Useful for experiment running.
    Note: set iterations to 1 if you want to save weights
    """
    all_h5_datasets = ddl.load_datasets_from_h5(adjustable.datasets)

    # all_ranking, all_training_pos, all_training_neg = \
    #     [ddl.create_training_and_ranking_set(name) for name in adjustable.datasets]

    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu

    number_of_datasets = len(adjustable.datasets)
    name = np.zeros(number_of_datasets)
    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, pc.RANKING_NUMBER))

    start = time.time()
    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)

        all_ranking, all_training_pos, all_training_neg = [], [], []
        for name in range(len(adjustable.datasets)):
            ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(adjustable.datasets[name])

            # TODO: debug
            if adjustable.cost_module_type == 'euclidean':
                ranking = pu.flip_labels(ranking)
                training_pos = pu.flip_labels(training_pos)
                training_neg = pu.flip_labels(training_neg)

            # all_ranking = all_ranking + ranking
            # all_training_pos = all_training_pos + training_pos
            # all_training_neg = all_training_neg + training_neg

            all_ranking.append(ranking)
            all_training_pos.append(training_pos)
            all_training_neg.append(training_neg)

        merged_training_pos, merged_training_neg = ddl.merge_datasets(adjustable, all_training_pos, all_training_neg)

        name, confusion_matrix, ranking_matrix = main(adjustable, all_h5_datasets, all_ranking, merged_training_pos,
                                                      merged_training_neg)

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
