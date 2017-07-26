import keras
from keras import models
from keras import layers
from keras import optimizers
import keras.backend as K
from keras import initializers
import tensorflow as tf

# from tensorflow.contrib import keras
# from tensorflow.contrib.keras import models
# from tensorflow.contrib.keras import layers
# from tensorflow.contrib.keras import optimizers
# from tensorflow.contrib.keras import backend as K
# from tensorflow.contrib.keras import initializers
# import tensorflow as tf

import dynamic_data_loading as ddl
import project_constants as pc
import project_utils as pu
import os
import numpy as np
import time
import h5py
from clr_callback import *
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def euclidean_distance(vects):
    """ Returns the euclidean distance between the 2 feature vectors
    """
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# unused
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


# unused
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# unused
def cosine_similarity(vects):
    """ Returns the cosine similarity between the 2 feature vectors
    """
    x, y = vects

    return K.sum(x * y, axis=1, keepdims=True) / K.sqrt(
        K.sum(K.square(x), axis=1, keepdims=True) * K.sum(K.square(y), axis=1, keepdims=True))


# note: fixed cosine
def cosine_distance_normalized(vects):
    """
    Returns the normalized cosine distance between 2 feature vectors. Normalizing it makes it a formal distance metric.
    :param vects:
    :return:
    """
    x, y = vects
    cos_ang = K.sum(x * y, axis=1, keepdims=True) / K.sqrt(
        K.sum(K.square(x), axis=1, keepdims=True) * K.sum(K.square(y), axis=1, keepdims=True))

    normalized_distance = tf.acos(cos_ang) / tf.constant(3.141592653589793)

    # normalized_distance = np.arccos(cos_ang) / np.pi

    return normalized_distance


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

    def subtract(x):
        output = x[0] - x[1]
        return output

    def divide(x):
        output = x[0] / x[1]
        return output

    def absolute(x):
        output = abs(x[0] - x[1])
        return output

    # unused
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
        dense_layer = layers.Dense(adjustable.neural_distance_layers[0], name='dense_1',
                                   trainable=adjustable.trainable_cost_module)(features)
        activation = layers.Activation(adjustable.activation_function)(dense_layer)
        if adjustable.activation_function == 'selu':
            dropout_layer = layers.AlphaDropout(adjustable.dropout_rate)(activation)
        else:
            dropout_layer = layers.Dropout(adjustable.dropout_rate)(activation)
        dense_layer = layers.Dense(adjustable.neural_distance_layers[1], name='dense_2',
                                   trainable=adjustable.trainable_cost_module)(dropout_layer)
        activation = layers.Activation(adjustable.activation_function)(dense_layer)
        if adjustable.activation_function == 'selu':
            dropout_layer = layers.AlphaDropout(adjustable.dropout_rate)(activation)
        else:
            dropout_layer = layers.Dropout(adjustable.dropout_rate)(activation)
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
        # distance = layers.Lambda(cosine_distance)(inputs)
        distance = layers.Lambda(cosine_distance_normalized)(inputs)
        # distance = layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)(inputs)
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
        # model.add(layers.BatchNormalization(name=batch_norm_name, trainable=adjustable.trainable_bn))
    return model


def create_siamese_head(adjustable):
    """Implements 1 head of the siamese network.
    :return:        a keras models.Sequential model
    """
    use_batch_norm = True if adjustable.head_type == 'batch_normalized' else False

    # convolutional unit 1
    model = models.Sequential()
    if use_batch_norm == True:
        model.add(layers.BatchNormalization(name='bn_1', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                                            trainable=adjustable.trainable_bn))
    model.add(layers.Conv2D(16 * adjustable.numfil, input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                            kernel_size=adjustable.kernel, padding='same', name='conv_1',
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


def train_network(adjustable, model, final_training_data, final_training_labels, h5_train, h5_test):
    # def train_network(adjustable, model, final_training_data, final_training_labels, h5_data_list):
    """
    Trains the network.
    :param adjustable:                  object of class ProjectVariable
    :param model:                       the model
    :param final_training_data:         list of pairs of images
    :param final_training_labels:       list of labels belonging to final_training_data
    :param h5_data_list:                the HDF5 data
    :return:                            trained model
    """

    # DONE TODO: do something like this to merge the h5 data lists
    # h5_data_list = h5_train + h5_test

    if adjustable.use_cyclical_learning_rate:

        clr = CyclicLR(step_size=(len(final_training_labels) / adjustable.batch_size) * 8, base_lr=adjustable.cl_min,
                       max_lr=adjustable.cl_max)
        call_back = [clr]
    else:
        call_back = None

    # note for mixing data: I think this should still work
    train_data = ddl.grab_em_by_the_keys(final_training_data, h5_train, h5_test)
    train_data = np.asarray(train_data)

    model.fit([train_data[0, :], train_data[1, :]], final_training_labels,
              batch_size=adjustable.batch_size,
              epochs=1,
              validation_split=0.01,
              verbose=2,
              callbacks=call_back)


def absolute_distance_difference(y_true, y_pred):
    """
    Returns the absolute distance between two numbers
    :param y_true:      The true number
    :param y_pred:      The predicted number
    :return:            the absolute distance between two numbers
    """
    return abs(y_true - y_pred)


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
        model = models.load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_model_name))

    else:
        # case 3
        model = create_siamese_network(adjustable)

        # case 2
        if adjustable.load_weights_name is not None:
            the_path = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_weights_name)
            model.load_weights(the_path, by_name=True)

        # compile
        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            model.compile(loss=adjustable.loss_function, optimizer=the_optimizer, metrics=['accuracy'])
        elif adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            model.compile(loss=contrastive_loss, optimizer=the_optimizer, metrics=[absolute_distance_difference])

    return model


def get_negative_sample(adjustable, train_pos, train_neg):
    number_of_datasets = len(train_pos)

    negative = []

    if adjustable.only_test == True:
        # only test, nothing to do
        print('Only testing, nothing to train here.')
    else:
        # train
        if number_of_datasets == 0:
            print('Error: no training datasets have been specified')
            return
        elif number_of_datasets == 1:
            # normal shuffle, just take subset
            random.shuffle(train_neg)
            negative = train_neg[0:len(train_pos)]
        else:
            # can be train + test on multiple datasets
            # can be only train on multiple datasets
            # mixing does matter
            if adjustable.mix == True:
                # shuffle the data with each other
                # here we need to know if we only train or train+test
                if adjustable.dataset_test is None:
                    # normal shuffle, just take subset
                    random.shuffle(train_neg)
                    negative = train_neg[0:len(train_pos)]
                else:
                    if adjustable.mix_with_test == True:
                        # mix with the test
                        # normal shuffle, just take subset
                        random.shuffle(train_neg)
                        negative = train_neg[0:len(train_pos)]
                    else:
                        # don't mix with the test (which is at the end)
                        # for each partition, shuffle and get a subset
                        for index in range(len(train_neg)):
                            random.shuffle(train_neg[index])
                            negative.append(train_neg[index][0:len(train_pos[index])])

            else:
                # train in order.
                # number of datasets don't matter
                # for each partition, shuffle and get a subset
                for index in range(len(train_neg)):
                    random.shuffle(train_neg[index])
                    negative.append(train_neg[index][0:len(train_pos[index])])

    return negative


def fix_positives(positives):
    positives_fixed = []
    if len(np.shape(positives)) > 1:
        for index in range(len(positives)):
            positives_fixed += positives[index]
    else:
        positives_fixed = positives

    return positives_fixed


# final_training_data = get_final_training_data(adjustable, merged_training_pos, training_neg_sample)


def get_final_training_data(adjustable, train_pos, train_neg):
    final_training_data = []

    number_of_datasets = len(train_pos)

    if adjustable.only_test == True:
        # only test, nothing to do
        print('Only testing, nothing to train here.')
        final_training_data = None
    else:
        # train
        if number_of_datasets == 0:
            print('Error: no training datasets have been specified')
            return
        elif number_of_datasets == 1:
            # normal shuffle, just take subset
            final_training_data = train_pos + train_neg
            random.shuffle(final_training_data)
        else:
            # can be train + test on multiple datasets
            # can be only train on multiple datasets
            # mixing does matter
            if adjustable.mix == True:
                # shuffle the data with each other
                # here we need to know if we only train or train+test
                if adjustable.dataset_test is None:
                    # normal shuffle, just take subset
                    final_training_data = train_pos + train_neg
                    random.shuffle(final_training_data)
                else:
                    if adjustable.mix_with_test == True:
                        # mix with the test
                        # normal shuffle, just take subset
                        final_training_data = train_pos + train_neg
                        random.shuffle(final_training_data)
                    else:
                        # don't mix with the test (which is at the end)
                        # for each partition, shuffle and get a subset
                        for index in range(len(train_neg)):
                            partition = train_pos[index] + train_neg[index]
                            random.shuffle(partition)
                            final_training_data += partition

            else:
                # train in order.
                # number of datasets don't matter
                # for each partition, shuffle and get a subset
                for index in range(len(train_neg)):
                    partition = train_pos[index] + train_neg[index]
                    random.shuffle(partition)
                    final_training_data += partition

    return final_training_data


def get_ranking(all_ranking):
    updated_ranking = all_ranking[-1]
    return updated_ranking


# def main(adjustable, h5_data_list, all_ranking, merged_training_pos, merged_training_neg):
def main(adjustable, training_h5, testing_h5, all_ranking, merged_training_pos, merged_training_neg):
    """
    Runs a the whole training and testing phase
    :param adjustable:              object of class ProjectVariable
    :param training_h5:             list of h5py object(s) containing training datasets
    :param testing_h5:              list of h5py object containing test/rank dataset
    :param all_ranking:             list of ranking pair string paths to images
    :param merged_training_pos:     list of training pos pair string paths to images
    :param merged_training_neg:     list of training neg pair string paths to images
    :return:    array of dataset names, array containing the confusion matrix for each dataset, array containing the
                ranking for each dataset
    """
    model = get_model(adjustable)

    ############################################################################################################
    #   Training phase
    ############################################################################################################
    if adjustable.test_only == False:
        for epoch in range(adjustable.epochs):
            print('Epoch %d/%d' % (epoch, adjustable.epochs))
            ############################################################################################################
            #   Prepare the training data
            ############################################################################################################
            # DONE TODO: for each training set, do the sampling in new method get_negative_sample()
            # DONE TODO: create get_negative_sample()
            training_neg_sample = get_negative_sample(adjustable, merged_training_pos, merged_training_neg)

            # DONE TODO: fix the positive samples
            # merged_training_pos = fix_positives(merged_training_pos)

            # sample from the big set of negative training instances
            # random.shuffle(merged_training_neg)
            # training_neg_sample = merged_training_neg[0:len(merged_training_pos)]

            # now we have the final list of keys to the instances we use for training
            # final_training_data = merged_training_pos + training_neg_sample
            # random.shuffle(final_training_data)
            # final_training_data = pu.sideways_shuffle(final_training_data)
            # DONE TODO: create get_final_training_data()
            final_training_data = get_final_training_data(adjustable, merged_training_pos, training_neg_sample)

            final_training_labels = [int(final_training_data[item].strip().split(',')[-1]) for item in
                                     range(len(final_training_data))]
            if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
                final_training_labels = keras.utils.to_categorical(final_training_labels, pc.NUM_CLASSES)

            ############################################################################################################
            #   Train the network
            ############################################################################################################

            # train_network(adjustable, model, final_training_data, final_training_labels, h5_data_list)
            train_network(adjustable, model, final_training_data, final_training_labels, training_h5, testing_h5)

            ############################################################################################################
            #   Save the model + weights (if specified with adjustable.save_inbetween and adjustable.save_points)
            ############################################################################################################
            time_stamp = time.strftime('scnn_%d%m%Y_%H%M')

            if adjustable.save_inbetween and adjustable.iterations == 1:
                if epoch + 1 in adjustable.save_points:
                    if adjustable.name_indication == 'epoch':
                        model_name = time_stamp + '_epoch_%s_model.h5' % str(epoch + 1)
                        weights_name = time_stamp + '_epoch_%s_weigths.h5' % str(epoch + 1)
                    elif adjustable.name_indication == 'dataset_name' and len(adjustable.datasets) == 1:
                        model_name = '%s_model_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                        weights_name = '%s_weigths_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                    else:
                        model_name = None
                        weights_name = None

                    model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, model_name))
                    model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name))
                    print('MODEL SAVED at epoch %d' % (epoch + 1))


                    #
                    #     if all_ranking is None:
                    #         if training_h5 is not None:
                    #             # only train, using all datasets
                    #             training_neg_sample = get_negative_sample(merged_training_pos, merged_training_neg)
                    #             final_training_data = get_final_training_data(adjustable, merged_training_pos, training_neg_sample)
                    #
                    #             final_training_labels = [int(final_training_data[item].strip().split(',')[-1]) for item in
                    #                                      range(len(final_training_data))]
                    #             pass
                    #         else:
                    #             print('ERROR')
                    #             return
                    #     else:
                    #         if adjustable.test_only == True:
                    #             # only test
                    #             pass
                    #         else:
                    #             if training_h5 is not None:
                    #                 # train on multiple datasets
                    #                 pass
                    #             else:
                    #                 # train on one dataset only
                    #                 pass

                    # ################################################################################################################
                    # #   Prepare the training data
                    # ################################################################################################################
                    #
                    # training_neg_sample = get_negative_sample(merged_training_pos, merged_training_neg)
                    #
                    # # sample from the big set of negative training instances
                    # # random.shuffle(merged_training_neg)
                    # # training_neg_sample = merged_training_neg[0:len(merged_training_pos)]
                    #
                    # # now we have the final list of keys to the instances we use for training
                    # # final_training_data = merged_training_pos + training_neg_sample
                    # # random.shuffle(final_training_data)
                    # # final_training_data = pu.sideways_shuffle(final_training_data)
                    # final_training_data = get_final_training_data(adjustable, merged_training_pos, training_neg_sample)
                    #
                    # final_training_labels = [int(final_training_data[item].strip().split(',')[-1]) for item in
                    #                          range(len(final_training_data))]
                    # if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
                    #     final_training_labels = keras.utils.to_categorical(final_training_labels, pc.NUM_CLASSES)

                    # ################################################################################################################
                    # #   Train the network, save if specified
                    # ################################################################################################################
                    #
                    # train_network(adjustable, model, final_training_data, final_training_labels, h5_data_list)
                    #
                    # time_stamp = time.strftime('scnn_%d%m%Y_%H%M')
                    #
                    # if adjustable.save_inbetween and adjustable.iterations == 1:
                    #     if epoch+1 in adjustable.save_points:
                    #         if adjustable.name_indication == 'epoch':
                    #             model_name = time_stamp + '_epoch_%s_model.h5' % str(epoch + 1)
                    #             weights_name = time_stamp + '_epoch_%s_weigths.h5' % str(epoch + 1)
                    #         elif adjustable.name_indication == 'dataset_name' and len(adjustable.datasets) == 1:
                    #             model_name = '%s_model_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                    #             weights_name = '%s_weigths_%s.h5' % (adjustable.datasets[0], adjustable.use_gpu)
                    #         else:
                    #             model_name = None
                    #             weights_name = None
                    #
                    #         model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, model_name))
                    #         model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name))
                    #         print('MODEL SAVED at epoch %d' % (epoch + 1))

    # confusion_matrices = []
    # gregor_matrices = []
    # ranking_matrices = []
    # names = []

    ############################################################################################################
    #   Testing phase
    ############################################################################################################
    if all_ranking is not None:
        confusion_matrices = []
        gregor_matrices = []
        ranking_matrices = []
        # DONE TODO: go through all the ranking files and save all the ones before last. Only test on the last.
        this_ranking = all_ranking[-1]

        # all_ranking = save_non_target_datasets_rankings(all_ranking)


        # HERE WE ARE WHEN LEAVING OFFICE



        # DONE TODO: modify that this is only for the dataset that we test on
        # dataset = 0
        # name = adjustable.dataset_test
        # for dataset in range(len(adjustable.datasets)):
        ################################################################################################################
        #   Prepare the testing/ranking data
        ################################################################################################################
        # this_ranking = all_ranking[dataset]

        # DONE TODO: update ddl.grab_em_by_the_keys() with new parameters
        # test_data = ddl.grab_em_by_the_keys(this_ranking, h5_data_list)
        test_data = ddl.grab_em_by_the_keys(this_ranking, training_h5, testing_h5)
        test_data = np.asarray(test_data)

        # prepare for testing the model
        final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in range(len(this_ranking))]

        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

        ################################################################################################################
        #   Test
        ################################################################################################################
        print('Testing...')
        predictions = model.predict([test_data[0, :], test_data[1, :]])

        ################################################################################################################
        #   Process the results
        ################################################################################################################
        print('Processing results...')
        if adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            new_thing = zip(predictions, final_testing_labels)
            print(new_thing[0:50])

        # create confusion matrix
        matrix = pu.make_confusion_matrix(adjustable, predictions, final_testing_labels)
        confusion_matrices.append(matrix)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0

        # [upon Gregor's request] create a 0.1 ratio version of the confusion matrix where for each positive instance
        #                                                                              there are 9 negative instances
        gregor_matrix = pu.make_gregor_matrix(adjustable, predictions, final_testing_labels)
        print(gregor_matrix)
        gregor_matrices.append(gregor_matrix)

        if (gregor_matrix[0] * 1.0 + gregor_matrix[3] * 1.0) == 0:
            detection_rate = 0
        else:
            detection_rate = (gregor_matrix[0] * 1.0 / (gregor_matrix[0] * 1.0 + gregor_matrix[3] * 1.0))

        if (gregor_matrix[1] * 1.0 + gregor_matrix[2] * 1.0) == 0:
            false_alarm = 0
        else:
            false_alarm = (gregor_matrix[1] * 1.0 / (gregor_matrix[1] * 1.0 + gregor_matrix[2] * 1.0))

        # calculate the Cumulative Matching Characteristic
        ranking = pu.calculate_CMC(adjustable, predictions)
        ranking_matrices.append(ranking)

        print(
        '%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \nCMC: \n%s \nDetection rate: %s  False alarm: %s'
        % (
        adjustable.dataset_test, accuracy, precision, str(matrix), str(ranking), str(detection_rate), str(false_alarm)))

    else:
        confusion_matrices = None
        ranking_matrices = None
        gregor_matrices = None

    del model
    return confusion_matrices, ranking_matrices, gregor_matrices


def super_main(adjustable):
    """Runs main for a specified iterations. Useful for experiment running.
    Note: set iterations to 1 if you want to save weights
    """
    # load the datasets from h5
    # all_h5_datasets = ddl.load_datasets_from_h5(adjustable.datasets)

    ################################################################################################################
    #   Load datasets, note: always 1 dataset_test, but multiple datasets_train
    ################################################################################################################
    datasets_train_h5 = ddl.load_datasets_from_h5(adjustable.datasets_train)
    dataset_test_h5 = ddl.load_datasets_from_h5(adjustable.dataset_test)

    ################################################################################################################
    #   Set the ranking number.
    ################################################################################################################
    if dataset_test_h5 is None:
        if datasets_train_h5 is not None:
            if adjustable.ranking_number_test is None:
                print('Note: Only training will be performed.')
                ranking_number = None
            else:
                print('Warning: No ranking number needed, ranking number defaults to `None`.')
                print('Note: Only training will be performed.')
                ranking_number = None
        else:
            print('Error: No training data specified.')
            return
    else:
        print('Note: Testing (Ranking) will also be performed.')
        if adjustable.ranking_number_test == 'half':
            ranking_number = pc.RANKING_DICT[dataset_test_h5[0]]
        elif isinstance(adjustable.ranking_number_test, int):
            ranking_number = adjustable.ranking_number
        else:
            print('Error: Unknown configuration.')
            return

    ################################################################################################################
    #   [IF dataset_test_h5 is not None] Create arrays in which we store the results
    ################################################################################################################
    # number_of_datasets = len(adjustable.datasets)
    if dataset_test_h5 is not None:
        # number_of_datasets = 1
        # name = np.zeros(number_of_datasets)
        # name = adjustable.dataset_test
        confusion_matrices = np.zeros((adjustable.iterations, 4))
        ranking_matrices = np.zeros((adjustable.iterations, ranking_number))
        gregor_matrices = np.zeros((adjustable.iterations, 4))
    else:
        # name = None
        confusion_matrices = None
        ranking_matrices = None
        gregor_matrices = None

    ################################################################################################################
    #   Start a number of experiment iterations
    ################################################################################################################
    start = time.time()
    for iter in range(adjustable.iterations):
        print('-----EXPERIMENT ITERATION %d' % iter)
        # lists for storing intermediate results
        all_ranking, all_training_pos, all_training_neg = [], [], []
        # create training and ranking set for all datasets
        ss = time.time()

        if dataset_test_h5 is None:
            print('Training using all data in datasets_train.')
            ############################################################################################################
            #   Prepare data for when we only train using all data
            ############################################################################################################
            if datasets_train_h5 is not None:
                for index in range(len(adjustable.datasets_train)):
                    # DONE TODO: modify `ddl.create_training_and_ranking_set` to handle `do_ranking=False`
                    # DONE TODO: add ranking_variable parameter to create_training_and_ranking_set
                    training_pos, training_neg = ddl.create_training_and_ranking_set(adjustable.datasets_train[index],
                                                                                     adjustable, ranking_variable=None,
                                                                                     do_ranking=False)
                    if adjustable.cost_module_type in ['euclidean', 'cosine']:
                        training_pos = pu.flip_labels(training_pos)
                        training_neg = pu.flip_labels(training_neg)

                    all_training_pos.append(training_pos)
                    all_training_neg.append(training_neg)
            else:
                print('Error: no training data specified.')
                return
        else:
            # fix issue where it can be datasets_train != none + test_only == true
            if adjustable.test_only == True:
                print('Testing only using ranking set based on dataset_test.')
                ########################################################################################################
                #   Prepare data for when we ONLY test. Randomly get the data or load from a file if file exists
                ########################################################################################################
                # DONE TODO: modify `ddl.create_training_and_ranking_set` to handle `do_training=False`
                # DONE TODO: add ranking_variable parameter to create_training_and_ranking_set
                ranking = ddl.create_training_and_ranking_set(adjustable.dataset_test, adjustable,
                                                              ranking_variable=adjustable.ranking_number_test,
                                                              do_training=False)
                if adjustable.cost_module_type in ['euclidean', 'cosine']:
                    ranking = pu.flip_labels(ranking)
                all_ranking.append(ranking)
            else:
                if datasets_train_h5 is not None:
                    print('Training and testing on multiple datasets.')
                    ####################################################################################################
                    #   Prepare data for when we train on multiple datasets and test
                    ####################################################################################################
                    # note: remember that only the last ranking in the ranking matrix will be tested on.
                    # DONE TODO: add ranking_variable parameter to create_training_and_ranking_set
                    for index in range(len(adjustable.datasets_train)):
                        ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(
                            adjustable.datasets_train[index],
                            adjustable,
                            ranking_variable=adjustable.ranking_number_train[index])
                        if adjustable.cost_module_type in ['euclidean', 'cosine']:
                            ranking = pu.flip_labels(ranking)
                            training_pos = pu.flip_labels(training_pos)
                            training_neg = pu.flip_labels(training_neg)

                        all_ranking.append(ranking)
                        all_training_pos.append(training_pos)
                        all_training_neg.append(training_neg)

                    # DONE TODO: add ranking_variable parameter to create_training_and_ranking_set
                    ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(
                        adjustable.dataset_test, adjustable, ranking_variable=adjustable.ranking_number_test)
                    if adjustable.cost_module_type in ['euclidean', 'cosine']:
                        ranking = pu.flip_labels(ranking)
                        training_pos = pu.flip_labels(training_pos)
                        training_neg = pu.flip_labels(training_neg)

                    all_ranking.append(ranking)
                    all_training_pos.append(training_pos)
                    all_training_neg.append(training_neg)

                else:
                    print('Training and testing on a single dataset.')
                    ####################################################################################################
                    #   Prepare data for when we train and test on a single dataset
                    ####################################################################################################
                    # DONE TODO: add ranking_variable parameter to create_training_and_ranking_set
                    ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(
                        adjustable.dataset_test, adjustable, ranking_variable=adjustable.ranking_number_test)
                    if adjustable.cost_module_type in ['euclidean', 'cosine']:
                        ranking = pu.flip_labels(ranking)
                        training_pos = pu.flip_labels(training_pos)
                        training_neg = pu.flip_labels(training_neg)

                    all_ranking.append(ranking)
                    all_training_pos.append(training_pos)
                    all_training_neg.append(training_neg)

        # for name in range(len(adjustable.datasets)):
        #     ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(adjustable.datasets[name], adjustable)
        #     # labels have different meanings in `euclidean` case, 0 for match and 1 for mismatch
        #     if adjustable.cost_module_type in ['euclidean', 'cosine']:
        #         ranking = pu.flip_labels(ranking)
        #         training_pos = pu.flip_labels(training_pos)
        #         training_neg = pu.flip_labels(training_neg)
        #
        #     # data gets appended in order
        #     all_ranking.append(ranking)
        #     all_training_pos.append(training_pos)
        #     all_training_neg.append(training_neg)
        # # put all the training data together

        st = time.time()
        print('%0.2f mins' % ((st - ss) / 60))

        ################################################################################################################
        #   Merge the training data.
        #   Here we decide how to merge: to mix or to order by using adjustable.mix
        #   Also for training on multiple datasets + testing: decide if we include test set in the training to be mixed:
        #   by using adjustable.mix_with_test
        ################################################################################################################
        # DONE TODO: update ddl.merge_datasets(), what if empty list is given in parameters?
        merged_training_pos, merged_training_neg = ddl.merge_datasets(adjustable, all_training_pos, all_training_neg)

        ################################################################################################################
        #   Run main()
        ################################################################################################################
        # name, confusion_matrix, ranking_matrix, gregor_matrix = main(adjustable, all_h5_datasets, all_ranking, merged_training_pos,
        #                                               merged_training_neg)
        # DONE TODO: update `main()`
        confusion_matrix, ranking_matrix, gregor_matrix = main(adjustable, datasets_train_h5, dataset_test_h5,
                                                               all_ranking, merged_training_pos,
                                                               merged_training_neg)

        if dataset_test_h5 is not None:
            # store results
            confusion_matrices[iter] = confusion_matrix
            ranking_matrices[iter] = ranking_matrix
            gregor_matrices[iter] = gregor_matrix

    stop = time.time()
    total_time = stop - start

    ################################################################################################################
    #   Calculate the means and standard deviations and log the results
    ################################################################################################################

    if dataset_test_h5 is not None:
        matrix_means = np.mean(confusion_matrices, axis=0)
        matrix_std = np.std(confusion_matrices, axis=0)
        ranking_means = np.mean(ranking_matrices, axis=0)
        ranking_std = np.std(ranking_matrices, axis=0)
        gregor_matrix_means = np.mean(gregor_matrices, axis=0)
        gregor_matrix_std = np.std(gregor_matrices, axis=0)
        name = adjustable.dataset_test
    else:
        matrix_means = None
        matrix_std = None
        ranking_means = None
        ranking_std = None
        gregor_matrix_means = None
        gregor_matrix_std = None
        name = None

    # matrix_means = np.zeros((1, 4))
    # matrix_std = np.zeros((1, 4))
    # ranking_means = np.zeros((1, ranking_number))
    # ranking_std = np.zeros((1, ranking_number))
    # gregor_matrix_means = np.zeros((1, 4))
    # gregor_matrix_std = np.zeros((1, 4))
    #
    # # for each dataset, create confusion and ranking matrices
    # for dataset in range(number_of_datasets):
    #     matrices = np.zeros((adjustable.iterations, 4))
    #     rankings = np.zeros((adjustable.iterations, ranking_number))
    #     g_matrices = np.zeros((adjustable.iterations, 4))
    #
    #     for iter in range(adjustable.iterations):
    #         matrices[iter] = confusion_matrices[iter][dataset]
    #         rankings[iter] = ranking_matrices[iter][dataset]
    #         g_matrices[iter] = gregor_matrices[iter][dataset]
    #
    #     # calculate the mean and std
    #     matrix_means[dataset] = np.mean(matrices, axis=0)
    #     matrix_std[dataset] = np.std(matrices, axis=0)
    #     ranking_means[dataset] = np.mean(rankings, axis=0)
    #     ranking_std[dataset] = np.std(rankings, axis=0)
    #     gregor_matrix_means[dataset] = np.mean(g_matrices, axis=0)
    #     gregor_matrix_std[dataset] = np.std(g_matrices, axis=0)

    # log the results
    if adjustable.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable, adjustable.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, gregor_matrix_means, gregor_matrix_std)