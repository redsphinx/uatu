import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda, BatchNormalization, AveragePooling2D
from keras import optimizers
from keras import backend as K
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
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


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
            features = keras.layers.concatenate(inputs)
        elif adjustable.neural_distance == 'add':
            features = keras.layers.add(inputs)
        elif adjustable.neural_distance == 'multiply':
            features = keras.layers.multiply(inputs)
        elif adjustable.neural_distance == 'subtract':
            features = keras.layers.merge(inputs=inputs, mode=subtract, output_shape=the_shape)
        elif adjustable.neural_distance == 'divide':
            features = keras.layers.merge(inputs=inputs, mode=divide, output_shape=the_shape)
        elif adjustable.neural_distance == 'absolute':
            features = keras.layers.merge(inputs=inputs, mode=absolute, output_shape=the_shape)
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

    elif adjustable.cost_module_type == 'euclidean_fc':
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        dense_layer = Dense(1, name='dense_1')(distance)
        activation = Activation(adjustable.activation_function)(dense_layer)
        output_layer = Dense(pc.NUM_CLASSES, name='ouput')(activation)
        softmax = Activation('softmax')(output_layer)
        return softmax

    elif adjustable.cost_module_type == 'cosine':
        distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)(inputs)
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
            model.add(AveragePooling2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
        else:  # max_pooling
            model.add(MaxPool2D(pool_size=(adjustable.pooling_size[0][0], adjustable.pooling_size[0][1])))
    else:
        if adjustable.pooling_type == 'avg_pooling':
            model.add(AveragePooling2D(pool_size=(adjustable.pooling_size[1][0], adjustable.pooling_size[1][1])))
        else:  # max_pooling
            model.add(MaxPool2D(pool_size=(adjustable.pooling_size[1][0], adjustable.pooling_size[1][1])))

    model.add(Activation(adjustable.activation_function))

    if use_batch_norm:
        model.add(BatchNormalization(name=batch_norm_name, trainable=adjustable.trainable))
    return model


def create_siamese_head(adjustable):
    """Implements 1 head of the siamese network.
    :return:                    a keras Sequential model
    """
    use_batch_norm = True if adjustable.head_type == 'batch_normalized' else False

    model = Sequential()
    if use_batch_norm == True:
        model.add(BatchNormalization(name='bn_1', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                                     trainable=adjustable.trainable))
    model.add(Conv2D(16 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_1',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_2', first_layer=True)
    model.add(Conv2D(32 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_2',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_3')
    model.add(Conv2D(64 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_3',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_4')
    model.add(Conv2D(128 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_4',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_5')
    model.add(Conv2D(256 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_5',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_6')
    model.add(Conv2D(512 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_6',
                     trainable=adjustable.trainable))
    model = add_activation_and_max_pooling(adjustable, model, use_batch_norm, batch_norm_name='bn_7')
    if adjustable.pooling_size == [[2, 2], [2, 2]]:
        model.add(Conv2D(1024 * adjustable.numfil, kernel_size=adjustable.kernel, padding='same', name='conv_7',
                         trainable=adjustable.trainable))
        model.add(Activation(adjustable.activation_function))
        if use_batch_norm == True:
            model.add(BatchNormalization(name='bn_8', trainable=adjustable.trainable))
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
        model = load_model(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.load_model_name))
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

        if adjustable.save_inbetween and adjustable.iterations == 1:
            if epoch+1 in adjustable.save_points:
                if adjustable.name_indication == 'epoch':
                    model_name = time_stamp + '_epoch_%s_model.h5' % str(epoch + 1)
                    weights_name = time_stamp + '_epoch_%s_weights.h5' % str(epoch + 1)
                elif adjustable.name_indication == 'dataset_name' and len(adjustable.datasets) == 1:
                    model_name = time_stamp + '_%s_model.h5' % adjustable.datasets[0]
                    weights_name = time_stamp + '_%s_weights.h5' % adjustable.datasets[0]
                else:
                    model_name = None
                    weights_name = None

                model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, model_name))
                model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name))

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
        if adjustable.save_inbetween and adjustable.iterations == 1:
            file_name = '%s_ranking_%s.txt' % (name, adjustable.ranking_time_name)
            file_name = os.path.join(pc.SAVE_LOCATION_RANKING_FILES, file_name)
            with open(file_name, 'w') as my_file:
                for item in this_ranking:
                    my_file.write(item)

        final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in range(len(this_ranking))]

        if adjustable.cost_module_type == 'neural_network' or adjustable.cost_module_type == 'euclidean_fc':
            final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

        predictions = model.predict([test_data[0, :], test_data[1, :]])
        # print predictions
        if adjustable.cost_module_type == 'euclidean' or adjustable.cost_module_type == 'cosine':
            new_thing = zip(predictions, final_testing_labels)
            print(new_thing[0:50])

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
    all_h5_datasets = ddl.load_datasets_from_h5(adjustable.datasets)
    # select which GPU to use, necessary to start tf session
    os.environ["CUDA_VISIBLE_DEVICES"] = adjustable.use_gpu
    # arrays for storing results
    number_of_datasets = len(adjustable.datasets)
    name = np.zeros(number_of_datasets)
    confusion_matrices = np.zeros((adjustable.iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((adjustable.iterations, number_of_datasets, pc.RANKING_NUMBER))

    start = time.time()
    for iter in range(adjustable.iterations):
        print('-----ITERATION %d' % iter)
        # lists for storing intermediate results
        all_ranking, all_training_pos, all_training_neg = [], [], []
        # create training and ranking set for all datasets
        ss = time.time()
        for name in range(len(adjustable.datasets)):
            ranking, training_pos, training_neg = ddl.create_training_and_ranking_set(adjustable.datasets[name])
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
    ranking_means = np.zeros((number_of_datasets, pc.RANKING_NUMBER))
    ranking_std = np.zeros((number_of_datasets, pc.RANKING_NUMBER))
    # for each dataset, create confusion and ranking matrices
    for dataset in range(number_of_datasets):
        matrices = np.zeros((adjustable.iterations, 4))
        rankings = np.zeros((adjustable.iterations, pc.RANKING_NUMBER))

        for iter in range(adjustable.iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]
        # calculate the mean and std
        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)
    # log the results
    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(adjustable.experiment_name, file_name, name, matrix_means, matrix_std, ranking_means, ranking_std,
                        total_time)
