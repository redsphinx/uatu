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


# FIXME use pv for module_type and neural_distance
def create_cost_module(inputs, module_type, neural_distance):
    """Implements the cost module of the siamese network.
    :param inputs:          list containing feature tensor from each siamese head
    :param module_type:     the type of cost module.
                            choice of: 'neural_network', 'euclidean'
    :param neural_distance: the operation to perform with the siamese head features.
                            choice of: 'concatenate', 'add', 'multiply'
    :return:                some type of distance
    """
    if module_type == 'neural_network':
        if neural_distance == 'concatenate':
            features = keras.layers.concatenate(inputs)
        elif neural_distance == 'add':
            features = keras.layers.add(inputs)
        elif neural_distance == 'multiply':
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

    elif module_type == 'euclidean':
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(inputs)
        return distance

    else:
        return None


def add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name, trainable):
    """One-liner for adding activation and max pooling
    :param model:       the model to add to
    :return:            the model with added activation and max pooling
    :param trainable:   boolean indicating if layer is trainable
    """
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if use_batch_norm:
        model.add(BatchNormalization(name=batch_norm_name, trainable=trainable))
    return model


def create_siamese_head(numfil, head_type, transfer_weights, weights_name, trainable):
    """Implements 1 head of the siamese network.
    :param numfil:              multiply the number of convolutional filters with this number
    :param head_type:                the type of head for the siamese network. choice of 'simple', 'batch_normalized'
    :param transfer_weights:    use trained weights from h5 file
    :param weights_name:        h5 file to load weights from
    :param trainable:           boolean indicating if layer is trainable
    :return:                    a keras Sequential model
    """
    use_batch_norm = True if head_type == 'batch_normalized' else False

    model = Sequential()
    model.add(Conv2D(16 * numfil, kernel_size=(3, 3), padding='same',
                     input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                     name='conv_1',
                     trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_1', trainable=trainable)
    model.add(Conv2D(32 * numfil, kernel_size=(3, 3), padding='same', name='conv_2', trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_2', trainable=trainable)
    model.add(Conv2D(64 * numfil, kernel_size=(3, 3), padding='same', name='conv_3', trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_3', trainable=trainable)
    model.add(Conv2D(128 * numfil, kernel_size=(3, 3), padding='same', name='conv_4', trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_4', trainable=trainable)
    model.add(Conv2D(256 * numfil, kernel_size=(3, 3), padding='same', name='conv_5', trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_5', trainable=trainable)
    model.add(Conv2D(512 * numfil, kernel_size=(3, 3), padding='same', name='conv_6', trainable=trainable))
    model = add_activation_and_max_pooling(model, use_batch_norm, batch_norm_name='bn_6', trainable=trainable)
    model.add(Conv2D(1024 * numfil, kernel_size=(3, 3), padding='same', name='conv_7', trainable=trainable))
    model.add(Activation('relu'))
    if use_batch_norm == True:
        model.add(BatchNormalization(name='bn_7', trainable=trainable))
    model.add(Flatten(name='cnn_flat'))

    if transfer_weights == True:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name), by_name=True)

    return model


def create_siamese_network(numfil, head_type, cost_module_type, neural_distance, trainable, transfer_weights,
                           weights_name):
    """Creates the siamese network.

    :param numfil:              multiply the number of convolutional filters with this number
    :param transfer_weights:    boolean indicating use trained weights from h5 file
    :param weights_name:        h5 file to load weights from
    :param head_type:           the type of head for the siamese network.
                                choice of 'simple', 'batch_normalized'
    :param cost_module_type:    the type of cost module.
                                choice of: 'neural_network', 'euclidean'
    :param neural_distance:     the operation to perform with the siamese head features.
                                choice of: 'concatenate', 'add', 'multiply'
    :param trainable:           boolean indicating if layer is trainable
    :return:
    """
    input_a = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    input_b = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

    siamese_head = create_siamese_head(numfil=numfil, head_type=head_type, transfer_weights=transfer_weights,
                                       weights_name=weights_name, trainable=trainable)

    processed_a = siamese_head(input_a)
    processed_b = siamese_head(input_b)

    distance = create_cost_module([processed_a, processed_b], module_type=cost_module_type,
                                  neural_distance=neural_distance)
    model = Model([input_a, input_b], distance)
    return model


def train_network(model, step, validation_interval, cl, cl_min, cl_max, batch_size, train_data, train_labels,
                  validation_data, validation_labels):
    if cl:
        clr = CyclicLR(step_size=(np.shape(train_data)[0] / batch_size) * 8, base_lr=cl_min,
                       max_lr=cl_max)
        if step % validation_interval == 0:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=batch_size,
                      epochs=1,
                      validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                      verbose=2,
                      callbacks=[clr])
        else:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=0,
                      callbacks=[clr])
    else:
        if step % validation_interval == 0:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=batch_size,
                      epochs=1,
                      validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                      verbose=2)
        else:
            model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=0)


def main(numfil, head_type, cost_module_type, neural_distance, trainable, transfer_weights, cnn_weights_name, lr,
         epochs, cl, cl_min, cl_max, batch_size, scnn_save_weights_name):
    total_data_list, validation, test = ddl.get_data_scnn()
    total_data_list_pos, total_data_list_neg = total_data_list
    validation_data, validation_labels = validation

    model = create_siamese_network(numfil=numfil,
                                   head_type=head_type,
                                   cost_module_type=cost_module_type,
                                   neural_distance=neural_distance,
                                   trainable=trainable,
                                   transfer_weights=transfer_weights,
                                   weights_name=cnn_weights_name)

    slice_size = 5000
    train_data_size = 2 * min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps_per_epoch = np.ceil(train_data_size * 1.0 / slice_size).astype(int)

    num_validations_per_epoch = 1  # note: must be at least 1
    if num_validations_per_epoch > num_steps_per_epoch: num_validations_per_epoch = num_steps_per_epoch
    validation_interval = np.floor(num_steps_per_epoch / num_validations_per_epoch).astype(int)

    if cost_module_type == 'neural_network':
        nadam = optimizers.Nadam(lr=lr, schedule_decay=pc.DECAY_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
    elif cost_module_type == 'euclidean':
        rms = keras.optimizers.RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)

    for epoch in range(epochs):
        print('------EPOCH: %d' % epoch)
        slice_size_queue = ddl.make_slice_queue(train_data_size, slice_size)
        total_train_data_list = ddl.make_train_batches(total_data_list_pos, total_data_list_neg, data_type='images')
        for step in range(num_steps_per_epoch):
            train_data_list = total_train_data_list[step * slice_size: step * slice_size + slice_size_queue[step]]

            train_data, train_labels = ddl.load_in_array(data_list=train_data_list,
                                                         heads=2,
                                                         data_type='images')

            train_network(model=model, step=step, validation_interval=validation_interval, cl=cl, cl_min=cl_min,
                          cl_max=cl_max, batch_size=batch_size, train_data=train_data, train_labels=train_labels,
                          validation_data=validation_data, validation_labels=validation_labels)

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
        matrix = pu.make_confusion_matrix(predictions, test_labels)
        accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
        if not matrix[0] == 0:
            precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
        else:
            precision = 0
        confusion_matrices.append(matrix)

        ranking = pu.calculate_CMC(predictions)
        ranking_matrices.append(ranking)

        print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
              % (name, accuracy, precision, str(matrix), str(ranking)))

    if not scnn_save_weights_name == None:
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, scnn_save_weights_name))

    del model
    return names, confusion_matrices, ranking_matrices


def super_main(iterations, experiment_name, numfil, head_type, cost_module_type, neural_distance, trainable,
               transfer_weights, cnn_weights_name, lr, epochs, cl, cl_min, cl_max, batch_size, scnn_save_weights_name):
    number_of_datasets = 2
    name = np.zeros(number_of_datasets)
    confusion_matrices = np.zeros((iterations, number_of_datasets, 4))
    ranking_matrices = np.zeros((iterations, number_of_datasets, pc.RANKING_NUMBER))

    start = time.time()
    for iter in range(0, iterations):
        print('-----ITERATION %d' % iter)

        name, confusion_matrix, ranking_matrix = main(numfil, head_type, cost_module_type, neural_distance,
                                                      trainable, transfer_weights, cnn_weights_name, lr,
                                                      epochs, cl, cl_min, cl_max, batch_size,
                                                      scnn_save_weights_name)

        confusion_matrices[iter] = confusion_matrix
        ranking_matrices[iter] = ranking_matrix

    stop = time.time()
    total_time = stop - start

    matrix_means = np.zeros((number_of_datasets, 4))
    matrix_std = np.zeros((number_of_datasets, 4))
    ranking_means = np.zeros((number_of_datasets, pc.RANKING_NUMBER))
    ranking_std = np.zeros((number_of_datasets, pc.RANKING_NUMBER))

    for dataset in range(number_of_datasets):
        matrices = np.zeros((iterations, 4))
        rankings = np.zeros((iterations, pc.RANKING_NUMBER))

        for iter in range(iterations):
            matrices[iter] = confusion_matrices[iter][dataset]
            rankings[iter] = ranking_matrices[iter][dataset]

        matrix_means[dataset] = np.mean(matrices, axis=0)
        matrix_std[dataset] = np.std(matrices, axis=0)
        ranking_means[dataset] = np.mean(rankings, axis=0)
        ranking_std[dataset] = np.std(rankings, axis=0)

    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(experiment_name, file_name, name, matrix_means, matrix_std, ranking_means, ranking_std, total_time)

