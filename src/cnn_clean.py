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

import project_constants as pc
import project_utils as pu
import dynamic_data_loading as ddl
from clr_callback import *
import h5py

import time
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


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


def create_cnn_model(adjustable):
    """Implements a convolutional neural network
    :return:                    a keras models.Sequential model
    """
    use_batch_norm = True if adjustable.head_type == 'batch_normalized' else False

    # convolutional unit 1
    model = models.Sequential()
    if use_batch_norm == True:
        model.add(layers.BatchNormalization(name='bn_1', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
                                            trainable=adjustable.trainable_bn))
        # model.add(layers.BatchNormalization(name='bn_1', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS),
        #                              trainable=adjustable.trainable_bn))
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


    model.add(layers.Dense(adjustable.neural_distance_layers[0], name='dense_1'))
    model.add(layers.Activation(adjustable.activation_function))
    model.add(layers.Dropout(pc.DROPOUT, name='dropout_1'))
    model.add(layers.Dense(adjustable.neural_distance_layers[1], name='dense_2'))
    model.add(layers.Activation(adjustable.activation_function))
    model.add(layers.Dropout(pc.DROPOUT, name='dropout_2'))
    model.add(layers.Dense(pc.NUM_CLASSES, name='output'))
    model.add(layers.Activation('softmax'))

    if not adjustable.weights_name == None:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, adjustable.weights_name), by_name=True)

    return model


def train(adjustable, model, train_data, train_labels, h5_data):
    if adjustable.use_cyclical_learning_rate:

        clr = CyclicLR(step_size=(len(train_labels) / adjustable.batch_size) * 8, base_lr=adjustable.cl_min,
                       max_lr=adjustable.cl_max)

        train_data = ddl.grab_em_by_the_keys(train_data, h5_data)

        train_data = np.asarray(train_data)

        model.fit(train_data, train_labels,
                  batch_size=adjustable.batch_size,
                  epochs=1,
                  validation_split=0.01,
                  verbose=2,
                  callbacks=[clr])
    else:
        train_data = ddl.grab_em_by_the_keys(train_data, h5_data)
        train_data = np.asarray(train_data)

        model.fit(train_data, train_labels,
                  batch_size=adjustable.batch_size,
                  epochs=1,
                  validation_split=0.01,
                  verbose=2)


def main(adjustable, test_data, train_data):
    """

    :param adjustable:      object of class ProjectVariable
    :param test_data:       list of keys for test data
    :param train_data:      list of keys for train data
    """

    model = create_cnn_model(adjustable)

    nadam = optimizers.Nadam(lr=adjustable.learning_rate, schedule_decay=pc.DECAY_RATE)
    model.compile(loss=adjustable.loss_function, optimizer=nadam, metrics=['accuracy'])

    for epoch in range(adjustable.epoch):
        print('Epoch: %d/%d' % (epoch, adjustable.epoch))


def super_main(adjustable):

    all_h5_datasets = ddl.load_datasets_from_h5(adjustable.datasets)
    


def main(experiment_name, weights_name, numfil, save_weights, epochs, batch_size, lr, data_type='hdf5'):
    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = data
    hdf5_file_path = '/home/gabi/PycharmProjects/uatu/data/pedestrian_all_data_uncompressed.h5'
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    if data_type=='hdf5':
        total_data_list_pos = np.array(xrange(hdf5_file['positives'].shape[0]))
        total_data_list_neg = np.array(xrange(hdf5_file['negatives'].shape[0]))
    else:
        total_data_list_pos, total_data_list_neg = pu.merge_pedestrian_sets()

    # val_list, test_list, total_data_list_pos, total_data_list_neg = ddl.make_validation_test_list(total_data_list_pos,
    #                                                                                               total_data_list_neg,
    #                                                                                               val_pos_percent=0.5,
    #                                                                                               test_pos_percent=0.5)
    val_list_pos, val_list_neg, test_list_pos, test_list_neg, total_data_list_pos, total_data_list_neg = \
    ddl.make_validation_test_list(total_data_list_pos, total_data_list_neg, val_pos_percent=0.5, test_pos_percent=0.5)

    # model = cnn_model_2d_conv_1d_filters(numfil)
    # model = cnn_model(numfil)
    model = cnn_model_2D_BN(numfil)

    if pc.VERBOSE:
        print(model.summary())

    nadam = optimizers.Nadam(lr=lr, schedule_decay=pc.DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    slice_size = 10000
    train_data_size = 2*min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps_per_epoch = np.ceil(train_data_size*1.0 / slice_size).astype(int)

    # in each epoch the training data gets partitioned into different batches. This way we break correlations between
    # instances and we can see many more negative examples
    start = time.time()
    val_data, val_labels = ddl.load_in_array(val_list_pos, val_list_neg, hdf5_file=hdf5_file)
    print('Time loading validation data: %0.3f seconds' % (time.time() - start))

    # note: minimal requirement that num_validations = 1
    num_validations = 1
    if num_validations > num_steps_per_epoch: num_validations = num_steps_per_epoch
    validation_interval = np.floor(num_steps_per_epoch / num_validations).astype(int)
    print('validation happens every %d step(s)' % validation_interval)

    for epoch in xrange(epochs):
        print('epoch: %d' % epoch)
        slice_size_queue = ddl.make_slice_queue(train_data_size, slice_size)
        total_train_data_list_pos, total_train_data_list_neg = \
                                                    ddl.make_train_batches(total_data_list_pos, total_data_list_neg)
        for step in xrange(num_steps_per_epoch):
            print('step: %d out of %d' % (step, num_steps_per_epoch))
            # start = time.time()
            class_slice = np.floor(slice_size_queue[step] / 2).astype(int)
            # train_data_list = total_train_data_list[step * batch_size : step * batch_size + slice_size_queue[step]]
            train_data_list_pos = total_train_data_list_pos[step * class_slice: step * class_slice + class_slice]
            train_data_list_neg = total_train_data_list_neg[step * class_slice: step * class_slice + class_slice]

            # train_data, train_labels = ddl.load_in_array(train_data_list)
            start = time.time()
            train_data, train_labels = ddl.load_in_array(train_data_list_pos, train_data_list_neg, hdf5_file=hdf5_file)
            print('Time loading training data: %0.3f seconds' % (time.time()-start))
            # let validation happen every x steps

            if step % validation_interval == 0:
                model.fit(train_data,
                          train_labels,
                          # batch_size=batch_size_queue[step],
                          batch_size=batch_size,
                          epochs=1,
                          validation_data=(val_data, val_labels),
                          verbose=2)
            else:
                model.fit(train_data,
                          train_labels,
                          # batch_size=batch_size_queue[step],
                          batch_size=batch_size,
                          epochs=1,
                          verbose=0)
            # stop = time.time()
            # the_time = stop - start
            # total_steps = pc.NUM_EPOCHS * num_steps
            # have_been = (epoch * num_steps) + step
            # remaining = (total_steps - have_been) * the_time
            # print('Epoch: %d / %d   Step: %d / %d  time: %0.2f s  remaining: %0.2f s' % (epoch, pc.NUM_EPOCHS, step,
            #                                                                            int(num_steps), the_time,
            #                                                                            remaining))
    # clr_triangular = CyclicLR(mode='exp_range', step_size=(np.shape(train_data)[0]/pc.BATCH_SIZE)*8)

    # model.fit(train_data,
    #           train_labels,
    #           batch_size=pc.BATCH_SIZE,
    #           epochs=pc.NUM_EPOCHS,
    #           validation_data=(validation_data, validation_labels),
    #           verbose=2)

    # test_data, test_labels = ddl.load_in_array(test_list)
    start = time.time()
    test_data, test_labels = ddl.load_in_array(test_list_pos, test_list_neg, hdf5_file=hdf5_file)
    print('Time loading testing data: %0.3f seconds' % (time.time() - start))

    hdf5_file.close()

    score = model.evaluate(test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_confusion_matrix = pu.make_confusion_matrix(model.predict(test_data), test_labels)

    # save model
    # note: change the saved weights names !!
    if save_weights:
        shit_name = weights_name
        name_weights = shit_name
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_weights))

    if pc.SAVE_CNN_MODEL:
        shit_name = 'shit.h5'
        name_model = shit_name
        model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_model))

    del model
    return test_confusion_matrix


def super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=False):
    accs = np.zeros((iterations, 4))

    start = time.time()
    for iter in range(0, iterations):
        accs[iter] = main(experiment_name, weights_name, numfil, save_weights, epochs, batch_size, lr)
    stop = time.time()

    total_time = stop - start

    print('mean values:')
    mean = np.mean(accs, axis=0)
    print(mean)

    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        dataset_name = 'INRIA, NICTA'
        pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name, total_time)

