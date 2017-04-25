import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda, BatchNormalization
from keras.optimizers import RMSprop
from keras import optimizers
from clr_callback import *
import dynamic_data_loading as ddl


from keras import backend as K

import project_constants as pc
import project_utils as pu
import os
import numpy as np
from keras.utils import plot_model
import time


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def alt_create_fc(inputs):
    # dense_layer = Dense(512)(inputs)
    # norm = BatchNormalization()(dense_layer)
    # dense_layer = Dense(1024)(norm)
    # norm = BatchNormalization()(dense_layer)
    # output_layer = Dense(pc.NUM_CLASSES)(norm)
    # softmax = Activation('softmax')(output_layer)
    # return softmax
    dense_layer = Dense(512, activation='relu') (inputs)
    activation = Activation('relu') (dense_layer)
    dropout_layer = Dropout(pc.DROPOUT)(activation)
    dense_layer = Dense(1024, activation='relu')(dropout_layer)
    activation = Activation('relu')(dense_layer)
    dropout_layer = Dropout(pc.DROPOUT)(activation)
    output_layer = Dense(pc.NUM_CLASSES) (dropout_layer)
    softmax = Activation('softmax')(output_layer)
    return  softmax


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def create_base_network_simple(numfil, weights_name):
    model = Sequential()
    model.add(Conv2D(16 * numfil, kernel_size=(3, 3), padding='same',
                     input_shape=(pc.NUM_SIAMESE_HEADS, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH,
                                  pc.NUM_CHANNELS), name='conv_1', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(32*numfil, kernel_size=(3, 3), padding='same', name='conv_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(64*numfil, kernel_size=(3, 3), padding='same', name='conv_3', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(128*numfil, kernel_size=(3, 3), padding='same', name='conv_4', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(256*numfil, kernel_size=(3, 3), padding='same', name='conv_5', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(512*numfil, kernel_size=(3, 3), padding='same', name='conv_6', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(1024*numfil, kernel_size=(3, 3), padding='same', name='conv_7', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name), by_name=True)

    return model


def create_base_network_1d_filter(train_data_, numfil, weights_name):
    model = Sequential()
    model.add(Conv2D(16*numfil, kernel_size=(1, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1_1',
                     trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(16*numfil, kernel_size=(3, 1), padding='same', name='conv_1_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(32*numfil, kernel_size=(1, 3), padding='same', name='conv_2_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(32*numfil, kernel_size=(3, 1), padding='same', name='conv_2_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(64*numfil, kernel_size=(1, 3), padding='same', name='conv_3_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(64*numfil, kernel_size=(3, 1), padding='same', name='conv_3_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(128*numfil, kernel_size=(1, 3), padding='same', name='conv_4_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(128*numfil, kernel_size=(3, 1), padding='same', name='conv_4_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(256*numfil, kernel_size=(1, 3), padding='same', name='conv_5_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(256*numfil, kernel_size=(3, 1), padding='same', name='conv_5_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(512*numfil, kernel_size=(1, 3), padding='same', name='conv_6_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(512*numfil, kernel_size=(3, 1), padding='same', name='conv_6_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(1024*numfil, kernel_size=(1, 3), padding='same', name='conv_7_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(1024*numfil, kernel_size=(3, 1), padding='same', name='conv_7_2', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))

    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name),
                           by_name=True)

    return model


def create_base_network_with_BN(train_data_):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1_1', 
                     use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_1', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same', name='conv_1_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_2', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', name='conv_2_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_3', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', name='conv_2_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_4', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(64, kernel_size=(1, 3), padding='same', name='conv_3_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_5', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same', name='conv_3_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_6', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(128, kernel_size=(1, 3), padding='same', name='conv_4_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_7', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(128, kernel_size=(3, 1), padding='same', name='conv_4_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_8', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(256, kernel_size=(1, 3), padding='same', name='conv_5_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_9', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(256, kernel_size=(3, 1), padding='same', name='conv_5_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_10', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(512, kernel_size=(1, 3), padding='same', name='conv_6_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_11', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(512, kernel_size=(3, 1), padding='same', name='conv_6_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_12', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(1024, kernel_size=(1, 3), padding='same', name='conv_7_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_13', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(1024, kernel_size=(3, 1), padding='same', name='conv_7_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_14', trainable=pc.TRAIN_CNN))

    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, 'cnn_model_weights_bn_clr_1_bias.h5'), by_name=True)

    return model


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


def create_siamese(numfil, weights_name, similarity_metric='fc_layers'):
    input_a = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    input_b = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    base_network = create_base_network_simple(numfil, weights_name)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    if similarity_metric == 'fc_layers':
        # use a fc to come up with a metric
        merged_processed = keras.layers.concatenate([processed_a, processed_b])
        distance = alt_create_fc(merged_processed)
        model = Model([input_a, input_b], distance)
    elif similarity_metric == 'euclid':
        # useful to show drawback of euclidean distance as a metric
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)

    return model


def confusion_matrix(name, predictions, labels, verbose=False):
    matrix = pu.make_confusion_matrix(predictions, labels)
    if verbose:
        pu.print_confusion_matrix(name, matrix)
    return matrix


def main(experiment_name, weights_name, numfil, similarity_metric='fc_layers'):

    total_data_list_pos = '/home/gabi/PycharmProjects/uatu/data/reid_all_positives.txt'
    total_data_list_neg = '/home/gabi/PycharmProjects/uatu/data/reid_all_negatives.txt'

    val_list, test_list, total_data_list_pos, total_data_list_neg = ddl.make_validation_test_list(total_data_list_pos,
                                                                                                  total_data_list_neg,
                                                                                                  val_pos_percent=0.1,
                                                                                                  test_pos_percent=0.1,
                                                                                                  data_type='images')

    model = create_siamese(numfil, weights_name)
    start = time.time()
    validation_data, validation_labels = ddl.load_in_array(data_list=val_list,
                                                           data_type='images',
                                                           heads=2)
    print('Time loading validation data: %0.3f seconds' % (time.time() - start))
    start = time.time()
    test_data, test_labels = ddl.load_in_array(data_list=test_list,
                                                           data_type='images',
                                                           heads=2)
    print('Time loading test data: %0.3f seconds' % (time.time() - start))

    slice_size = 5000
    train_data_size = 2 * min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps_per_epoch = np.ceil(train_data_size * 1.0 / slice_size).astype(int)

    num_validations = 5
    if num_validations > num_steps_per_epoch: num_validations = num_steps_per_epoch
    validation_interval = np.floor(num_steps_per_epoch / num_validations).astype(int)
    print('validation happens every %d step(s)' % validation_interval)

    if similarity_metric == 'fc_layers':
        nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

        # model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
        #           batch_size=pc.BATCH_SIZE,
        #           epochs=pc.NUM_EPOCHS,
        #           validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
        #           verbose=2)

        for epoch in xrange(pc.NUM_EPOCHS):
            print('epoch: %d' % epoch)
            slice_size_queue = ddl.make_slice_queue(train_data_size, slice_size)

            total_train_data_list = ddl.make_train_batches(total_data_list_pos, total_data_list_neg, data_type='images')
            for step in xrange(num_steps_per_epoch):
                print('step: %d out of %d' % (step, num_steps_per_epoch))
                train_data_list = total_train_data_list[step * slice_size : step * slice_size + slice_size_queue[step]]

                start = time.time()
                train_data, train_labels = ddl.load_in_array(data_list=train_data_list,
                                                             heads=2,
                                                             data_type='images')
                print('Time loading training data: %0.3f seconds' % (time.time() - start))
                # let validation happen every x steps
                if step % validation_interval == 0:
                    model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                              batch_size=pc.BATCH_SIZE,
                              epochs=1,
                              validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                              verbose=0)
                else:
                    model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                              batch_size=pc.BATCH_SIZE,
                              epochs=1,
                              verbose=0)



    elif similarity_metric == 'euclid':
        # rms = RMSprop()
        # model.compile(loss=contrastive_loss, optimizer=rms)
        # model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
        #           batch_size=pc.BATCH_SIZE,
        #           epochs=pc.NUM_EPOCHS,
        #           validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))
        pass


    te_pred = model.predict([test_data[:, 0], test_data[:, 1]])
    te_matrix = confusion_matrix('Testing', te_pred, test_labels)

    # delete objects else we run out of memory
    del model
    accuracy = (te_matrix[0] + te_matrix[2]) * 1.0 / (sum(te_matrix) * 1.0)
    print('accuracy = %0.2f, confusion matrix = %s' %(accuracy, str(te_matrix)))
    return te_matrix


def super_main(experiment_name, iterations, numfil, weights_name):
    accs = np.zeros((iterations, 4))

    start = time.time()
    for iter in range(0, iterations):
        print('-----ITERATION %d' % iter)
        accs[iter] = main(experiment_name, weights_name, numfil)
    stop = time.time()

    total_time = stop - start

    print('\nTP    FP    TN    FN')
    print(accs)

    print('mean values:')
    mean = np.mean(accs, axis=0)
    print(mean)

    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        dataset_name = 'VIPeR, CUHK1'
        pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name, total_time)


