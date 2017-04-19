import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras import optimizers
import project_constants as pc
import project_utils as pu
from clr_callback import *

import time
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def cnn_model(train_data):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=train_data.shape[1:], name='conv_1'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_2'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_3'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_5'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(1024, kernel_size=(3, 3), padding='same', name='conv_6'))
    model = add_activation_and_relu(model)
    model.add(Conv2D(2048, kernel_size=(3, 3), padding='same', name='conv_7'))
    model.add(Activation('relu'))

    # model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))
    model.add(Dense(512))
    model.add(Dense(pc.NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def cnn_model_2d_conv_1d_filters(train_data):
    model = Sequential()
    model.add(Conv2D(16*2, kernel_size=(1, 3), padding='same', input_shape=train_data.shape[1:], name='conv_1_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(16*2, kernel_size=(3, 1), padding='same', name='conv_1_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(32*2, kernel_size=(1, 3), padding='same', name='conv_2_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32*2, kernel_size=(3, 1), padding='same', name='conv_2_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(64*2, kernel_size=(1, 3), padding='same', name='conv_3_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(64*2, kernel_size=(3, 1), padding='same', name='conv_3_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(128*2, kernel_size=(1, 3), padding='same', name='conv_4_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(128*2, kernel_size=(3, 1), padding='same', name='conv_4_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(256*2, kernel_size=(1, 3), padding='same', name='conv_5_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(256*2, kernel_size=(3, 1), padding='same', name='conv_5_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(512*2, kernel_size=(1, 3), padding='same', name='conv_6_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(512*2, kernel_size=(3, 1), padding='same', name='conv_6_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(1024*2, kernel_size=(1, 3), padding='same', name='conv_7_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(1024*2, kernel_size=(3, 1), padding='same', name='conv_7_2'))
    model.add(Activation('relu'))

    model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))
    model.add(Dense(512))
    model.add(Dense(pc.NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def cnn_model_2d_conv_1d_filters_BN(train_data, do_dropout):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=train_data.shape[1:], name='conv_1_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_1'))
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same', name='conv_1_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_2'))

    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', name='conv_2_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_3'))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', name='conv_2_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_4'))

    model.add(Conv2D(64, kernel_size=(1, 3), padding='same', name='conv_3_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_5'))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same', name='conv_3_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_6'))

    model.add(Conv2D(128, kernel_size=(1, 3), padding='same', name='conv_4_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_7'))
    model.add(Conv2D(128, kernel_size=(3, 1), padding='same', name='conv_4_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_8'))

    model.add(Conv2D(256, kernel_size=(1, 3), padding='same', name='conv_5_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_9'))
    model.add(Conv2D(256, kernel_size=(3, 1), padding='same', name='conv_5_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_10'))

    model.add(Conv2D(512, kernel_size=(1, 3), padding='same', name='conv_6_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_11'))
    model.add(Conv2D(512, kernel_size=(3, 1), padding='same', name='conv_6_2', use_bias=pc.USE_BIAS))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_12'))

    model.add(Conv2D(1024, kernel_size=(1, 3), padding='same', name='conv_7_1', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_13'))
    model.add(Conv2D(1024, kernel_size=(3, 1), padding='same', name='conv_7_2', use_bias=pc.USE_BIAS))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_14'))

    if do_dropout:
        model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))
    model.add(Dense(512, use_bias=pc.USE_BIAS))
    model.add(BatchNormalization(name='bn_15'))
    model.add(Dense(pc.NUM_CLASSES, use_bias=pc.USE_BIAS))
    model.add(Activation('softmax'))

    return model


def main(experiment_name, data):
    [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = data

    # model = cnn_model_2d_conv_1d_filters(train_data, do_dropout)
    model = cnn_model_2d_conv_1d_filters(train_data)

    if pc.VERBOSE:
        print(model.summary())

    nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    # TODO implement dynamic loading of data


    # create a validation and test set outside of the epochs
    # for epoch in range(pc.NUM_EPOCHS):
    #     train_data_batch, train_label_batch, validation_data_batch, validation_label_batch = pu.generate_batches(pc.BATCH_SIZE)
    #     model.fit(train_data_batch,
    #               train_label_batch,
    #               batch_size=pc.BATCH_SIZE,
    #               epochs=1,
    #               validation_data=(validation_data_batch, validation_label_batch))
    #


    # clr_triangular = CyclicLR(mode='exp_range', step_size=(np.shape(train_data)[0]/pc.BATCH_SIZE)*8)

    model.fit(train_data,
              train_labels,
              batch_size=pc.BATCH_SIZE,
              epochs=pc.NUM_EPOCHS,
              validation_data=(validation_data, validation_labels),
              verbose=2)
    score = model.evaluate(test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_confusion_matrix = pu.make_confusion_matrix(model.predict(test_data), test_labels)

    # save model
    # TODO change the saved weights names !!
    if pc.SAVE_CNN_WEIGHTS:
        shit_name = 'cnn_model_weights_simple_1D_filters_32.h5'
        name_weights = shit_name
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_weights))

    if pc.SAVE_CNN_MODEL:
        shit_name = 'cnn_model_simple_1D_filters_32.h5'
        name_model = shit_name
        model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_model))

    del model
    return test_confusion_matrix


def super_main(experiment_name, data, iterations):
    accs = np.zeros((iterations, 4))

    start = time.time()
    for iter in range(0, iterations):
        accs[iter] = main(experiment_name, data)
    stop = time.time()

    total_time = stop - start

    print('mean values:')
    mean = np.mean(accs, axis=0)
    print(mean)

    # TODO: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        dataset_name = 'INRIA, NICTA'
        pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name, total_time)

