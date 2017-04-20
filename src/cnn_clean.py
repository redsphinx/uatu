import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras import optimizers
import project_constants as pc
import project_utils as pu
import dynamic_data_loading as ddl
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


def cnn_model_2d_conv_1d_filters():
    numfil = 1
    model = Sequential()
    model.add(Conv2D(16*numfil, kernel_size=(1, 3), padding='same', input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH,
                                                                                 pc.NUM_CHANNELS), name='conv_1_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(16*numfil, kernel_size=(3, 1), padding='same', name='conv_1_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(32*numfil, kernel_size=(1, 3), padding='same', name='conv_2_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32*numfil, kernel_size=(3, 1), padding='same', name='conv_2_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(64*numfil, kernel_size=(1, 3), padding='same', name='conv_3_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(64*numfil, kernel_size=(3, 1), padding='same', name='conv_3_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(128*numfil, kernel_size=(1, 3), padding='same', name='conv_4_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(128*numfil, kernel_size=(3, 1), padding='same', name='conv_4_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(256*numfil, kernel_size=(1, 3), padding='same', name='conv_5_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(256*numfil, kernel_size=(3, 1), padding='same', name='conv_5_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(512*numfil, kernel_size=(1, 3), padding='same', name='conv_6_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(512*numfil, kernel_size=(3, 1), padding='same', name='conv_6_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(1024*numfil, kernel_size=(1, 3), padding='same', name='conv_7_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(1024*numfil, kernel_size=(3, 1), padding='same', name='conv_7_2'))
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


def main(experiment_name, weights_name):
    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = data
    total_data_list_pos, total_data_list_neg = pu.merge_pedestrian_sets()
    val_list, test_list, total_data_list_pos, total_data_list_neg = ddl.make_validation_test_list(total_data_list_pos,
                                                                                                  total_data_list_neg,
                                                                                                  val_pos_percent=0.5,
                                                                                                  test_pos_percent=0.5)
    model = cnn_model_2d_conv_1d_filters()

    if pc.VERBOSE:
        print(model.summary())

    nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    batch_size = pc.BATCH_SIZE * 20
    train_data_size = 2*min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps = np.ceil(train_data_size*1.0 / batch_size).astype(int)

    # in each epoch the training data gets partitioned into different batches. This way we break correlations between
    # instances and we can see many more negative examples
    val_data, val_labels = ddl.load_in_array(val_list)
    for epoch in range(pc.NUM_EPOCHS):
        batch_size_queue = ddl.make_batch_queue(train_data_size, batch_size)
        total_train_data_list = ddl.make_train_batches(total_data_list_pos, total_data_list_neg)
        for step in range(num_steps):
            start = time.time()
            train_data_list = total_train_data_list[step * batch_size : step * batch_size + batch_size_queue[step]]
            train_data, train_labels = ddl.load_in_array(train_data_list)

            model.fit(train_data,
                      train_labels,
                      # batch_size=batch_size_queue[step],
                      batch_size=pc.BATCH_SIZE*2,
                      epochs=1,
                      validation_data=(val_data, val_labels),
                      verbose=2)
            stop = time.time()
            the_time = stop - start
            total_steps = pc.NUM_EPOCHS * num_steps
            have_been = (epoch * num_steps) + step
            remaining = (total_steps - have_been) * the_time
            print('Epoch: %d / %d   Step: %d / %d  time: %0.2f s  remaining: %0.2f s' % (epoch, pc.NUM_EPOCHS, step,
                                                                                       int(num_steps), the_time,
                                                                                       remaining))
    # clr_triangular = CyclicLR(mode='exp_range', step_size=(np.shape(train_data)[0]/pc.BATCH_SIZE)*8)

    # model.fit(train_data,
    #           train_labels,
    #           batch_size=pc.BATCH_SIZE,
    #           epochs=pc.NUM_EPOCHS,
    #           validation_data=(validation_data, validation_labels),
    #           verbose=2)

    test_data, test_labels = ddl.load_in_array(test_list)
    score = model.evaluate(test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_confusion_matrix = pu.make_confusion_matrix(model.predict(test_data), test_labels)

    # save model
    # TODO change the saved weights names !!
    if pc.SAVE_CNN_WEIGHTS:
        shit_name = weights_name
        name_weights = shit_name
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_weights))

    if pc.SAVE_CNN_MODEL:
        shit_name = 'shit.h5'
        name_model = shit_name
        model.save(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_model))

    del model
    return test_confusion_matrix


def super_main(experiment_name, iterations, weights_name):
    accs = np.zeros((iterations, 4))

    start = time.time()
    for iter in range(0, iterations):
        accs[iter] = main(experiment_name, weights_name)
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

