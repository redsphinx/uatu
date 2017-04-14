import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D
from keras import optimizers
import project_constants as pc
import project_utils as pu
import time
import os
import numpy as np


def initialize():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.load_inria_nicta()

    train_data = np.asarray(train_data)
    validation_data = np.asarray(validation_data)
    test_data = np.asarray(test_data)

    train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
    validation_labels = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)
    print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))
    return [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]


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

    model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))
    model.add(Dense(512))
    model.add(Dense(pc.NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def cnn_model_2d_conv_1d_filters(train_data):
    # figure out how the number of filters change
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=train_data.shape[1:], name='conv_1_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same', name='conv_1_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', name='conv_2_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', name='conv_2_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(64, kernel_size=(1, 3), padding='same', name='conv_3_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same', name='conv_3_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(128, kernel_size=(1, 3), padding='same', name='conv_4_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 1), padding='same', name='conv_4_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(256, kernel_size=(1, 3), padding='same', name='conv_5_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 1), padding='same', name='conv_5_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(512, kernel_size=(1, 3), padding='same', name='conv_6_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 1), padding='same', name='conv_6_2'))
    model = add_activation_and_relu(model)

    model.add(Conv2D(1024, kernel_size=(1, 3), padding='same', name='conv_7_1'))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(3, 1), padding='same', name='conv_7_2'))
    model.add(Activation('relu'))

    model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))
    model.add(Dense(512))
    model.add(Dense(pc.NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def main(experiment_name):
    # model = cnn_model()
    start = time.time()
    [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = initialize()

    model = cnn_model_2d_conv_1d_filters(train_data)
    if pc.VERBOSE:
        print(model.summary())

    nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    # TODO implement dynamic loading of data
    '''

    total_data = len(train_labels) / batch_size
    for epoch in range(pc.NUM_EPOCHS):
        for step in range(0, total_data):
            [train_images_1, train_images_2, train_labels] = pu.generate_data_batch_siamese(
            train_data_list, step, batch_size)
            model.fit()
    '''

    model.fit(train_data,
              train_labels,
              batch_size=pc.BATCH_SIZE,
              epochs=pc.NUM_EPOCHS,
              validation_data=(validation_data, validation_labels),
              verbose=2)
    stop = time.time()
    score = model.evaluate(test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # save model
    if pc.SAVE_CNN:
        name_weights = 'cnn_model_weights_1D_filters_1-2_extra.h5'
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, name_weights))

    if pc.LOGGING:
        iterations = 1
        test_confusion_matrix = pu.make_confusion_matrix(model.predict(test_data), test_labels)
        file_name = os.path.basename(__file__)
        # experiment_name = 'CNN weights trained with more data'
        dataset_name = 'INRIA, NICTA'
        pu.enter_in_log(experiment_name, file_name, iterations, test_confusion_matrix, dataset_name, stop-start)

# main()

