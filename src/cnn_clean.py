import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D
from keras import optimizers
import project_constants as pc
import project_utils as pu

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/humans/1/'
# LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/humans/0/'

[train_data, train_labels, validation_data_, validation_labels_] = pu.load_human_detection_data()
train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
validation_labels_ = keras.utils.to_categorical(validation_labels_, pc.NUM_CLASSES)

print(str(train_data.shape))

test_data = validation_data_[len(validation_data_) / 2:len(validation_data_)]
test_labels = validation_labels_[len(validation_labels_) / 2:len(validation_labels_)]

validation_data = validation_data_[0:len(validation_data_) / 2]
validation_labels = validation_labels_[0:len(validation_labels_) / 2]

# reshape data to be processed through 1D conv
# train_data = np.reshape(train_data, (len(train_data), pc.IMAGE_HEIGHT*pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
# test_data = np.reshape(test_data, (len(test_data), pc.IMAGE_HEIGHT*pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
# validation_data = np.reshape(validation_data, (len(validation_data), pc.IMAGE_HEIGHT*pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def cnn_model():
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


def cnn_model_2d_conv_1d_filters():
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


def main():
    # model = cnn_model()
    model = cnn_model_2d_conv_1d_filters()
    if pc.VERBOSE:
        print(model.summary())

    nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=pc.BATCH_SIZE, epochs=pc.NUM_EPOCHS,
              validation_data=(validation_data, validation_labels))

    score = model.evaluate(test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save model
    if pc.SAVE_CNN:
        model.save('cnn_model_1D_filters_1-2.h5')
        model.save_weights('cnn_model_weights_1D_filters_1-2.h5')

    # if pc.LOGGING:
    #     predictions = model.predict(test_data, test_labels)
    #     mean = pu.make_confusion_matrix(predictions, test_labels)
    #     iterations = 1
    #     file_name = os.path.basename(__file__)
    #     experiment_name = 'trying out 1D filters'
    #     dataset_name = 'INRIA'
    #     pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name)

main()

