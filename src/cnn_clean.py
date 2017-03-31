import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
import project_constants as pc
import project_utils as pu

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/humans/1/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/humans/0/'

[train_data, train_labels, validation_data_, validation_labels_] = pu.load_human_detection_data()
train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
validation_labels_ = keras.utils.to_categorical(validation_labels_, pc.NUM_CLASSES)

print(str(train_data.shape))

test_data = validation_data_[len(validation_data_) / 2:len(validation_data_)]
test_labels = validation_labels_[len(validation_labels_) / 2:len(validation_labels_)]

validation_data = validation_data_[0:len(validation_data_) / 2]
validation_labels = validation_labels_[0:len(validation_labels_) / 2]

print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def cnn_model():
    model = Sequential()
    # 1
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=train_data.shape[1:]))
    model = add_activation_and_relu(model)
    # 2
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model = add_activation_and_relu(model)
    # 3
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model = add_activation_and_relu(model)
    # 4
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model = add_activation_and_relu(model)
    # 5
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
    model = add_activation_and_relu(model)
    # 6
    model.add(Conv2D(1024, kernel_size=(3, 3), padding='same'))
    model = add_activation_and_relu(model)

    model.add(Dropout(pc.DROPOUT))

    model.add(Flatten())


    model.add(Dense(512))
    model.add(Dense(pc.NUM_CLASSES))

    model.add(Activation('softmax'))

    return model

def main():
    model = cnn_model()

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
    # model.save('cnn_model.h5')
    # model.save_weights('cnn_model_weights.h5')

main()