import keras
from keras.models import sequential
from keras.layers import dense, dropout, activation, conv2d, maxpool2d, flatten
from keras import optimizers
import project_constants as pc
import project_utils as pu

import os

os.environ["cuda_visible_devices"]="0"

# load INRIA data
[train_data, train_labels, validation_data_, validation_labels_] = pu.load_human_detection_data()
train_labels = keras.utils.to_categorical(train_labels, pc.num_classes)
validation_labels_ = keras.utils.to_categorical(validation_labels_, pc.num_classes)

print(str(train_data.shape))

test_data = validation_data_[len(validation_data_) / 2:len(validation_data_)]
test_labels = validation_labels_[len(validation_labels_) / 2:len(validation_labels_)]

validation_data = validation_data_[0:len(validation_data_) / 2]
validation_labels = validation_labels_[0:len(validation_labels_) / 2]

print('train: %d, validation: %d, test: %d' % (len(train_data), len(validation_data), len(test_data)))


def add_activation_and_relu(model):
    model.add(activation('relu'))
    model.add(maxpool2d(pool_size=(2, 2)))
    return model


def cnn_model():
    model = sequential()
    model.add(conv2d(32, kernel_size=(3, 3), padding='same', input_shape=train_data.shape[1:], name='conv_1'))
    model = add_activation_and_relu(model)
    model.add(conv2d(64, kernel_size=(3, 3), padding='same', name='conv_2'))
    model = add_activation_and_relu(model)
    model.add(conv2d(128, kernel_size=(3, 3), padding='same', name='conv_3'))
    model = add_activation_and_relu(model)
    model.add(conv2d(256, kernel_size=(3, 3), padding='same', name='conv_4'))
    model = add_activation_and_relu(model)
    model.add(conv2d(512, kernel_size=(3, 3), padding='same', name='conv_5'))
    model = add_activation_and_relu(model)
    model.add(conv2d(1024, kernel_size=(3, 3), padding='same', name='conv_6'))
    model = add_activation_and_relu(model)

    model.add(dropout(pc.dropout, name='cnn_drop'))

    model.add(flatten(name='cnn_flat'))


    model.add(dense(512))
    model.add(dense(pc.num_classes))

    model.add(activation('softmax'))

    return model

def main():
    model = cnn_model()
    if pc.verbose:
        print(model.summary())

    nadam = optimizers.nadam(lr=pc.start_learning_rate, schedule_decay=pc.decay_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=pc.batch_size, epochs=pc.num_epochs,
              validation_data=(validation_data, validation_labels))

    score = model.evaluate(test_data, test_labels)
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    # save model
    model.save('cnn_model.h5')
    model.save_weights('cnn_model_weights.h5')

main()

