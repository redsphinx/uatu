import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

import project_constants as pc
import project_utils as pu
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# load the data
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = pu.load_viper()
train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)

# for some reason it solves TypeError: cannot perform reduce with flexible type
validation_labels = validation_labels.astype(np.float)

validation_labels_ = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)

#TODO fix input_dim
input_dim = 2048
# to get the correct shape in the CNN
train_data_ = train_data[:, 0, ...]


def alt_create_fc(inputs):

    # Layer dense_2 expects 1 inputs, but it received 2 input tensors. Input received: <built-in function input>
    # inputs = Input(shape=)

    dense_layer = Dense(512, activation='relu') (inputs)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    output_layer = Dense(pc.NUM_CLASSES) (dropout_layer)
    activations = Activation('softmax')(output_layer)
    return  activations


# def create_fc():
#     seq = Sequential()
#     seq.add(Dense(512, input_shape=(pc.NUM_CAMERAS, ), activation='relu'))
#     seq.add(Dropout(pc.DROPOUT))
#     seq.add(Dense(pc.NUM_CLASSES))
#     seq.add(Activation('softmax'))
#     return seq


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def alt_create_base_network():
    pass



def create_base_network():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1'))
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
    model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights('cnn_model_weights.h5', by_name=True)

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


# this is meaningless
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_siamese():
    input_a = Input(shape=(train_data_.shape[1:]))
    input_b = Input(shape=(train_data_.shape[1:]))
    base_network = create_base_network()

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    merged_processed = keras.layers.concatenate([processed_a, processed_b])

    # use a fc to come up with a metric
    distance = alt_create_fc(merged_processed)

    # useful to show drawback of euclidean distance as a metric
    # distance = Lambda(euclidean_distance,
    #                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    return model


def calculate_accuracy(predictions, labels):

    good = 0
    total = len(predictions)
    print(total)

    for pred in range(0, len(predictions)):
        a = predictions[pred][0]
        b = labels[pred][0]
        if predictions[pred][0] == labels[pred][0]:
            good += 1

    acc = good / total
    print(acc)
    return acc



def main():

    model = create_siamese()

    print('asdf')

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
              batch_size=pc.BATCH_SIZE,
              epochs=pc.NUM_EPOCHS,
              validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))

    # score = model.evaluate([test_data[:, 0], test_data[:, 1]], test_labels)
    # print(score)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on validation set: %0.2f%%' % (100 * va_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    prediction = model.predict_on_batch([test_data[:,0], test_data[:,0]])

    accuracy = calculate_accuracy(prediction, test_labels)
    print(accuracy)

main()