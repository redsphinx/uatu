import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda
from keras.optimizers import RMSprop
from keras import optimizers

from keras import backend as K

import project_constants as pc
import project_utils as pu
import os
import numpy as np
from keras.utils import plot_model


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# load the data
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = pu.load_viper()

train_labels = train_labels.astype(np.int64)
validation_labels = validation_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)

train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
validation_labels = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)

# to use as the input shape later on
train_data_ = train_data[:, 0, ...]


def alt_create_fc(inputs):
    dense_layer = Dense(512, activation='relu') (inputs)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    dense_layer = Dense(1024, activation='relu')(dropout_layer)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    output_layer = Dense(pc.NUM_CLASSES) (dropout_layer)
    softmax = Activation('softmax')(output_layer)
    return  softmax


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def create_base_network():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_3', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_5', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(1024, kernel_size=(3, 3), padding='same', name='conv_6', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(2048, kernel_size=(3, 3), padding='same', name='conv_7', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))

    # model.add(Dropout(pc.DROPOUT, name='cnn_drop'))
    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights('cnn_model_weights_7.h5', by_name=True)

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


def create_siamese():
    input_a = Input(shape=(train_data_.shape[1:]))
    input_b = Input(shape=(train_data_.shape[1:]))
    base_network = create_base_network()

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    if pc.SIMILARITY_METRIC == 'fc_layers':
        # use a fc to come up with a metric
        merged_processed = keras.layers.concatenate([processed_a, processed_b])
        distance = alt_create_fc(merged_processed)
        model = Model([input_a, input_b], distance)
    elif pc.SIMILARITY_METRIC == 'euclid':
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


def main():
    model = create_siamese()

    if pc.SIMILARITY_METRIC == 'fc_layers':
        nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                  batch_size=pc.BATCH_SIZE,
                  epochs=pc.NUM_EPOCHS,
                  validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))
    elif pc.SIMILARITY_METRIC == 'euclid':
        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)
        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                  batch_size=pc.BATCH_SIZE,
                  epochs=pc.NUM_EPOCHS,
                  validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))

    tr_pred = model.predict([train_data[:, 0], train_data[:, 1]])
    tr_matrix = confusion_matrix('Training', tr_pred, train_labels)

    va_pred = model.predict([validation_data[:, 0], validation_data[:, 1]])
    va_matrix = confusion_matrix('Validation', va_pred, validation_labels)

    te_pred = model.predict([test_data[:, 0], test_data[:, 1]])
    te_matrix = confusion_matrix('Testing', te_pred, test_labels)

    return (tr_matrix, va_matrix, te_matrix)

def super_main():
    iterations = 10
    accs = np.zeros((iterations, 3, 4))

    for iter in range(0, iterations):
        accs[iter] = main()

    test_mat = np.zeros((iterations, 4))

    print('\nTP    FP    TN    FN')
    for item in range(0, len(accs)):
        test_mat[item] = accs[item, 2]
        print(test_mat[item])

    print('mean values:')
    mean = np.mean(test_mat, axis=0)
    print(mean)

    # TODO: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        experiment_name = 'does freezing cnn layers help'
        dataset_name = 'VIPeR'
        pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name)


super_main()
