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

# train_labels = map(int, train_labels)
# validation_labels = map(int, validation_labels)
# test_labels = map(int, test_labels)

train_labels = train_labels.astype(np.int64)
validation_labels = validation_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)



# train_labels = pu.flip_labels(train_labels)
# validation_labels = pu.flip_labels(validation_labels)
# test_labels = pu.flip_labels(test_labels)

train_labels = keras.utils.to_categorical(train_labels, pc.NUM_CLASSES)
validation_labels = keras.utils.to_categorical(validation_labels, pc.NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, pc.NUM_CLASSES)

# for some reason it solves TypeError: cannot perform reduce with flexible type
# validation_labels = validation_labels.astype(np.float)

# input_dim = 2048
# to get the correct shape in the CNN
train_data_ = train_data[:, 0, ...]





def alt_create_fc(inputs):

    # Layer dense_2 expects 1 inputs, but it received 2 input tensors. Input received: <built-in function input>
    # inputs = Input(shape=)

    dense_layer = Dense(512, activation='relu') (inputs)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    dense_layer = Dense(1024, activation='relu')(dropout_layer)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    dense_layer = Dense(2048, activation='relu')(dropout_layer)
    # dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    # dense_layer = Dense(4096, activation='relu')(dropout_layer)
    # dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    # dense_layer = Dense(8192, activation='relu')(dropout_layer)
    dropout_layer = Dropout(pc.DROPOUT)(dense_layer)
    output_layer = Dense(pc.NUM_CLASSES) (dropout_layer)
    softmax = Activation('softmax')(output_layer)

    return  softmax


def create_fc(inputs):
    seq = Sequential()
    seq.add(Dense(512, input_shape=(pc.NUM_CAMERAS, ), activation='relu'))
    seq.add(Dropout(pc.DROPOUT))
    seq.add(Dense(1))

    # seq.add(Activation('softmax'))
    return seq


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
    # plot_model(model, to_file='model.png')

    return model


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()



def threshold_predictions(predictions):
    new_predictions = np.zeros((len(predictions), 2))
    for item in range(0, len(predictions)):
        new_predictions[item, np.argmax(predictions[item])] = 1

    return new_predictions


def calculate_accuracy(predictions, labels):
    predictions = threshold_predictions(predictions)
    good = 0.0
    total = len(predictions) * 1.0

    if len(np.shape(labels)) > 1:
        for pred in range(0, len(predictions)):
            a = predictions[pred][0]
            b = labels[pred][0]
            if predictions[pred][0] == labels[pred][0]:
                good += 1
    else:
        for pred in range(0, len(predictions)):
            a = predictions[pred]
            b = labels[pred]
            if predictions[pred] == labels[pred]:
                good += 1

    acc = good / total
    return acc



def main():

    model = create_siamese()

    # train
    rms = RMSprop()
    nadam = optimizers.Nadam(lr=pc.START_LEARNING_RATE, schedule_decay=pc.DECAY_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
              batch_size=pc.BATCH_SIZE,
              epochs=pc.NUM_EPOCHS,
              validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))

    # prediction = model.predict([test_data[:,0], test_data[:,1]], batch_size=32, verbose=1)
    # print(prediction[0:20])

    # accuracy = calculate_accuracy(prediction, test_labels)
    # print('accuracy: %f' %accuracy)
    # score = model.evaluate([test_data[:,0], test_data[:,1]], test_labels)
    # print(score)

    pred = model.predict([train_data[:, 0], train_data[:, 1]])
    tr_acc = calculate_accuracy(pred, train_labels)
    pred = model.predict([validation_data[:, 0], validation_data[:, 1]])
    va_acc = calculate_accuracy(pred, validation_labels)
    pred = model.predict([test_data[:, 0], test_data[:, 1]])
    te_acc = calculate_accuracy(pred, test_labels)
    print(pred[0:20])


    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on validation set: %0.2f%%' % (100 * va_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

main()