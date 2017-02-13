import tensorflow as tf
import numpy as np
import time
import sys
import argparse
import os
import random as rd
from PIL import Image
from scipy import ndimage
import project_constants as pc


def create_weights_biases(model_parameters, )


def create_cnn(data, model_parameters, stream_num):
    # should look like this, we comment this out in the end
    stream = data[stream_num]

    # should look like this, we comment this out in the end
    model_parameters = {
        'num_epochs': 100,
        'batch_size': 1,
        'num_tracklets': 100,
        'num_frames': 3,
        'image_height': 200,
        'image_width': 100,
        'num_channels': 3,
        'num_labels': 2,
        'start_learning_rate': 0.0001

    }

    # TODO: figure out the details of the input
    # define placeholders
    stream_node = tf.placeholder(
        pc.DATA_TYPE,
        shape=(model_parameters['batch_size'], model_parameters['num_frames'], model_parameters['image_height'],
               model_parameters['image_width'], model_parameters['num_channels'])
    )

    # define weights using Xavier initialization
    conv_1_weights = tf.get_variable('conv_1_weights', shape=(3, 3, model_parameters['num_channels'], 16),
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=pc.DATA_TYPE, trainable=True)
    conv_2_weights = tf.get_variable('conv_2_weights', shape=(3, 3, 16, 32),
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=pc.DATA_TYPE, trainable=True)
    conv_3_weights = tf.get_variable('conv_3_weights', shape=(3, 3, 32, 64),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=pc.DATA_TYPE, trainable=True)
    conv_4_weights = tf.get_variable('conv_4_weights', shape=(3, 3, 64, 128),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=pc.DATA_TYPE, trainable=True)
    conv_5_weights = tf.get_variable('conv_5_weights', shape=(3, 3, 128, 256),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=pc.DATA_TYPE, trainable=True)

    # define biases with value of zero
    conv_1_biases = tf.Variable(tf.zeros([16], dtype=pc.DATA_TYPE))
    conv_2_biases = tf.Variable(tf.zeros([32], dtype=pc.DATA_TYPE))
    conv_3_biases = tf.Variable(tf.zeros([64], dtype=pc.DATA_TYPE))
    conv_4_biases = tf.Variable(tf.zeros([128], dtype=pc.DATA_TYPE))
    conv_5_biases = tf.Variable(tf.zeros([256], dtype=pc.DATA_TYPE))

    # define fully connected weights
    fc_1_weights = tf.get_variable('fc_1_weights', shape=(256, 512),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=pc.DATA_TYPE, trainable=True)
    fc_2_weights = tf.get_variable('fc_2_weights', shape=(512, model_parameters['num_labels']),
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   dtype=pc.DATA_TYPE, trainable=True)

    # define fully connected biases
    fc_1_biases = tf.Variable(tf.zeros([512], dtype=pc.DATA_TYPE))
    fc_2_biases = tf.Variable(tf.zeros([model_parameters['num_labels']], dtype=pc.DATA_TYPE))

    # define model architecture


    pass


def main():

    pass