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
import project_utils as pu
import siamese_cnn as scnn


def make_model(data, name):
    # make cnn part
    cnn_part = scnn.build_cnn(data, name)

    # define fc layers
    fc_1_weights = tf.get_variable(
        'fc_1_weights',
        shape=[256, 512],
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    fc_2_weights = tf.get_variable(
        'fc_2_weights',
        shape=[512, pc.NUM_CLASSES],
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )
    fc_1_biases = tf.get_variable(
        'fc_1_bias',
        shape=[512],
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    fc_2_biases = tf.get_variable(
        'fc_2_bias',
        shape=[pc.NUM_CLASSES],
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    # TODO figure out the shape of cnn part
    fc_1 = tf.nn.relu(tf.matmul(cnn_part, fc_1_weights) + fc_1_biases)
    return tf.matmul(fc_1, fc_2_weights) + fc_2_biases


def main():
    train_node_1 = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    train_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.NUM_CLASSES])

    validation_node_1 = tf.placeholder(pc.DATA_TYPE,
                                       shape=[pc.EVAL_BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    validation_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.EVAL_BATCH_SIZE, pc.NUM_CLASSES])

    logits = make_model(train_node_1, 'cnn_model')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_label_node))

    # TODO: maybe put regularizers and dropout

    step = tf.Variable(0, dtype=pc.DATA_TYPE)
    learning_rate = tf.train.exponential_decay(
        pc.START_LEARNING_RATE,
        step * pc.BATCH_SIZE,
        pc.DECAY_STEP,
        pc.DECAY_RATE,
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, pc.MOMENTUM)
    optimizer = optimizer.minimize(loss, global_step=step)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    # scope.reuse_variables()
    validation_prediction = tf.nn.softmax(make_model(validation_node_1, 'validate'))

    # running everything
    if tf.__version__ == '0.10.0':
        init = tf.initialize_all_variables()
    elif tf.__version__ == '0.12.0' or tf.__version__ == '0.12.1':
        init = tf.global_variables_initializer()
    else:
        print('version not supported. add what to do manually.')
        init = tf.global_variables_initializer()
        print(tf.__version__)

    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        data = np.asarray(data)

        size = np.shape(data)[0]

        if size < pc.EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, pc.NUM_CLASSES), dtype=np.float32)
        for begin in range(0, size, pc.EVAL_BATCH_SIZE):
            end = begin + pc.EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    validation_prediction,
                    feed_dict={validation_node_1: data[begin:end, 0, ...]
                               }
                )
                # feed_dict={validation_data: data[begin:end]})
            else:
                batch_predictions = sess.run(
                    validation_prediction,
                    feed_dict={validation_data: data[-pc.EVAL_BATCH_SIZE:]})
                predictions[begin:, :] = batch_predictions[begin - size:]
        return predictions

    train_data = pu.load_human_detection_data('training_data')
    train_labels = pu.load_human_detection_data('training_labels')
    validation_data = pu.load_human_detection_data('validation_data')
    validation_labels = pu.load_human_detection_data('validation_labels')
    test_data = pu.load_human_detection_data('test_data')
    test_labels = pu.load_human_detection_data('test_labels')

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(init)
        print('Initialized!')

        for train_step in range(0, pc.NUM_EPOCHS):
            offset = (train_step * pc.BATCH_SIZE) % (pc.NUM_TRAIN - pc.BATCH_SIZE)
            batch_data = train_data[offset:(offset + pc.BATCH_SIZE)]
            batch_labels = train_labels[offset:(offset + pc.BATCH_SIZE)]

            train_batch_1 = batch_data[:, 0, ...]

            feed_dict = {
                train_node_1: train_batch_1,
                train_label_node: batch_labels}

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if train_step % pc.EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (train_step, float(train_step) * pc.BATCH_SIZE / pc.NUM_TRAIN,
                       1000 * elapsed_time / pc.EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % pu.error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % pu.error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
                # Finally print the result!
                # TODO: make a test set
                # test_error = error_rate(eval_in_batches(validation_data, sess), validation_labels)
                # print('Test error: %.1f%%' % test_error)