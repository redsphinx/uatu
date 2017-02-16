import tensorflow as tf
import numpy as np
import time
import sys
import argparse
import os
import random as rd
from PIL import Image
from scipy import ndimage
from tensorflow.contrib.layers import flatten


import project_constants as pc
import project_utils as pu
from siamese_cnn import build_cnn

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():
    # define weights and biases
    
    conv1_weights = tf.Variable(
        tf.truncated_normal(pc.KERNEL_SHAPE_1,  # 5x5 filter, depth 32.
                             dtype=pc.DATA_TYPE))
    
    
    conv_1_weights = tf.Variable(
        tf.truncated_normal(
            shape=pc.KERNEL_SHAPE_1,
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE)
    )

    conv_2_weights = tf.Variable(
        tf.truncated_normal(
            shape=pc.KERNEL_SHAPE_2,
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE)
    )

    conv_3_weights = tf.Variable(
        tf.truncated_normal(
            shape=pc.KERNEL_SHAPE_3,
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE)
    )

    conv_4_weights = tf.Variable(
        tf.truncated_normal(
            shape=pc.KERNEL_SHAPE_4,
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE)
    )

    conv_5_weights = tf.Variable(
        tf.truncated_normal(
            shape=pc.KERNEL_SHAPE_5,
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE)
    )

    conv_1_biases = tf.Variable(tf.zeros(shape=pc.BIAS_SHAPE_1, dtype=pc.DATA_TYPE))
    conv_2_biases = tf.Variable(tf.zeros(shape=pc.BIAS_SHAPE_2, dtype=pc.DATA_TYPE))
    conv_3_biases = tf.Variable(tf.zeros(shape=pc.BIAS_SHAPE_3, dtype=pc.DATA_TYPE))
    conv_4_biases = tf.Variable(tf.zeros(shape=pc.BIAS_SHAPE_4, dtype=pc.DATA_TYPE))
    conv_5_biases = tf.Variable(tf.zeros(shape=pc.BIAS_SHAPE_5, dtype=pc.DATA_TYPE))


    # define fc layers

    fc_1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [2048, 512],
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE))


    fc_2_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [512, pc.NUM_CLASSES],
            stddev=0.1,
            seed=pc.SEED,
            dtype=pc.DATA_TYPE))

    fc_1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=pc.DATA_TYPE))
    fc_2_biases = tf.Variable(tf.constant(0.1, shape=[pc.NUM_CLASSES], dtype=pc.DATA_TYPE))

    
    def conv_unit(inp, weights, biases):
        conv = tf.nn.conv2d(
            inp,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
        pool = tf.nn.max_pool(
            relu,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        return pool



    def make_model(data, name):
        # define model
        conv_1_layer = conv_unit(data, conv_1_weights, conv_1_biases)
        conv_2_layer = conv_unit(conv_1_layer, conv_2_weights, conv_2_biases)
        conv_3_layer = conv_unit(conv_2_layer, conv_3_weights, conv_3_biases)
        conv_4_layer = conv_unit(conv_3_layer, conv_4_weights, conv_4_biases)
        conv_5_layer = conv_unit(conv_4_layer, conv_5_weights, conv_5_biases)

        # cnn_part = flatten(conv_5_layer)
        pool_shape = conv_5_layer.get_shape().as_list()
        reshape = tf.reshape(
            conv_5_layer,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])


        fc_1 = tf.nn.relu(tf.matmul(reshape, fc_1_weights) + fc_1_biases)
        return tf.matmul(fc_1, fc_2_weights) + fc_2_biases


    train_node_1 = tf.placeholder(pc.DATA_TYPE,
                                  shape=[pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])

    validation_node_1 = tf.placeholder(pc.DATA_TYPE,
                                       shape=[pc.EVAL_BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])

    logits = make_model(train_node_1, 'cnn_model')
    # Predictions for the test and validation, which we'll compute less often.
    validation_prediction = tf.nn.softmax(make_model(validation_node_1, 'validate'))

    [train_data, train_labels, validation_data, validation_labels] = pu.load_human_detection_data()

    train_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.NUM_CLASSES])


    validation_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.EVAL_BATCH_SIZE, pc.NUM_CLASSES])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_label_node))
    regularizers = (tf.nn.l2_loss(fc_1_weights) + tf.nn.l2_loss(fc_1_biases) +
                    tf.nn.l2_loss(fc_2_weights) + tf.nn.l2_loss(fc_2_biases))
    loss += 5e-4 * regularizers

    # TODO: maybe put regularizers and dropout

    step = tf.Variable(0, dtype=pc.DATA_TYPE)
    # learning_rate = tf.train.exponential_decay(
    #     pc.START_LEARNING_RATE,
    #     step * pc.BATCH_SIZE,
    #     len(train_data),
    #     pc.DECAY_RATE,
    #     staircase=True)


    # Use simple momentum for the optimization.
    # optimizer = tf.train.MomentumOptimizer(learning_rate, pc.MOMENTUM)
    # optimizer = optimizer.minimize(loss, global_step=step)

    optimizer = tf.train.AdamOptimizer(pc.START_LEARNING_RATE)
    optimizer = optimizer.minimize(loss)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)



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
                    feed_dict={validation_node_1: data[begin:end]
                               }
                )
                # feed_dict={validation_data: data[begin:end]})
            else:
                batch_predictions = sess.run(
                    validation_prediction,
                    feed_dict={validation_node_1: data[-pc.EVAL_BATCH_SIZE:]})
                predictions[begin:, :] = batch_predictions[begin - size:]
        return predictions


    start_time = time.time()
    with tf.Session() as sess:
        sess.run(init)
        print('Initialized!')

        for train_step in range(0, pc.NUM_EPOCHS):
            offset = (train_step * pc.BATCH_SIZE) % (len(train_data) - pc.BATCH_SIZE)
            batch_data = train_data[offset:(offset + pc.BATCH_SIZE)]
            batch_labels = train_labels[offset:(offset + pc.BATCH_SIZE)]

            train_batch_1 = batch_data

            feed_dict = {
                train_node_1: train_batch_1,
                train_label_node: batch_labels}

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if train_step % pc.EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, predictions = sess.run([loss, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (train_step, float(train_step) * pc.BATCH_SIZE / len(train_data),
                       1000 * elapsed_time / pc.EVAL_FREQUENCY))
                print('Minibatch loss: %.3f' % l)
                print('Minibatch error: %.1f%%' % pu.error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % pu.error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
                # Finally print the result!
                # TODO: make a test set
                # test_error = error_rate(eval_in_batches(validation_data, sess), validation_labels)
                # print('Test error: %.1f%%' % test_error)


main()