import tensorflow as tf
import numpy as np
from PIL import Image
import time
import sys

import project_constants as pc
import project_utils as pu

from tensorflow.contrib.layers import flatten

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def build_cnn(data, scope_name):
    # define weights and biases
    conv_1_weights = tf.get_variable(
        'weights_1',
        shape=pc.KERNEL_SHAPE_1,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    conv_2_weights = tf.get_variable(
        'weights_2',
        shape=pc.KERNEL_SHAPE_2,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_3_weights = tf.get_variable(
        'weights_3',
        shape=pc.KERNEL_SHAPE_3,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_4_weights = tf.get_variable(
        'weights_4',
        shape=pc.KERNEL_SHAPE_4,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_5_weights = tf.get_variable(
        'weights_5',
        shape=pc.KERNEL_SHAPE_5,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_1_biases = tf.get_variable(
        'biases_1',
        shape=pc.BIAS_SHAPE_1,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_2_biases = tf.get_variable(
        'biases_2',
        shape=pc.BIAS_SHAPE_2,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_3_biases = tf.get_variable(
        'biases_3',
        shape=pc.BIAS_SHAPE_3,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_4_biases = tf.get_variable(
        'biases_4',
        shape=pc.BIAS_SHAPE_4,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_5_biases = tf.get_variable(
        'biases_5',
        shape=pc.BIAS_SHAPE_5,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

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

    # define model
    with tf.variable_scope(scope_name):
        with tf.variable_scope('conv_1'):
            conv_1_layer = conv_unit(data, conv_1_weights, conv_1_biases)
        with tf.variable_scope('conv_2'):
            conv_2_layer = conv_unit(conv_1_layer, conv_2_weights, conv_2_biases)
        with tf.variable_scope('conv_3'):
            conv_3_layer = conv_unit(conv_2_layer, conv_3_weights, conv_3_biases)
        with tf.variable_scope('conv_4'):
            conv_4_layer = conv_unit(conv_3_layer, conv_4_weights, conv_4_biases)
        with tf.variable_scope('conv_5'):
            conv_5_layer = conv_unit(conv_4_layer, conv_5_weights, conv_5_biases)
    return conv_5_layer


def build_model_siamese_cnn(train_node_1, train_node_2):
    with tf.variable_scope('cnn_models') as scope:
        model_1 = build_cnn(train_node_1, 'cnn_model_1')
        scope.reuse_variables()
        model_2 = build_cnn(train_node_2, 'cnn_model_2')

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

    # get the distance between the 2 features (RMS)
    subtract = tf.sub(model_1, model_2)
    power = tf.pow(subtract, 2)
    reduce_sum = tf.reduce_sum(power, [1,2], keep_dims=True)
    square = tf.sqrt(reduce_sum)
    flattenit = flatten(square)

    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model_1, model_2), 2), [1,2], keep_dims=True))
    the_rest = flatten(distance)
    distance = the_rest
    # distance = tf.reshape(distance, [pc.BATCH_SIZE, ])

    fc_1 = tf.nn.relu(tf.matmul(distance, fc_1_weights) + fc_1_biases)
    return tf.matmul(fc_1, fc_2_weights) + fc_2_biases


def main():
    with tf.variable_scope('train-test') as scope:
        data = pu.load_data()
        labels = pu.load_labels()

        train_data = data[0]
        train_labels = labels[0]
        validation_data = data[1]
        validation_labels = labels[1]
        # training
        train_node_1 = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
        train_node_2 = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
        train_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.BATCH_SIZE, pc.NUM_CLASSES]) # needs to be OHE

        # validation
        validation_node_1 = tf.placeholder(pc.DATA_TYPE, shape=[pc.EVAL_BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
        validation_node_2 = tf.placeholder(pc.DATA_TYPE, shape=[pc.EVAL_BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
        validation_label_node = tf.placeholder(pc.DATA_TYPE, shape=[pc.EVAL_BATCH_SIZE, pc.NUM_CLASSES])

        logits = build_model_siamese_cnn(train_node_1, train_node_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, train_label_node))

        # TODO: maybe put regularizers and dropout

        # step = 0
        step = tf.Variable(0, dtype=pc.DATA_TYPE)
        # Decay once per epoch
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
        scope.reuse_variables()
        validation_prediction = tf.nn.softmax(build_model_siamese_cnn(validation_node_1, validation_node_2))

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
                        feed_dict={validation_node_1:data[begin:end, 0, ...],
                                   validation_node_2:data[begin:end, 1, ...]
                                   }
                    )
                        # feed_dict={validation_data: data[begin:end]})
                else:
                    batch_predictions = sess.run(
                        validation_prediction,
                        feed_dict={validation_data: data[-pc.EVAL_BATCH_SIZE:]})
                    predictions[begin:, :] = batch_predictions[begin - size:]
            return predictions


        start_time = time.time()
        with tf.Session() as sess:
            sess.run(init)
            epoch = 0
            print('Initialized!')
            # Loop through training steps.
            for train_step in range(0, pc.NUM_EPOCHS):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (train_step * pc.BATCH_SIZE) % (pc.NUM_TRAIN - pc.BATCH_SIZE)
                batch_data = train_data[offset:(offset + pc.BATCH_SIZE)]
                batch_labels = train_labels[offset:(offset + pc.BATCH_SIZE)]

                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.
                train_batch_1 = batch_data[:, 0, ...]
                train_batch_2 = batch_data[:, 1, ...]

                feed_dict = {
                                train_node_1: train_batch_1,
                                train_node_2: train_batch_2,
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
