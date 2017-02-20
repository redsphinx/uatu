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


# tensorboard
LOG_DIR = '/tmp/TF'

LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/humans/1/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/humans/0/'


def create_labels(number):
    return [1]*number + [0]*number


def make_list_with_full_path(path, list):
    list_with_full_path = []
    for item in range(0, len(list)):
        list_with_full_path.append(os.path.join(path, list[item]))
    return list_with_full_path


def error_rate(predictions, labels, step, log_file):
    if step == 'testing':
        with open(log_file, 'a') as my_file:
            for line in range(0, len(labels)):
                predict = np.argmax(predictions[line])
                target = labels[line]
                my_file.write('step,' + str(step) + ',target,' + str(target) + ',' +
                                'prediction,' + str(predict) + ',' + '\n')

    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def main(_):
    # tensorboard
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    log_file = 'wrong_predictions.txt'
    with open(log_file, 'w') as my_file:
        print('new log file made')


    # data stuff
    test_data = []
    test_labels = []

    [train_data, train_labels, validation_data_, validation_labels_] = pu.load_human_detection_data()


    test_data = validation_data_[len(validation_data_)/2:len(validation_data_)]
    test_labels = validation_labels_[len(validation_labels_)/2:len(validation_labels_)]


    validation_data = validation_data_[0:len(validation_data_)/2]
    validation_labels = validation_labels_[0:len(validation_labels_)/2]

    print('train: %d, validation: %d, test: %d)' % (len(train_data), len(validation_data), len(test_data)))

    num_epochs = pc.NUM_EPOCHS

    # train_size = train_labels.shape[0]
    train_size = len(train_data)

    # defining placeholders
    train_data_node = tf.placeholder(
        pc.DATA_TYPE,
        shape=(pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(pc.BATCH_SIZE,))
    eval_data = tf.placeholder(
        pc.DATA_TYPE,
        shape=(pc.EVAL_BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))

    # tensorboard
    image_shaped_input = tf.reshape(train_data_node, [pc.BATCH_SIZE, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    tf.summary.image('input', image_shaped_input, pc.BATCH_SIZE)


    # tensorboard
    def variable_summaries(var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


    # load saved weights if they exist and if the flag pc.LOAD_WEIGHTS is True
    if pc.LOAD_WEIGHTS and os.path.exists(pc.CHECKPOINT):
        print('loading the weights and biases later')
    else:
        conv1_weights = tf.get_variable('conv1_weights', shape=(3, 3, pc.NUM_CHANNELS, 32),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)

        conv1_biases = tf.Variable(tf.zeros([32], dtype=pc.DATA_TYPE))
        variable_summaries(conv1_biases)


        conv2_weights = tf.get_variable('conv2_weights', shape=(3, 3, 32, 64),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)
        variable_summaries(conv2_weights)

        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=pc.DATA_TYPE))
        variable_summaries(conv2_biases)


        conv3_weights = tf.get_variable('conv3_weights', shape=(3, 3, 64, 128),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)
        variable_summaries(conv3_weights)

        conv3_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=pc.DATA_TYPE))
        variable_summaries(conv3_biases)

        conv4_weights = tf.get_variable('conv4_weights', shape=(3, 3, 128, 256),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)
        variable_summaries(conv4_weights)

        conv4_biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=pc.DATA_TYPE))
        variable_summaries(conv4_biases)


    #--
        conv5_weights = tf.get_variable('conv5_weights', shape=(3, 3, 256, 512),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)
        variable_summaries(conv5_weights)

        conv5_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=pc.DATA_TYPE))
        variable_summaries(conv5_biases)

    #--
        # --
        conv6_weights = tf.get_variable('conv6_weights', shape=(3, 3, 512, 1024),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=pc.DATA_TYPE, trainable=True)
        variable_summaries(conv6_weights)

        conv6_biases = tf.Variable(tf.constant(0.1, shape=[1024], dtype=pc.DATA_TYPE))
        variable_summaries(conv6_biases)

        # # --
        num_pools = 6
        shape_last_cnn = [pc.BATCH_SIZE, pc.IMAGE_HEIGHT / (2**num_pools), pc.IMAGE_WIDTH / (2**num_pools), 1024]
        input_to_fc = shape_last_cnn[1]*shape_last_cnn[2]*shape_last_cnn[3]

        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([input_to_fc, 512],
                                stddev=0.1,
                                seed=pc.SEED,
                                dtype=pc.DATA_TYPE))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=pc.DATA_TYPE))
        fc2_weights = tf.Variable(tf.truncated_normal([512, pc.NUM_CLASSES],
                                                      stddev=0.1,
                                                      seed=pc.SEED,
                                                      dtype=pc.DATA_TYPE))
        fc2_biases = tf.Variable(tf.constant(
            0.1, shape=[pc.NUM_CLASSES], dtype=pc.DATA_TYPE))

        tf.add_to_collection('variables', conv1_weights)
        tf.add_to_collection('variables', conv1_biases)
        tf.add_to_collection('variables', conv2_weights)
        tf.add_to_collection('variables', conv2_biases)
        tf.add_to_collection('variables', conv3_weights)
        tf.add_to_collection('variables', conv3_biases)
        tf.add_to_collection('variables', conv4_weights)
        tf.add_to_collection('variables', conv4_biases)
        tf.add_to_collection('variables', conv5_weights)
        tf.add_to_collection('variables', conv5_biases)
        tf.add_to_collection('variables', conv6_weights)
        tf.add_to_collection('variables', conv6_biases)

        tf.add_to_collection('variables', fc1_weights)
        tf.add_to_collection('variables', fc1_biases)
        tf.add_to_collection('variables', fc2_weights)
        tf.add_to_collection('variables', fc2_biases)

        saver = tf.train.Saver()



    # defining the model architecture
    def model(data, train=False):
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv5_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv6_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv6_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=pc.SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases


    # TODO fix reference before assignment issue of the fc_weights

    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, train_labels_node))


    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
    # tensorboard
    tf.summary.scalar('loss', loss)

    batch = tf.Variable(0, dtype=pc.DATA_TYPE)
    learning_rate = tf.train.exponential_decay(
        pc.START_LEARNING_RATE,  # Base learning rate.
        batch * pc.BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    # tensorboard
    tf.summary.scalar('learning rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9, use_nesterov=True).minimize(loss,
                                                         global_step=batch)
    # remember: logits = model(train_data_node, True)
    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data))


    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < pc.EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, pc.NUM_CLASSES), dtype=np.float32)
        for begin in range(0, size, pc.EVAL_BATCH_SIZE):
            end = begin + pc.EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-pc.EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]

        return predictions


    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()

        # TODO load the stored weights
        if pc.LOAD_WEIGHTS and os.path.exists(pc.CHECKPOINT):
            print('loading weights and biases')
            import_saver = tf.train.import_meta_graph(pc.CHECKPOINT)
            import_saver.restore(sess, tf.train.latest_checkpoint('./'))
            all_vars = tf.get_collection('variables')
            conv1_weights = all_vars[0]
            conv1_biases = all_vars[1]
            conv2_weights = all_vars[2]
            conv2_biases = all_vars[3]
            conv3_weights = all_vars[4]
            conv3_biases = all_vars[5]
            conv4_weights = all_vars[6]
            conv4_biases = all_vars[7]
            conv5_weights = all_vars[8]
            conv5_biases = all_vars[9]
            conv6_weights = all_vars[10]
            conv6_biases = all_vars[11]

            fc1_weights = all_vars[12]
            fc1_biases = all_vars[13]
            fc2_weights = all_vars[14]
            fc2_biases = all_vars[15]
        else:
            saver.save(sess, pc.CHECKPOINT.split('.')[0])

        print('Initialized!')
        tot_steps = int(num_epochs * train_size) / pc.BATCH_SIZE
        # Loop through training steps.
        for step in range(int(num_epochs * train_size) / pc.BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * pc.BATCH_SIZE) % (train_size - pc.BATCH_SIZE)
            batch_data = train_data[offset:(offset + pc.BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + pc.BATCH_SIZE)]
            # This dictionary maps the batch data (as a np array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % pc.EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print('Step %d out of %d (epoch %.2f), %.1f ms' %
                      (step, tot_steps, float(step) * float(pc.BATCH_SIZE) / train_size,
                       1000 * float(elapsed_time) / pc.EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels, step, log_file))

                error = error_rate(predictions, batch_labels, step, log_file)
                tf.summary.scalar('minibatch error', error)

                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels, step, log_file))
                print('\n')
                val_error = error_rate(eval_in_batches(validation_data, sess), validation_labels, step, log_file)
                tf.summary.scalar('validation error', val_error)

                # tensorboard
                summary, ls = sess.run([merged, loss], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)


                sys.stdout.flush()
            else:
                summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels, 'testing', log_file)
        print('Test error: %.1f%%' % test_error)
        train_writer.close()
        test_writer.close()
    pu.get_wrong_predictions()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)