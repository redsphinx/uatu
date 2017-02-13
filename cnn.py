import tensorflow as tf
import numpy as np
import time
import sys
import argparse
import os
import random as rd
from PIL import Image
from scipy import ndimage

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
CHANNELS = 3
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
DATA_TYPE = tf.float32
NUM_LABELS = 2
SEED = 42
NUM_EPOCHS = 100
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

# tensorboard
LOG_DIR = '/tmp/TF'

LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/humans/1/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/humans/0/'


def create_not_humans(num_images):
    for number in range(0, num_images):
        imarray = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        im.save(LOCATION_DATA_NEGATIVE + 'not_human_' + str(number) + '.png')
    pass


def create_labels(number):
    return [1]*number + [0]*number


def make_list_with_full_path(path, list):
    list_with_full_path = []
    for item in range(0, len(list)):
        list_with_full_path.append(os.path.join(path, list[item]))
    return list_with_full_path


def do_things_data():
    # split data into train and validation
    pos_data = os.listdir(LOCATION_DATA_POSITIVE)
    num_images = len(pos_data)

    if os._exists(LOCATION_DATA_NEGATIVE):
        if len(os.listdir(LOCATION_DATA_NEGATIVE)) != num_images:
            print('creating negative instances')
            create_not_humans(num_images)

    all_data = make_list_with_full_path(LOCATION_DATA_POSITIVE, os.listdir(LOCATION_DATA_POSITIVE)) \
                + make_list_with_full_path(LOCATION_DATA_NEGATIVE, os.listdir(LOCATION_DATA_NEGATIVE))
    labels = create_labels(num_images)

    # shuffle
    data_and_labels = list(zip(all_data, labels))
    rd.shuffle(data_and_labels)
    all_data, labels = zip(*data_and_labels)

    percentage_test = 0.2
    split_here = int((1-percentage_test)*len(all_data))

    train_data = all_data[0:split_here]
    train_labels = labels[0:split_here]

    validation_data = all_data[split_here:len(all_data)]
    validation_labels = labels[split_here:len(all_data)]

    train_data_file = 'train_data.csv'
    train_labels_file = 'train_labels.csv'
    validation_data_file = 'validation_data.csv'
    validation_labels_file = 'validation_labels.csv'
    with open(train_data_file, 'wr') as file:
        for item in range(0, len(train_data)):
            file.write(str(train_data[item]) + '\n')
    with open(train_labels_file, 'wr') as file:
        for item in range(0, len(train_labels)):
            file.write(str(train_labels[item]) + '\n')
    with open(validation_data_file, 'wr') as file:
        for item in range(0, len(validation_data)):
            file.write(str(validation_data[item]) + '\n')
    with open(validation_labels_file, 'wr') as file:
        for item in range(0, len(validation_labels)):
            file.write(str(validation_labels[item]) + '\n')

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    validation_data = np.asarray(validation_data)
    validation_labels = np.asarray(validation_labels)

    return [train_data, train_labels, validation_data, validation_labels]


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


# load data from list into np.ndarray
def load_data(filenames_list):
    data = np.zeros(shape=(len(filenames_list), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    for item in range(0, len(filenames_list)):
        data[item] = ndimage.imread(filenames_list[item])
    return data


def main(_):
    # tensorboard
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    # data stuff
    test_data = []
    test_labels = []

    [train_data_list, train_labels_list, validation_data_list, validation_labels_list] = do_things_data()

    # ---load the images here
    train_data = load_data(train_data_list)
    train_labels = train_labels_list
    validation_data = load_data(validation_data_list)
    validation_labels = validation_labels_list
    # ---

    num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]

    # defining placeholders
    train_data_node = tf.placeholder(
        DATA_TYPE,
        shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        DATA_TYPE,
        shape=(EVAL_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))

    # tensorboard
    image_shaped_input = tf.reshape(train_data_node, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
    tf.summary.image('input', image_shaped_input, BATCH_SIZE)


    # tensorboard
    def variable_summaries(var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


    # weights and biases
    conv1_weights = tf.Variable(
        tf.truncated_normal([32, 32, CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=DATA_TYPE))
    variable_summaries(conv1_weights)

    # conv1_weights = tf.get_variable('conv1_weights', shape=(3, 3, CHANNELS, 32),
    #                                 initializer=tf.contrib.layers.xavier_initializer(),
    #                                 dtype=DATA_TYPE, trainable=True)

    conv1_biases = tf.Variable(tf.zeros([32], dtype=DATA_TYPE))
    # conv1_biases = tf.get_variable('conv1_biases', shape=(32), dtype=DATA_TYPE, trainable=True)
    variable_summaries(conv1_biases)

    conv2_weights = tf.Variable(tf.truncated_normal(
        [3, 3, 32, 64], stddev=0.1,
        seed=SEED, dtype=DATA_TYPE))
    variable_summaries(conv2_weights)

    # conv2_weights = tf.get_variable('conv2_weights', shape=(3, 3, 32, 64),
    #                                 initializer=tf.contrib.layers.xavier_initializer(),
    #                                 dtype=DATA_TYPE, trainable=True)

    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=DATA_TYPE))
    # conv2_biases = tf.get_variable('conv2_biases', shape=(64), dtype=DATA_TYPE, trainable=True)
    variable_summaries(conv2_biases)


    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=DATA_TYPE))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=DATA_TYPE))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=DATA_TYPE))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=DATA_TYPE))

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

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
    # tensorboard
    tf.summary.scalar('loss', loss)

    batch = tf.Variable(0, dtype=DATA_TYPE)
    learning_rate = tf.train.exponential_decay(
        0.0001,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    # tensorboard
    tf.summary.scalar('learning rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data))


    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
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
        print('Initialized!')



        # Loop through training steps.
        for step in range(int(num_epochs * train_size) / BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a np array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()

                # tensorboard
                summary, ls = sess.run([merged, loss], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)


                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * float(BATCH_SIZE) / train_size,
                       1000 * float(elapsed_time) / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
            else:
                summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
        # Finally print the result!
        # test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        # print('Test error: %.1f%%' % test_error)
        train_writer.close()
        test_writer.close()

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