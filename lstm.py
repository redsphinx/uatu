import os
import random as rd
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell

IMAGE_HEIGHT = 20
IMAGE_WIDTH = 10
CHANNELS = 3
N_INPUT = IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS
CLASSES = 2

NUM_NEURONS = 200
NUM_LAYERS = 3

TRAINING_ITERS = 1000
DATA_TYPE = tf.float32
NUM_SEQUENCES = 200
NUM_IMAGES_IN_SEQUENCE = 5
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 0.0001
DISPLAY_STEP = 1

LOCATION_DATA_POSITIVE = '/home/gabi/Documents/datasets/noise/positive/'
LOCATION_DATA_NEGATIVE = '/home/gabi/Documents/datasets/noise/negative/'

data_paths = [LOCATION_DATA_POSITIVE, LOCATION_DATA_NEGATIVE]

# creates random sequences of images
def create_random_sequences(data_paths, num_sequences, num_images):
    # TODO find out if fixing number will break things
    FLAG_UPDATED= [0, 0]
    count = 0

    for path_name in data_paths:
        FLAG_CORRUPTED = 0
        print('checking: ' + str(path_name))

        if os.path.exists(path_name) and len(os.listdir(path_name)) == num_sequences:
            for number in range(0, num_sequences):
                if len(os.listdir(os.path.join(path_name, str(number)))) == num_images:
                    if number == num_images-1:
                        print('everything exists and looks ok')
                        FLAG_UPDATED[count] = 0
                else:
                    print('error in ' + str(path_name) + str(number))
                    FLAG_CORRUPTED = 1
                    break
        else:
            print('folder ' + str(path_name) + ' does not exist or it is corrupted')
            FLAG_CORRUPTED = 1

        if FLAG_CORRUPTED:
            FLAG_UPDATED[count] = 1
            print('removing corrupted folder ' + str(path_name) + ' and creating it again')
            if os.path.exists(path_name):
                os.rmdir(path_name)
            os.makedirs(path_name)
            print('generating images')
            for number in range(0, num_sequences):
                path = os.path.join(path_name, str(number))
                print('made: ' + str(path))
                os.mkdir(path)

                for number in range(0, num_images):
                    imarray = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) * 255
                    im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
                    file = os.path.join(path, path_name.split('/')[-2] + str(number) + '.jpg')
                    im.save(file)
        count = count + 1
    return FLAG_UPDATED


# create the labels
def create_labels(number):
    return [1] * number + [0] * number


# adds the full path to a file name
def make_list_with_full_path(path, my_list):
    my_list_with_full_path = []
    for item in range(0, len(list)):
        my_list_with_full_path.append(os.path.join(path, my_list[item]))
    return my_list_with_full_path


# create one hot encoding of a binary
def ohe(old, new):
    for item in range(0, len(old)):
        new[item][old[item]] = 1
    return new


# loads the data into numpy arrays
def load_data():
    FLAG_UPDATED = create_random_sequences(data_paths, NUM_SEQUENCES, NUM_IMAGES_IN_SEQUENCE)
    the_noise_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noise_folder')

    if os.path.exists(the_noise_folder) and sum(FLAG_UPDATED) == 0:
        print('loading data from files')
        # load file names
        train_data_ = np.genfromtxt(os.path.join(the_noise_folder, 'train_data.csv'), dtype=None)
        train_labels_ = np.genfromtxt(os.path.join(the_noise_folder, 'train_labels.csv'), dtype=None)
        validation_data_ = np.genfromtxt(os.path.join(the_noise_folder, 'validation_data.csv'), dtype=None)
        validation_labels_ = np.genfromtxt(os.path.join(the_noise_folder, 'validation_labels.csv'), dtype=None)
        testing_data_ = np.genfromtxt(os.path.join(the_noise_folder, 'testing_data.csv'), dtype=None)
        testing_labels_ = np.genfromtxt(os.path.join(the_noise_folder, 'testing_labels.csv'), dtype=None)

        train_data = np.zeros(shape=(len(train_data_), NUM_IMAGES_IN_SEQUENCE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        validation_data = np.zeros(shape=(len(validation_data_), NUM_IMAGES_IN_SEQUENCE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        testing_data = np.zeros(shape=(len(testing_data_), NUM_IMAGES_IN_SEQUENCE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))

        # put them in ndarrays
        for sequence in range(0, len(train_data_)):
            for image in range(0, NUM_IMAGES_IN_SEQUENCE):
                train_data[sequence][image] = ndimage.imread(os.path.join(train_data_[sequence], os.listdir(train_data_[sequence])[image]))

        for sequence in range(0, len(validation_data_)):
            for image in range(0, NUM_IMAGES_IN_SEQUENCE):
                validation_data[sequence][image] = ndimage.imread(os.path.join(validation_data_[sequence], os.listdir(validation_data_[sequence])[image]))

        for sequence in range(0, len(testing_data_)):
            for image in range(0, NUM_IMAGES_IN_SEQUENCE):
                testing_data[sequence][image] = ndimage.imread(os.path.join(testing_data_[sequence], os.listdir(testing_data_[sequence])[image]))

        train_labels = np.zeros(shape=(len(train_labels_), 2))
        validation_labels = np.zeros(shape=(len(validation_labels_), 2))
        testing_labels = np.zeros(shape=(len(testing_labels_), 2))

        train_labels = ohe(train_labels_, train_labels)
        validation_labels = ohe(validation_labels_, validation_labels)
        testing_labels = ohe(testing_labels_, testing_labels)
        print('sdf1')


    else:
        print('data files do not exist or are corrupted')
        print('creating files')
        if os.path.exists(the_noise_folder):
            shutil.rmtree(the_noise_folder)

        os.mkdir(the_noise_folder)
        files_positive = make_list_with_full_path(LOCATION_DATA_POSITIVE, os.listdir(LOCATION_DATA_POSITIVE))
        files_negative = make_list_with_full_path(LOCATION_DATA_NEGATIVE, os.listdir(LOCATION_DATA_NEGATIVE))
        all_files = files_positive + files_negative

        total_number_of_sequences = len(all_files)
        labels = create_labels(total_number_of_sequences/2)

        # assuming all sequences are the same length
        number_of_images_in_a_sequence = len(os.listdir(os.path.join(data_paths[0], os.listdir(data_paths[0])[0])))
        data = np.zeros(shape=(total_number_of_sequences, number_of_images_in_a_sequence, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        print('shape data:' + str(np.shape(data)))

        train_percentage = 0.5
        validation_percentage = (1 - train_percentage) / 2

        everything = list(zip(all_files, labels))
        rd.shuffle(everything)
        all_files, labels = zip(*everything)

        for sequence in range(0, len(all_files)):
            for image in range(0, number_of_images_in_a_sequence):
                data[sequence][image] = ndimage.imread(os.path.join(all_files[sequence], os.listdir(all_files[sequence])[image]))

        split_here_1 = int(len(all_files) * train_percentage)
        split_here_2 = int(len(all_files) * validation_percentage)

        train_data = all_files[0:split_here_1]
        train_labels = labels[0:split_here_1]
        validation_data = all_files[split_here_1:split_here_1 + split_here_2]
        validation_labels = labels[split_here_1:split_here_1 + split_here_2]
        testing_data = all_files[split_here_2:len(all_files)]
        testing_labels = labels[split_here_2:len(all_files)]

        train_data_file = os.path.join(the_noise_folder, 'train_data.csv')
        train_labels_file =  os.path.join(the_noise_folder, 'train_labels.csv')
        validation_data_file =  os.path.join(the_noise_folder, 'validation_data.csv')
        validation_labels_file =  os.path.join(the_noise_folder, 'validation_labels.csv')
        testing_data_file = os.path.join(the_noise_folder, 'testing_data.csv')
        testing_labels_file = os.path.join(the_noise_folder, 'testing_labels.csv')

        with open(train_data_file, 'wr') as my_file:
            for item in range(0, len(train_data)):
                my_file.write(str(train_data[item]) + '\n')
        with open(train_labels_file, 'wr') as my_file:
            for item in range(0, len(train_labels)):
                my_file.write(str(train_labels[item]) + '\n')

        with open(validation_data_file, 'wr') as my_file:
            for item in range(0, len(validation_data)):
                my_file.write(str(validation_data[item]) + '\n')
        with open(validation_labels_file, 'wr') as my_file:
            for item in range(0, len(validation_labels)):
                my_file.write(str(validation_labels[item]) + '\n')

        with open(testing_data_file, 'wr') as my_file:
            for item in range(0, len(testing_data)):
                my_file.write(str(testing_data[item]) + '\n')
        with open(testing_labels_file, 'wr') as my_file:
            for item in range(0, len(testing_labels)):
                my_file.write(str(testing_labels[item]) + '\n')

        train_data = np.asarray(data[0:split_here_1])
        train_labels = np.asarray(train_labels)
        validation_data = np.asarray(data[split_here_1:split_here_1 + split_here_2])
        validation_labels = np.asarray(validation_labels)
        testing_data = np.asarray(data[split_here_2:len(all_files)])
        testing_labels = np.asarray(testing_labels)

        train_labels_ = np.zeros(shape=(len(train_labels), 2))
        validation_labels_ = np.zeros(shape=(len(validation_labels), 2))
        testing_labels_ = np.zeros(shape=(len(testing_labels), 2))

        train_labels = ohe(train_labels, train_labels_)
        validation_labels = ohe(validation_labels, validation_labels_)
        testing_labels = ohe(testing_labels, testing_labels_)
        print('sdf2')

    return [train_data, train_labels, validation_data, validation_labels, testing_data, testing_labels]


# turns ndarray to a tuple all the way down
def tupconv(lst):
    tuplst = []
    for x in lst:
        if isinstance(x, np.ndarray):
            tuplst.append(tupconv(x))
        elif isinstance(x, list):
            tuplst.append(tupconv(x))
        else:
            tuplst.append(x)
    return tuple(tuplst)


def main():
    # load the data
    [train_data, train_labels, validation_data, validation_labels, testing_data, testing_labels] = load_data()
    print('shape train data: ' + str(np.shape(train_data)))
    # shape train data: (200, 5, 20, 10, 3)

    # create placeholders
    train_data_node = tf.placeholder(
        DATA_TYPE,
        shape=(BATCH_SIZE, NUM_IMAGES_IN_SEQUENCE, N_INPUT)
    )

    # >transform the data into multiple tensors, 1 for each frame in the sequence
    # Permuting batch_size and n_steps
    train_data_node = tf.transpose(train_data_node, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    train_data_node = tf.reshape(train_data_node, [-1, N_INPUT])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    train_data_node = tf.split(0, NUM_IMAGES_IN_SEQUENCE, train_data_node)

    print(type(train_data_node))
    print(train_data_node)

    train_labels_node = tf.placeholder(
        DATA_TYPE,
        shape=(None, CLASSES)
    )
    eval_data_node = tf.placeholder(
        DATA_TYPE,
        shape=(EVAL_BATCH_SIZE, NUM_IMAGES_IN_SEQUENCE, N_INPUT)
    )

    weights = {
        'out': tf.Variable(tf.random_normal([NUM_NEURONS, CLASSES]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([CLASSES]))
    }

    def model(data, weights, biases):
        cell = LSTMCell(NUM_NEURONS)  # Or LSTMCell(num_neurons)
        cell = MultiRNNCell([cell] * NUM_LAYERS)

        output, _ = tf.nn.rnn(cell, train_data_node, dtype=DATA_TYPE)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        out_size = int(train_labels_node.get_shape()[1])

        prediction = tf.nn.softmax(tf.matmul(last, weights['out']) + biases['out'])
        # cross_entropy = -tf.reduce_sum(train_labels_node * tf.log(prediction))
        return prediction


    pred = model(train_data_node, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=train_labels_node))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(train_labels_node, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # running everything
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * BATCH_SIZE < TRAINING_ITERS:
            print(step)
            print(step * BATCH_SIZE )

            batch_x = train_data[step-1, ...]
            asdf = train_data[step-1]
            print(np.shape(batch_x))
            batch_x = batch_x.reshape((BATCH_SIZE, NUM_IMAGES_IN_SEQUENCE, N_INPUT))
            print(np.shape(batch_x))


            batch_x = tupconv(batch_x)
            # print(type(batch_x))
            # print(np.shape(batch_x))
            # xx = batch_x[0][0]
            # print(np.shape(xx))
            # batch_x = xx
            batch_y = train_labels[step-1, :]
            # batch_y = tuple(batch_y)

            # a = batch_x
            # a = a[0][0]
            # print('asdf')

            # a = a.astype(np.float)

            # TODO: fix "TypeError: unhashable type: 'list'"
            sess.run(optimizer, feed_dict={train_data_node: batch_x,
                                           train_labels_node: batch_y})
            # sess.run(optimizer, feed_dict={train_data_node: a})
            # sess.run(optimizer, feed_dict={train_labels_node: batch_y})
            if step % DISPLAY_STEP == 0:

                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={train_data_node: batch_x,
                                                    train_labels_node: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={train_data_node: batch_x, train_labels_node: batch_y})
                print("Iter " + str(step * BATCH_SIZE) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
            step = step + 1
        print("Optimization Finished!")

        # # Calculate accuracy for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        # test_label = mnist.test.labels[:test_len]
        # print("Testing Accuracy:", \
        #       sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


main()