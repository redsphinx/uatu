import tensorflow as tf
import os
import pandas as pd
import csv
import random as rd

rd.seed(42)

DATA_DIRECTORY = 'PRID2011'
NUMBER_OF_TRACKLETS = len(os.listdir(DATA_DIRECTORY)) - 1
VALIDATION_SIZE = 50
NUM_LABELS = 2
DTYPE = tf.float32
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_CHANNELS = 3

# things that can change, depending on person
IMAGE_WIDTH = 0
IMAGE_HEIGHT = 0
NUM_FRAMES = 0


# make the dataset balanced
def balance_data():
    labels_ = DATA_DIRECTORY + '/pair_labels.csv'
    labels = pd.read_csv(labels_)
    labels = labels.sort_values('class')
    labels.to_csv('labels_sorted.csv', index=False)
    # now labels are sorted with class 0 on top and 1 at the bottom
    with open('labels_sorted.csv') as f:
        reader = csv.reader(f)
        labels = list(reader)
    labels_neg = labels[1:len(labels)-200]
    rd.shuffle(labels_neg)
    labels_pos = labels[len(labels)-201:-1]
    rd.shuffle(labels_pos)
    balanced_labels = labels_neg[0:200] + labels_pos
    rd.shuffle(balanced_labels)
    return balanced_labels


# create the validation dataset
def create_data():
    labels = balance_data()
    print(labels[-1])
    train = labels[0:len(labels)-VALIDATION_SIZE]
    validation = labels[len(labels)-VALIDATION_SIZE:len(labels)]
    return [train, validation]


def main():
    # TODO:extract data into numpy arrays

    train_data_node = tf.placeholder(
        DTYPE,
        shape=(BATCH_SIZE, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        DTYPE,
        shape=(EVAL_BATCH_SIZE, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

    # model architecture
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))


    pass

create_data()