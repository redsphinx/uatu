import tensorflow as tf
import numpy as np


NUM_CHANNELS = 3
DATA_TYPE = tf.float32
NUM_CLASSES = 2
START_LEARNING_RATE = 0.0001

KERNEL_SHAPE_1 = [3, 3, NUM_CHANNELS, 16]
KERNEL_SHAPE_2 = [3, 3, 16, 32]
KERNEL_SHAPE_3 = [3, 3, 32, 64]
KERNEL_SHAPE_4 = [3, 3, 64, 128]
KERNEL_SHAPE_5 = [3, 3, 128, 256]

BIAS_SHAPE_1 = [16]
BIAS_SHAPE_2 = [32]
BIAS_SHAPE_3 = [64]
BIAS_SHAPE_4 = [128]
BIAS_SHAPE_5 = [256]

BATCH_SIZE = 1
NUM_TRAIN = 10
DECAY_STEP = NUM_TRAIN
DECAY_RATE = 0.95
MOMENTUM = 0.9