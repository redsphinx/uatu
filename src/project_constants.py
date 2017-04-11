import tensorflow as tf
import numpy as np


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
SEED = 42
AMOUNT_DATA = 10 # variable, TODO fix it

NUM_CHANNELS = 3
DATA_TYPE = tf.float32
NUM_CLASSES = 2
START_LEARNING_RATE = 0.00001

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

BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE

NUM_TRAIN = AMOUNT_DATA
DECAY_STEP = NUM_TRAIN
DECAY_RATE = 0.95
MOMENTUM = 0.9
NUM_EPOCHS = 15
EVAL_FREQUENCY = 50

LOAD_WEIGHTS = False
CHECKPOINT = 'cnn_model_10_epochs.meta'

DROPOUT = 0.5
NUM_CAMERAS = 2
FEATURES = IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS
VERBOSE = False
TRANSFER_LEARNING = True
TRAIN_CNN = False
LOG_FILE_PATH = 'experiment_log.txt'
LOGGING = True
SIMILARITY_METRIC = 'fc_layers'
TEST_DATA_SIZE = 150
NUM_SIAMESE_HEADS = 2