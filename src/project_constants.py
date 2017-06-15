import tensorflow as tf
import numpy as np


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64

NUM_CHANNELS = 3
DATA_TYPE = tf.float32
NUM_CLASSES = 2
DECAY_RATE = 0.95
MOMENTUM = 0.9
LOAD_WEIGHTS = False

DROPOUT = 0.5
NUM_CAMERAS = 2
VERBOSE = False

TRANSFER_LEARNING = False
TRAIN_CNN = False
LOG_FILE_PATH = 'experiment_log_2.txt'

LOGGING = True
SIMILARITY_METRIC = 'fc_layers'
NUM_SIAMESE_HEADS = 2
SAVE_CNN_WEIGHTS = True
SAVE_CNN_MODEL = False

SAVE_LOCATION_MODEL_WEIGHTS = '../model_weights'
SAVE_LOCATION_RANKING_FILES = '../ranking_files'

SAVE_LOCATION_VIPER_CUHK = '../data/VIPeR_CUHK'

POSITIVE_DATA = '../data/reid_all_positives.txt'
NEGATIVE_DATA = '../data/reid_all_negatives_uncompressed.h5'

LOCATION_RAW_VIPER = '../raw_data/VIPER'
LOCATION_RAW_CUHK01 = '../raw_data/CUHK01'

USE_BIAS = True
# RANKING_NUMBER = 125
