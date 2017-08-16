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
# SAVE_LOCATION_MODEL_WEIGHTS = '/home/gabi/PycharmProjects/uatu/model_weights'
SAVE_LOCATION_RANKING_FILES = '../ranking_files'

SAVE_LOCATION_VIPER_CUHK = '../data/VIPeR_CUHK'

SAVE_LOCATION_RAW_DATA = '../raw_data'

POSITIVE_DATA = '../data/reid_all_positives.txt'
NEGATIVE_DATA = '../data/reid_all_negatives_uncompressed.h5'

LOCATION_RAW_VIPER = '../raw_data/VIPER'
LOCATION_RAW_CUHK01 = '../raw_data/CUHK01'

USE_BIAS = True
# RANKING_NUMBER = 125

RANKING_DICT = {'viper': 316,
                'cuhk01': 485,
                'cuhk02': 904, # removed 032_, 077_, 085_ from the IDs. this should work with 'half'. if you want custom ranking_number then set to a number divisible by 5
                'market': 750,
                'grid': 125,
                'prid450': 225,
                'caviar': 36}

RANKING_CUHK02_PARTS = {'P1': 485,
                        'P2': 152,
                        'P3': 52,
                        'P4': 96,
                        'P5': 119}

VIPER_DATA_STORAGE = '../data/VIPER'
VIPER_FIXED = '/home/gabi/Documents/datasets/VIPeR/padded'

CUHK01_DATA_STORAGE = '../data/CUHK'
CUHK01_FIXED = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images'

MARKET_DATA_STORAGE = '../data/market'
MARKET_FIXED = '/home/gabi/Documents/datasets/market-1501/identities'

CAVIAR_DATA_STORAGE = '../data/caviar'
CAVIAR_FIXED = '/home/gabi/Documents/datasets/CAVIAR4REID/fixed_caviar'

GRID_DATA_STORAGE = '../data/GRID'
GRID_FIXED = '/home/gabi/Documents/datasets/GRID/fixed_grid'

PRID450_DATA_STORAGE = '../data/prid450'
PRID450_FIXED = '/home/gabi/Documents/datasets/PRID450/fixed_prid'

PRID2011_DATA_STORAGE = '../data/prid2011'
PRID2011_FIXED = '/home/gabi/Documents/datasets/prid2011-fixed'

ILIDS_DATA_STORAGE = '../data/ilids-vid'
ILIDS_FIXED = '/home/gabi/Documents/datasets/ilids-vid-fixed'

ILIDS_20_DATA_STORAGE = '../data/ilids-vid-20'
ILIDS_20_FIXED = '/home/gabi/Documents/datasets/ilids-vid-fixed-20'