import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
# import cnn_human_detection as cnn
import numpy as np
import project_utils as pu


def test():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.experiment_name = 'testing'
    a.iterations = 1
    a.epochs = 10
    a.batch_size = 32
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    scn.super_main(a)


# ------------------------------------------------------------------------------------
# neural layers: type of merging
# ------------------------------------------------------------------------------------
# 1_0	neural_distance=absolute
def ex_1_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_0_0: viper, neural_distance=absolute'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'absolute'
    scn.super_main(a)


def ex_1_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_0_1: grid, neural_distance=absolute'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'absolute'
    scn.super_main(a)


def ex_1_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_0_2: prid450, neural_distance=absolute'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'absolute'
    scn.super_main(a)


# 1_1	neural_distance=subtract
def ex_1_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_0: viper, neural_distance=subtract'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'subtract'
    scn.super_main(a)


def ex_1_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_1: grid, neural_distance=subtract'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'subtract'
    scn.super_main(a)


def ex_1_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_2: prid450, neural_distance=subtract'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'subtract'
    scn.super_main(a)


# 1_2	neural_distance=concatenate
def ex_1_2_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_0: viper, neural_distance=concatenate'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'concatenate'
    scn.super_main(a)


def ex_1_2_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_1: grid, neural_distance=concatenate'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'concatenate'
    scn.super_main(a)


def ex_1_2_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_2: prid450, neural_distance=concatenate'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'concatenate'
    scn.super_main(a)


# 1_3	neural_distance=divide
def ex_1_3_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_0: viper, neural_distance=divide'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'divide'
    scn.super_main(a)


def ex_1_3_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_1: grid, neural_distance=divide'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'divide'
    scn.super_main(a)


def ex_1_3_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_2: prid450, neural_distance=divide'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'divide'
    scn.super_main(a)


# 1_4	neural_distance=add
def ex_1_4_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_0: viper, neural_distance=add'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)

def ex_1_4_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_1: grid, neural_distance=add'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)

def ex_1_4_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_2: prid450, neural_distance=add'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


# 1_5	neural_distance=multiply
def ex_1_5_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_0: viper, neural_distance=multiply'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'multiply'
    scn.super_main(a)

def ex_1_5_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_1: grid, neural_distance=multiply'
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'multiply'
    scn.super_main(a)

def ex_1_5_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_2: prid450, neural_distance=multiply'
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'multiply'
    scn.super_main(a)

# ------------------------------------------------------------------------------------
#
# 2	non-neural vs. neural on [viper, grid, prid450] (note: vital)
#
# 2_0	cost_module_type=euclidean, lr=0.00001
def ex_2_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_0: viper, cost_module_type=euclidean, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_1: grid, cost_module_type=euclidean, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_2: prid450, cost_module_type=euclidean, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


# 2_1 cost_module_type=cosine, lr=0.00001
def ex_2_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_0: viper, cost_module_type=cosine, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_1: grid, cost_module_type=cosine, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_2: prid450, cost_module_type=cosine, lr=0.00001'
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


# ------------------------------------------------------------------------------------
# experiments video -> siamese_cnn_video.py (note: vital)
# ------------------------------------------------------------------------------------
#
# 10	3D convolution vs. cnn_lstm (single dataset)
#
# 10_0	video_head_type=3d_convolution, no batchnorm
def ex_10_0_0():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_0_0: video_head_type=3d_convolution on ilids, no batchnorm'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    srcn.super_main(a)


def ex_10_0_1():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_0_1: video_head_type=3d_convolution on prid2011, no batchnorm'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    srcn.super_main(a)


# 10_3	video_head_type=3d_convolution, with batchnorm, with concatenation
def ex_10_3_0():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_3_0: video_head_type=3d_convolution on ilids, with batchnorm, with concatenation'
    a.neural_distance = 'concatenate'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)


def ex_10_3_1():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_3_1: video_head_type=3d_convolution on prid2011, with batchnorm, with concatenation'
    a.neural_distance = 'concatenate'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)


# 10_4	video_head_type=cnn_lstm, with concatenation
def ex_10_4_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_2_0: video_head_type=cnn_lstm on ilids, with concatenation'
    a.neural_distance = 'concatenate'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = 'cnn_lstm'
    a.use_cyclical_learning_rate = False
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    a.lstm_units = 256
    srcn.super_main(a)


def ex_10_4_1():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_2_1: video_head_type=cnn_lstm on prid2011, with concatenation'
    a.neural_distance = 'concatenate'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = 'cnn_lstm'
    a.use_cyclical_learning_rate = False
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    a.lstm_units = 256
    srcn.super_main(a)

# 10_1	video_head_type=3d_convolution, with batchnorm
def ex_10_1_0():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_1_0: video_head_type=3d_convolution on ilids, with batchnorm'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]

    srcn.super_main(a)


def ex_10_1_1():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_1_1: video_head_type=3d_convolution on prid2011, with batchnorm'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)


# 10_2	video_head_type=cnn_lstm
def ex_10_2_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_2_0: video_head_type=cnn_lstm on ilids'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = 'cnn_lstm'
    a.use_cyclical_learning_rate = False
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    a.lstm_units = 256
    srcn.super_main(a)


def ex_10_2_1():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 10_2_1: video_head_type=cnn_lstm on prid2011'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = 'cnn_lstm'
    a.use_cyclical_learning_rate = False
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    a.lstm_units = 256
    srcn.super_main(a)


# ------------------------------------------------------------------------------------
#
# 4	training: mix [viper, grid, prid450] including test (note: vital)
#
# 4_0	no batchnorm
def ex_4_0_0():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_0_0: test=viper, train=[grid, prid450], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_0_1():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_0_1: test=grid, train=[viper, prid450], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_0_2():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_0_2: test=prid450, train=[viper, grid], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# 4_1 with batchnorm
def ex_4_1_0():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_1_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_1():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_1_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_2():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_1_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# 4_2 no batchnorm, no selu, no AD
def ex_4_2_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_2_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_2_1():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_2_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_2_2():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_2_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# ------------------------------------------------------------------------------------
#
# 5	training: train on all mixed, exclude test + Test. (note: vital)
# so training will be mixed+the target train data on the end
#
# 5_0	no batch_norm
def ex_5_0_0():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_0_0: test=viper, train=[grid, prid450], no batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_1():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_0_1: test=grid, train=[viper, prid450], no batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_2():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_0_2: test=prid450, train=[viper, grid], no batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


# 5_1	with batch_norm
def ex_5_1_0():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_1_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_1():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_1_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_2():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_1_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 20

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


# 5_2, no batchnorm, no selu no AD
def ex_5_2_0():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_2_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_2_1():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_2_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_2_2():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_2_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 20
    a.head_type = 'simple'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)



# ------------------------------------------------------------------------------------
#
# 6	training: train on all mixed, exclude test. Then retrain trained network on the test. (note: vital)
# so the network will learn on mixed data and then retrain on the target train dataset
#
# 6_0	no batch_norm
def ex_6_0_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['grid', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'g_p_mix'

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_0_0: train only=[grid, prid450], no batchnorm, mix then retrain on test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_0_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['viper', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_p_mix'

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_0_1: train only=[viper, prid450], no batchnorm, mix then retrain on test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_0_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['viper', 'grid']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_g_mix'

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_0_2: train only=[viper, grid], no batchnorm, mix then retrain on test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 6_1	with batch_norm
def ex_6_1_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['grid', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'g_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_1_0: train only=[grid, prid450], with batchnorm, mix then retrain on test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_1_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['viper', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_1_1: train only=[viper, prid450], with batchnorm, mix then retrain on test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_1_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.datasets_train = ['viper', 'grid']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_g_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_1_2: train only=[viper, grid], with batchnorm, mix then retrain on test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)

# 6_2, no batchnorm, no selu, no AD
def ex_6_2_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.datasets_train = ['grid', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'g_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_2_0: train only=[grid, prid450], with batchnorm, mix then retrain on test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)

def ex_6_2_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.datasets_train = ['viper', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_2_1: train only=[viper, prid450], with batchnorm, mix then retrain on test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)

def ex_6_2_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.datasets_train = ['viper', 'grid']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_g_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.head_type = 'simple'

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_2_2: train only=[viper, grid], with batchnorm, mix then retrain on test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# ------------------------------------------------------------------------------------
#
# 7	training: train on all ordered for subset={viper, grid, prid450}, with batchnorm (note: vital)
# so train on dataset A, save, then B, save then train+test on C

# 7_0 train order: grid, prid450, viper
def ex_7_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_0: train only then save on each dataset [grid, prid450]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 7_1 train order: prid450, grid, viper
def ex_7_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_1: train only then save on each dataset [prid450, grid]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 7_2 train order: grid, viper, prid450
def ex_7_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu

        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_2: train only then save on each dataset [grid, viper]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 7_3 train order: viper, grid, prid450
def ex_7_3():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_3: train only then save on each dataset [viper, grid]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 7_4 train order: viper, prid450, grid
def ex_7_4():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_4: train only then save on each dataset [viper, prid450]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 7_5 train order: prid450, viper, grid
def ex_7_5():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 7_5: train only then save on each dataset [prid450, viper]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# ------------------------------------------------------------------------------------
#
# 8	training: train on all ordered for subset={viper, grid, prid450}, no batchnorm (note: vital)
# so train on dataset A, save, then B, save then train+test on C

# 8_0 train order: grid, prid450, viper
def ex_8_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_0: train only then save on each dataset [grid, prid450]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 8_1 train order: prid450, grid, viper
def ex_8_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_1: train only then save on each dataset [prid450, grid]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 8_2 train order: grid, viper, prid450
def ex_8_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_2: train only then save on each dataset [grid, viper]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 8_3 train order: viper, grid, prid450
def ex_8_3():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_3: train only then save on each dataset [viper, grid]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 8_4 train order: viper, prid450, grid
def ex_8_4():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_4: train only then save on each dataset [viper, prid450]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 8_5 train order: prid450, viper, grid
def ex_8_5():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s.txt' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s.txt' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s.txt' % a.use_gpu
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s.txt' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 8_5: train only then save on each dataset [prid450, viper]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# ------------------------------------------------------------------------------------
#
# 9	training: train on all ordered for subset={viper, grid, prid450}, no batchnorm, no selu, no AD
# so train on dataset A, save, then B, save then train+test on C

# 9_0 train order: grid, prid450, viper
def ex_9_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_0: train only then save on each dataset [grid, prid450]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 9_1 train order: prid450, grid, viper
def ex_9_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'viper'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_1: train only then save on each dataset [prid450, grid]. load, retrain and test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 9_2 train order: grid, viper, prid450
def ex_9_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu

        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '3'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_2: train only then save on each dataset [grid, viper]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 9_3 train order: viper, grid, prid450
def ex_9_3():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '0'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid_%s' % a.use_gpu
        a.dataset_test = 'prid450'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_3: train only then save on each dataset [viper, grid]. load, retrain and test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 9_4 train order: viper, prid450, grid
def ex_9_4():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '1'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '1'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '1'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_4: train only then save on each dataset [viper, prid450]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 9_5 train order: prid450, viper, grid
def ex_9_5():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450_%s' % a.use_gpu
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper_%s' % a.use_gpu
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.head_type = 'simple'
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper_%s' % a.use_gpu
        a.dataset_test = 'grid'
        a.ranking_number_test = 100
        a.log_experiment = False
        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 9_5: train only then save on each dataset [prid450, viper]. load, retrain and test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# 4_3, with batchnorm, with neural_distance=concatenation
def ex_4_3_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_3_0: test=viper, train=[grid, prid450], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_3_1():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_3_1: test=grid, train=[viper, prid450], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_3_2():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 4_3_2: test=prid450, train=[viper, grid], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# 5_3	with batch_norm, with neural_distance=concatenate
def ex_5_3_0():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_3_0: test=viper, train=[grid, prid450], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_3_1():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_3_1: test=grid, train=[viper, prid450], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_3_2():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 5_3_2: test=prid450, train=[viper, grid], with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


# 6_3	with batch_norm, neural_distance=concatenate
def ex_6_3_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.datasets_train = ['grid', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'g_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_3_0: train only=[grid, prid450], with batchnorm, neural_d=concatenate, mix then retrain on test=viper'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_3_1():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'grid'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.datasets_train = ['viper', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_3_1: train only=[viper, prid450], with batchnorm, neural_d=concatenate, mix then retrain on test=grid'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


def ex_6_3_2():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'prid450'
    iterations = 20

    for itera in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.datasets_train = ['viper', 'grid']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'v_g_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.use_gpu = '2'
        a.log_file = 'thesis_results_%s.txt' % a.use_gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 100

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 6_3_2: train only=[viper, grid], with batchnorm, neural_d=concatenate, mix then retrain on test=prid450'

    # get the means
    # TODO: debug this, check if it works
    matrix_means = np.mean(all_confusion, axis=0)
    matrix_std = np.std(all_confusion, axis=0)
    ranking_means = np.mean(all_cmc, axis=0)
    ranking_std = np.std(all_cmc, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log(a, a.experiment_name, file_name, name, matrix_means, matrix_std,
                        ranking_means, ranking_std, total_time, None, None)


# ex_11_0, train various networks and save them for priming.
# Train on all datasets, with mixing including test + concatenation
def ex_11_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 11_0: test=viper, train=[grid, prid450, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'all_viper_mix'

    a.datasets_train = ['grid', 'prid450', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_11_1():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 11_1: test=grid, train=[viper, prid450, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'all_grid_mix'

    a.datasets_train = ['viper', 'prid450', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_11_2():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 11_2: test=prid450, train=[viper, grid, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'all_prid450_mix'

    a.datasets_train = ['viper', 'grid', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_11_3():
    a = ProjectVariable()
    a.use_gpu = '2'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 11_3: test=market, train=[viper, prid450, grid, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'all_market_mix'

    a.datasets_train = ['viper', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)
    

def ex_11_4():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 11_4: test=cuhk02, train=[viper, prid450, grid, market], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'all_cuhk02_mix'

    a.datasets_train = ['viper', 'grid', 'prid450', 'market']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# ex_12_0, train various networks and save them for priming.
# Train on single datasets + concatenation
def ex_12_0():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 12_0: test=viper, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'single_viper'

    scn.super_main(a)


def ex_12_1():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 12_1: test=grid, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'single_grid'

    scn.super_main(a)


def ex_12_2():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 12_2: test=prid450, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'single_prid450'

    scn.super_main(a)


def ex_12_3():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 12_3: test=market, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'single_market'

    scn.super_main(a)


def ex_12_4():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 12_4: test=cuhk02, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 1
    a.neural_distance = 'concatenate'

    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'single_cuhk02'

    scn.super_main(a)

# ex_13_0, train various networks baseline concatenation
# Train on single datasets + concatenation
def ex_13_0():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 13_0: test=viper, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    scn.super_main(a)


def ex_13_1():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 13_1: test=grid, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    scn.super_main(a)


def ex_13_2():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 13_2: test=prid450, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    scn.super_main(a)


def ex_13_3():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 13_3: test=market, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.neural_distance = 'concatenate'

    a.dataset_test = 'market'
    a.ranking_number_test = 100

    scn.super_main(a)


def ex_13_4():
    a = ProjectVariable()
    a.use_gpu = '1'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 13_4: test=cuhk02, with batchnorm, neural_d=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.neural_distance = 'concatenate'

    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100

    scn.super_main(a)


# ex_14_0, train various networks on all datasets
# Train on all datasets, with mixing including test + concatenation
def ex_14_0():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 14_0: test=viper, train=[grid, prid450, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    a.datasets_train = ['grid', 'prid450', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_14_1():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 14_1: test=grid, train=[viper, prid450, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'grid'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'prid450', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_14_2():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 14_2: test=prid450, train=[viper, grid, market, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid', 'market', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_14_3():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 14_3: test=market, train=[viper, prid450, grid, cuhk02], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'market'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_14_4():
    a = ProjectVariable()
    a.use_gpu = '3'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 14_4: test=cuhk02, train=[viper, prid450, grid, market], with batchnorm, ' \
                        'neural_d=concatenate'
    a.epochs = 100
    a.iterations = 20
    a.neural_distance = 'concatenate'

    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100

    a.datasets_train = ['viper', 'grid', 'prid450', 'market']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)



def main():
    num = sys.argv[1]
    print(sys.argv)

    # gpu 0
    if num == '10_0_0': ex_10_0_0()
    if num == '10_0_1': ex_10_0_1()
    if num == '6_0_0': ex_6_0_0()
    if num == '6_1_0': ex_6_1_0()
    if num == '7_0': ex_7_0()
    if num == '7_1': ex_7_1()
    if num == '7_2': ex_7_2()
    if num == '1_0_0': ex_1_0_0()
    if num == '1_1_0': ex_1_1_0()
    if num == '1_2_0': ex_1_2_0()
    if num == '1_3_0': ex_1_3_0()
    if num == '1_4_0': ex_1_4_0()
    if num == '1_5_0': ex_1_5_0()
    if num == '4_2_0': ex_4_2_0()
    if num == '4_2_1': ex_4_2_1()
    if num == '4_2_2': ex_4_2_2()
    if num == '9_0': ex_9_0()
    if num == '9_1': ex_9_1()
    if num == '9_2': ex_9_2()
    if num == '4_3_0': ex_4_3_0()
    if num == '4_3_1': ex_4_3_1()
    if num == '4_3_2': ex_4_3_2()
    if num == '11_0': ex_11_0()
    if num == '11_1': ex_11_1()
    if num == '11_2': ex_11_2()
    if num == '11_3': ex_11_3()
    if num == '11_4': ex_11_4()

    # gpu 1
    if num == '10_1_0': ex_10_1_0()
    if num == '10_1_1': ex_10_1_1()
    if num == '6_0_1': ex_6_0_1()
    if num == '6_1_1': ex_6_1_1()
    if num == '7_3': ex_7_3()
    if num == '7_4': ex_7_4()
    if num == '7_5': ex_7_5()
    if num == '1_0_1': ex_1_0_1()
    if num == '1_1_1': ex_1_1_1()
    if num == '1_2_1': ex_1_2_1()
    if num == '1_3_1': ex_1_3_1()
    if num == '1_4_1': ex_1_4_1()
    if num == '1_5_1': ex_1_5_1()
    if num == '5_2_0': ex_5_2_0()
    if num == '5_2_1': ex_5_2_1()
    if num == '5_2_2': ex_5_2_2()
    if num == '9_3': ex_9_3()
    if num == '5_3_0': ex_5_3_0()
    if num == '5_3_1': ex_5_3_1()
    if num == '5_3_2': ex_5_3_2()
    if num == '12_0': ex_12_0()
    if num == '12_1': ex_12_1()
    if num == '12_2': ex_12_2()
    if num == '12_3': ex_12_3()
    if num == '12_4': ex_12_4()
    if num == '13_0': ex_13_0()
    if num == '13_1': ex_13_1()
    if num == '13_2': ex_13_2()
    if num == '13_3': ex_13_3()
    if num == '13_4': ex_13_4()

    # gpu 2
    if num == '10_2_0': ex_10_2_0()
    if num == '10_2_1': ex_10_2_1()
    if num == '6_0_2': ex_6_0_2()
    if num == '6_1_2': ex_6_1_2()
    if num == '8_0': ex_8_0()
    if num == '8_1': ex_8_1()
    if num == '8_2': ex_8_2()
    if num == '1_0_2': ex_1_0_2()
    if num == '1_1_2': ex_1_1_2()
    if num == '1_2_2': ex_1_2_2()
    if num == '1_3_2': ex_1_3_2()
    if num == '1_4_2': ex_1_4_2()
    if num == '1_5_2': ex_1_5_2()
    if num == '6_2_0': ex_6_2_0()
    if num == '6_2_1': ex_6_2_1()
    if num == '6_2_2': ex_6_2_2()
    if num == '9_4': ex_9_4()
    if num == '6_3_0': ex_6_3_0()
    if num == '6_3_1': ex_6_3_1()
    if num == '6_3_2': ex_6_3_2()

    # gpu 3
    if num == '4_0_0': ex_4_0_0()
    if num == '4_0_1': ex_4_0_1()
    if num == '4_0_2': ex_4_0_2()
    if num == '4_1_0': ex_4_1_0()
    if num == '4_1_1': ex_4_1_1()
    if num == '4_1_2': ex_4_1_2()
    if num == '5_0_0': ex_5_0_0()
    if num == '5_0_1': ex_5_0_1()
    if num == '5_0_2': ex_5_0_2()
    if num == '5_1_0': ex_5_1_0()
    if num == '5_1_1': ex_5_1_1()
    if num == '5_1_2': ex_5_1_2()
    if num == '8_3': ex_8_3()
    if num == '8_4': ex_8_4()
    if num == '8_5': ex_8_5()
    if num == '2_0_0': ex_2_0_0()
    if num == '2_0_1': ex_2_0_1()
    if num == '2_0_2': ex_2_0_2()
    if num == '2_1_0': ex_2_1_0()
    if num == '2_1_1': ex_2_1_1()
    if num == '2_1_2': ex_2_1_2()
    if num == '9_5': ex_9_5()
    if num == '10_3_0': ex_10_3_0()
    if num == '10_3_1': ex_10_3_1()
    if num == '14_0': ex_14_0()
    if num == '14_1': ex_14_1()
    if num == '14_2': ex_14_2()
    if num == '14_3': ex_14_3()
    if num == '14_4': ex_14_4()


main()