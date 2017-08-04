import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn
import project_utils as pu
import numpy as np
# ------------------------------------------------------------------------------------
# experiments images -> siamese_cnn_image.py
# ------------------------------------------------------------------------------------
# 0	CLR vs. LR with decay on [viper, grid, prid450, cuhk01] (note: experiment is not vital)
#
# no CLR
# 0_0	no CLR: lr=0.001, decay=0.95
# 0_1	no CLR: lr=0.0001, decay=0.95
# 0_2	no CLR: lr=0.00001, decay=0.95
# 0_3	no CLR: lr=0.000001, decay=0.95
#
# with CLR (note: only the baseline is vital)
# 0_4	with CLR: min=0.000001, max=0.00001
# 0_5	with CLR: min=0.00001, max=0.0001
# 0_6	with CLR: min=0.0001, max=0.001
# 0_7	with CLR: min=0.00005, max=0.001 [BASELINE]
def ex_0_7_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_0: viper, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    scn.super_main(a)

def ex_0_7_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_1: grid, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    scn.super_main(a)

def ex_0_7_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_2: prid450, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    scn.super_main(a)

def ex_0_7_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_3: cuhk01, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 100
    scn.super_main(a)

def ex_0_7_4():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: cuhk02, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    scn.super_main(a)

def ex_0_7_5():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: market, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    scn.super_main(a)

# ------------------------------------------------------------------------------------
#
# 1	neural layers: type of merging on [viper, grid, prid450, cuhk01] (note: vital)
#
# 1_0	neural_distance=absolute
# note: we already have these in experiments 0_7

# 1_1	neural_distance=subtract
def ex_1_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_0: viper, neural_distance=subtract'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.neural_distance = 'subtract'
    scn.super_main(a)

def ex_1_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_1: grid, neural_distance=subtract'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.neural_distance = 'subtract'
    scn.super_main(a)

def ex_1_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_2: prid450, neural_distance=subtract'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.neural_distance = 'subtract'
    scn.super_main(a)

def ex_1_1_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_1_3: cuhk01, neural_distance=subtract'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.neural_distance = 'subtract'
    scn.super_main(a)

# 1_2	neural_distance=concatenate
def ex_1_2_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_0: viper, neural_distance=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.neural_distance = 'concatenate'
    scn.super_main(a)

def ex_1_2_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_1: grid, neural_distance=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.neural_distance = 'concatenate'
    scn.super_main(a)

def ex_1_2_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_2: prid450, neural_distance=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.neural_distance = 'concatenate'
    scn.super_main(a)

def ex_1_2_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_2_3: cuhk01, neural_distance=concatenate'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.neural_distance = 'concatenate'
    scn.super_main(a)

# 1_3	neural_distance=divide
def ex_1_3_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_0: viper, neural_distance=divide'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.neural_distance = 'divide'
    scn.super_main(a)

def ex_1_3_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_1: grid, neural_distance=divide'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.neural_distance = 'divide'
    scn.super_main(a)

def ex_1_3_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_2: prid450, neural_distance=divide'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.neural_distance = 'divide'
    scn.super_main(a)

def ex_1_3_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_3_3: cuhk01, neural_distance=divide'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.neural_distance = 'divide'
    scn.super_main(a)


# 1_4	neural_distance=add
def ex_1_4_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_0: viper, neural_distance=add'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.neural_distance = 'add'
    scn.super_main(a)

def ex_1_4_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_1: grid, neural_distance=add'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.neural_distance = 'add'
    scn.super_main(a)

def ex_1_4_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_2: prid450, neural_distance=add'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.neural_distance = 'add'
    scn.super_main(a)

def ex_1_4_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_4_3: cuhk01, neural_distance=add'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.neural_distance = 'add'
    scn.super_main(a)

# 1_5	neural_distance=multiply
def ex_1_5_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_0: viper, neural_distance=multiply'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.neural_distance = 'multiply'
    scn.super_main(a)

def ex_1_5_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_1: grid, neural_distance=multiply'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.neural_distance = 'multiply'
    scn.super_main(a)

def ex_1_5_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_2: prid450, neural_distance=multiply'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.neural_distance = 'multiply'
    scn.super_main(a)

def ex_1_5_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 1_5_3: cuhk01, neural_distance=multiply'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.neural_distance = 'multiply'
    scn.super_main(a)

# ------------------------------------------------------------------------------------
#
# 2	non-neural vs. neural on [viper, grid, prid450, cuhk01] (note: vital)
#
# 2_0	cost_module_type=euclidean, lr=0.00001
def ex_2_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_0: viper, cost_module_type=euclidean, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_1: grid, cost_module_type=euclidean, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False    
    scn.super_main(a)

def ex_2_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_2: prid450, cost_module_type=euclidean, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_0_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_0_3: cuhk01, cost_module_type=euclidean, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

# 2_1 cost_module_type=cosine, lr=0.00001
def ex_2_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_0: viper, cost_module_type=cosine, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_1: grid, cost_module_type=cosine, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False    
    scn.super_main(a)

def ex_2_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_2: prid450, cost_module_type=cosine, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

def ex_2_1_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 2_1_3: cuhk01, cost_module_type=cosine, lr=0.00001'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)
#
# ------------------------------------------------------------------------------------
#
# 3	training: single dataset, for all datasets? (note: vital)
#
# 3_0	no batchnorm, using selu + alphadropout=0.05
def ex_3_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 3_0_0: viper, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    
    scn.super_main(a)

def ex_3_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 3_0_1: grid, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
        
    scn.super_main(a)

def ex_3_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 3_0_2: prid450, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    
    scn.super_main(a)

def ex_3_0_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 3_0_3: cuhk01, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    
    scn.super_main(a)


# 3_1   with batchnorm (note: can be found in experiment 0_7)
#
# ------------------------------------------------------------------------------------
#
# 4	training: mix [viper, grid, prid450] including test (note: vital)
#
# 4_0	no batchnorm
def ex_4_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_0: test=viper, train=[grid, prid450], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    
    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]
    
    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_1: test=grid, train=[viper, prid450], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)

def ex_4_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_2: test=prid450, train=[viper, grid], no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# 4_1 with batchnorm
def ex_4_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

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
    a.experiment_name = 'experiment 5_0_0: test=viper, train=[grid, prid450], no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_1: test=grid, train=[viper, prid450], no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_2: test=prid450, train=[viper, grid], no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['viper', 'grid']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)

# 5_1	with batch_norm
def ex_5_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_0: test=viper, train=[grid, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['grid', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_1: test=grid, train=[viper, prid450], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['viper', 'prid450']
    a.ranking_number_train = [5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_2: test=prid450, train=[viper, grid], with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 316

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 125

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 225

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'g_p_mix'

        a.log_experiment = False

        a.dataset_test = 'viper'
        a.ranking_number_test = 316

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'v_p_mix'

        a.log_experiment = False

        a.dataset_test = 'grid'
        a.ranking_number_test = 125

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
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
        a.epochs = 100
        a.iterations = 1

        a.load_weights_name = 'v_g_mix'

        a.log_experiment = False

        a.dataset_test = 'prid450'
        a.ranking_number_test = 225

        confusion, cmc, the_time = scn.super_main(a, get_data=True)

        # store the intermediary results
        # -----------------------------------------------------------------------------------------------------------
        all_confusion.append(confusion)
        all_cmc.append(cmc)
        total_time += the_time

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.dataset_test = 'viper'
        a.ranking_number_test = 316
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.dataset_test = 'viper'
        a.ranking_number_test = 316
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.dataset_test = 'prid450'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.dataset_test = 'prid450'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.dataset_test = 'grid'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.dataset_test = 'grid'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.dataset_test = 'viper'
        a.ranking_number_test = 316
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.dataset_test = 'viper'
        a.ranking_number_test = 316
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.dataset_test = 'prid450'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.datasets_train = ['grid']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'grid'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'grid'
        a.dataset_test = 'prid450'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.dataset_test = 'grid'
        a.ranking_number_test = 225
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
    iterations = 30

    for iter in range(iterations):
        # first train on the datasets A and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.datasets_train = ['prid450']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'prid450'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste B and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'prid450'
        a.datasets_train = ['viper']
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = 'viper'
        a.log_experiment = False
        a.head_type = 'simple'
        a.activation_function = 'selu'
        a.dropout_rate = 0.05
        scn.super_main(a)

        # then load + retrain on dataste C and train + test
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 100
        a.iterations = 1
        a.load_weights_name = 'viper'
        a.dataset_test = 'grid'
        a.ranking_number_test = 225
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
# 9   priming (note: vital)
#
# 9_0 TODO
#
#
# ------------------------------------------------------------------------------------
# experiments video -> siamese_cnn_video.py (note: vital)
# ------------------------------------------------------------------------------------
#
# 10	3D convolution vs. cnn_lstm (single dataset)
#
# 10_0	video_head_type=3d_convolution, no batchnorm
def ex_10_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_0_0: video_head_type=3d_convolution on ilids, no batchnorm'
    a.epochs = 200
    a.iterations = 30
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
    a.experiment_name = 'experiment 10_0_1: video_head_type=3d_convolution on prid2011, no batchnorm'
    a.epochs = 200
    a.iterations = 30
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


# 10_1	video_head_type=3d_convolution, with batchnorm
def ex_10_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_1_0: video_head_type=3d_convolution on ilids, with batchnorm'
    a.epochs = 200
    a.iterations = 30
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]

    srcn.super_main(a)

def ex_10_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_1_1: video_head_type=3d_convolution on prid2011, with batchnorm'
    a.epochs = 200
    a.iterations = 30
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
    a.experiment_name = 'experiment 10_2_0: video_head_type=cnn_lstm on ilids'
    a.epochs = 200
    a.iterations = 30
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = 'cnn_lstm'
    srcn.super_main(a)


def ex_10_2_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_2_1: video_head_type=cnn_lstm on prid2011'
    a.epochs = 200
    a.iterations = 30
    a.dataset_test = 'prid2011'
    a.ranking_number_test = 30
    a.sequence_length = 20
    a.video_head_type = 'cnn_lstm'
    srcn.super_main(a)

# ------------------------------------------------------------------------------------
#
# 11	training: mixing all datasets, including test
#
# 11_0	3d_conv, no batchnorm
# TODO: figure out the sequence number when mixing video data
# draft:
def ex_11_2_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 11_2_0: video_head_type=3d_convolution on ilids'
    a.epochs = 200
    a.iterations = 30
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


# 11_1	3d_conv, with batchnorm
# 11_2	cnn_lstm
#
# ------------------------------------------------------------------------------------
#
# 12 	training: retrain network on test
#
# 12_0	3d_conv, no batchnorm
# 12_1	3d_conv, with batchnorm
# 12_2	cnn_lstm
#
#
#
