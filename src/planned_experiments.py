import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn
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
    a.ranking_number_test = 316
    scn.super_main(a)

def ex_0_7_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_1: grid, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 125
    scn.super_main(a)

def ex_0_7_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_2: prid450, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225
    scn.super_main(a)

def ex_0_7_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_3: cuhk01, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 485
    scn.super_main(a)

def ex_0_7_4():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: cuhk02, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 'half'
    scn.super_main(a)

def ex_0_7_5():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: market, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'market'
    a.ranking_number_test = 750
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
# 4	training: mix all datasets including test (note: vital)
#
# 4_0	no batchnorm
def ex_4_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_0: test=viper, train=all-cuhk01, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05
    
    a.dataset_test = 'viper'
    a.ranking_number_test = 316
    
    a.datasets_train = ['market', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]
    
    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_1: test=grid, train=all-cuhk01, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['market', 'viper', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)

def ex_4_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_0_2: test=prid450, train=all-cuhk01, no batchnorm, using selu + alphadropout=0.05'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['market', 'viper', 'grid', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


# 4_1 with batchnorm
def ex_4_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_0: test=viper, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['market', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_1: test=grid, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['market', 'viper', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def ex_4_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 4_1_2: test=prid450, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['market', 'viper', 'grid', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

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
    a.experiment_name = 'experiment 5_0_0: test=viper, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['market', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_1: test=grid, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['market', 'viper', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_2: test=prid450, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['market', 'viper', 'grid', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)

# 5_1	with batch_norm
def ex_5_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_0: test=viper, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['market', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_1: test=grid, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['market', 'viper', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_1_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_1_2: test=prid450, train=all-cuhk01, with batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['market', 'viper', 'grid', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)

# ------------------------------------------------------------------------------------
#
# 6	training: train on all mixed, exclude test. Then retrain trained network on the test. (note: vital)
# so the network will learn on mixed data and then retrain on the target train dataset
#
# 6_0	no batch_norm
def ex_5_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_0: test=viper, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.datasets_train = ['market', 'grid', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_1: test=grid, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.datasets_train = ['market', 'viper', 'prid450', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


def ex_5_0_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 5_0_2: test=prid450, train=all-cuhk01, no batchnorm'
    a.epochs = 100
    a.iterations = 30

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.datasets_train = ['market', 'viper', 'grid', 'cuhk02']
    a.ranking_number_train = [5, 5, 5, 5]

    a.mix = True

    scn.super_main(a)


# 6_1	with batch_norm


# ------------------------------------------------------------------------------------
#
# 7	training: train on all ordered for subset={viper, grid, prid450} (note: vital)
#
# 7_0 train order: grid, prid450, viper
# 7_1 train order: prid450, grid, viper
# 7_2 train order: grid, viper, prid450
# 7_3 train order: viper, grid, prid450
# 7_4 train order: viper, prid450, grid
# 7_5 train order: prid450, viper, grid
#
# ------------------------------------------------------------------------------------
#
# 8   priming (note: vital)
#
# 8_0 TODO
#
#
# ------------------------------------------------------------------------------------
# experiments video -> siamese_cnn_video.py (note: vital)
# ------------------------------------------------------------------------------------
#
# 9	3D convolution vs. cnn_lstm (single dataset)
#
# 9_0	video_head_type=3d_convolution, no batchnorm
# 9_1	video_head_type=3d_convolution, with batchnorm
# 9_2	video_head_type=cnn_lstm
#
# ------------------------------------------------------------------------------------
#
# 10	training: mixing all datasets, including test
#
# 10_0	3d_conv, no batchnorm
# 10_1	3d_conv, with batchnorm
# 10_2	cnn_lstm
#
# ------------------------------------------------------------------------------------
#
# 11 	training: retrain network on test
#
# 11_0	3d_conv, no batchnorm
# 11_1	3d_conv, with batchnorm
# 11_2	cnn_lstm
#
#
#
