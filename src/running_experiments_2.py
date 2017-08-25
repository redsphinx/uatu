import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn
import numpy as np
import project_utils as pu


def test():
    a = ProjectVariable()
    a.experiment_name = 'test'
    a.epochs = 1
    a.iterations = 1
    a.datasets = ['viper']
    scn.super_main(a)


def experiment_000():
    a = ProjectVariable()
    a.experiment_name = 'improved baseline SCNN setup 105, minus caviar: rank=100'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['cuhk02', 'market', 'viper', 'prid450', 'grid']
    scn.super_main(a)


def experiment_001():
    a = ProjectVariable()
    a.experiment_name = 'train on viper only: rank=100'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    scn.super_main(a)


def experiment_002():
    a = ProjectVariable()
    a.experiment_name = 'train on cuhk02 only: rank=100'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['cuhk02']
    scn.super_main(a)


def experiment_003():
    a = ProjectVariable()
    a.experiment_name = 'train on market only: rank=100'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['market']
    scn.super_main(a)


def experiment_004():
    a = ProjectVariable()
    a.experiment_name = 'train on prid450 only: rank=225'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['prid450']
    scn.super_main(a)


def experiment_005():
    a = ProjectVariable()
    a.experiment_name = 'train on caviar only: rank=36'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['caviar']
    scn.super_main(a)


def experiment_006():
    a = ProjectVariable()
    a.experiment_name = 'train on grid only: rank=125'
    a.epochs = 100
    a.iterations = 10
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['grid']
    scn.super_main(a)


def experiment_007():
    a = ProjectVariable()
    a.experiment_name = 'train on viper only: rank=632'
    a.epochs = 100
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    scn.super_main(a)


def experiment_order_viper_2():
    a = ProjectVariable()
    a.experiment_name = 'train on viper'
    a.epochs = 50
    a.iterations = 1
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['viper']
    a.name_indication = 'dataset_name'
    a.batch_size = 32
    scn.super_main(a)


def experiment_order_cuhk02_2():
    a = ProjectVariable()
    a.experiment_name = 'load viper, train on cuhk02'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1339_viper_model.h5'  #
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['cuhk02']
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def rem_k():
    a = ProjectVariable()
    a.experiment_name = 'shit'
    a.epochs = 1
    a.iterations = 1
    a.datasets = ['grid']
    # a.neural_distance = 'add'
    scn.super_main(a)


def something():
    a = ProjectVariable()
    a.experiment_name = 'something'
    a.ranking_number = 2
    a.epochs = 1
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [1]
    a.name_indication = 'dataset_name'
    a.datasets = ['grid']
    scn.super_main(a)

    a.ranking_number = 3
    a.datasets = ['viper']
    a.load_weights_name = 'grid_weigths.h5'
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    a.trainable_bn = False
    a.trainable_cost_module = False
    scn.super_main(a)


def something_prime():
    a = ProjectVariable()
    a.experiment_name = 'something prime'
    a.priming = True
    a.load_model_name = 'grid_model.h5'
    a.load_weights_name = 'grid_weigths.h5'
    a.ranking_number = 2
    a.prime_epochs = 1
    a.batch_size = 32
    a.iterations = 1
    prime.super_main(a)


def experiment_008():
    a = ProjectVariable()
    a.experiment_name = '008. train on cuhk02 only baseline'
    a.datasets = ['cuhk02']
    a.epochs = 100  # rem
    a.iterations = 10  # rem
    a.ranking_number = 100
    scn.super_main(a)


def experiment_009():
    a = ProjectVariable()
    a.experiment_name = '009. train on viper, then cuhk02'
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.datasets = ['cuhk02']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def experiment_010():
    a = ProjectVariable()
    a.experiment_name = '010. train on market only baseline'
    a.datasets = ['market']
    a.ranking_number = 100
    a.epochs = 100  # rem
    a.iterations = 10  # rem
    scn.super_main(a)


def experiment_011():
    a = ProjectVariable()
    a.experiment_name = '011. train on viper then market'
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.datasets = ['market']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def experiment_012():
    a = ProjectVariable()
    a.experiment_name = '012. train on prid450 only baseline'
    a.datasets = ['prid450']
    a.ranking_number = 100
    a.epochs = 100  # rem
    a.iterations = 10  # rem
    scn.super_main(a)


def experiment_013():
    a = ProjectVariable()
    a.experiment_name = '013. train on viper then prid450, train everything'
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def experiment_014():
    a = ProjectVariable()
    a.experiment_name = '014. train on viper then prid450, train classifier only'
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    a.datasets = ['prid450']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def test_it():
    a = ProjectVariable()
    a.experiment_name = ''
    a.ranking_number = 5
    a.epochs = 10
    a.iterations = 1
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)


def experiment_test():
    a = ProjectVariable()
    a.experiment_name = 'shit'
    a.use_gpu = '0'
    a.log_file = 'log_0.txt'
    a.ranking_number = 2
    a.epochs = 1
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [1]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 10
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


#################################################################################################
#    BASELINE
#################################################################################################

def e_001():
    a = ProjectVariable()
    a.experiment_name = '001. baseline viper'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['viper']
    scn.super_main(a)


def e_002():
    a = ProjectVariable()
    a.experiment_name = '002. baseline market'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['market']
    scn.super_main(a)


def e_003():
    a = ProjectVariable()
    a.experiment_name = '003. baseline cuhk02'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['cuhk02']
    scn.super_main(a)


def e_004():
    a = ProjectVariable()
    a.experiment_name = '004. baseline grid'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['grid']
    scn.super_main(a)


def e_005():
    a = ProjectVariable()
    a.experiment_name = '005. baseline prid450'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['prid450']
    scn.super_main(a)


def e_006():
    a = ProjectVariable()
    a.experiment_name = '006. baseline caviar'
    a.ranking_number = 36
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['caviar']
    scn.super_main(a)


#################################################################################################
#    TRAIN ON VIPER, FULL NETWORK
#################################################################################################

def e_007():
    a = ProjectVariable()
    a.experiment_name = '007. train on viper -> market (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['market']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_008():
    a = ProjectVariable()
    a.experiment_name = '008. train on viper -> cuhk02 (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['cuhk02']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_009():
    a = ProjectVariable()
    a.experiment_name = '009. train on viper -> grid (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_010():
    a = ProjectVariable()
    a.experiment_name = '010. train on viper -> prid450 (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_011():
    a = ProjectVariable()
    a.experiment_name = '011. train on viper -> caviar (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


#################################################################################################
#    TRAIN ON VIPER, ONLY CLASSIFIER
#################################################################################################

def e_012():
    a = ProjectVariable()
    a.experiment_name = '012. train on viper -> market (classifier only)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['market']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_013():
    a = ProjectVariable()
    a.experiment_name = '013. train on viper -> cuhk02 (classifier only)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['cuhk02']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_014():
    a = ProjectVariable()
    a.experiment_name = '014. train on viper -> grid (classifier only)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_015():
    a = ProjectVariable()
    a.experiment_name = '015. train on viper -> prid450 (classifier only)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_016():
    a = ProjectVariable()
    a.experiment_name = '016. train on viper -> caviar (classifier only)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 2
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


#################################################################################################
#    TRAIN ON CUHK02, FULL NETWORK
#################################################################################################

def e_017():
    a = ProjectVariable()
    a.experiment_name = '017. train on cuhk02 -> market (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['market']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_018():
    a = ProjectVariable()
    a.experiment_name = '018. train on cuhk02 -> viper (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['viper']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_019():
    a = ProjectVariable()
    a.experiment_name = '019. train on cuhk02 -> grid (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_020():
    a = ProjectVariable()
    a.experiment_name = '020. train on cuhk02 -> prid450 (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_021():
    a = ProjectVariable()
    a.experiment_name = '021. train on cuhk02 -> caviar (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


#################################################################################################
#    TRAIN ON CUHK02, ONLY CLASSIFIER
#################################################################################################

def e_022():
    a = ProjectVariable()
    a.experiment_name = '022. train on cuhk02 -> market (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['market']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_023():
    a = ProjectVariable()
    a.experiment_name = '023. train on cuhk02 -> viper (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['viper']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_024():
    a = ProjectVariable()
    a.experiment_name = '024. train on cuhk02 -> grid (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_025():
    a = ProjectVariable()
    a.experiment_name = '025. train on cuhk02 -> prid450 (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_026():
    a = ProjectVariable()
    a.experiment_name = '026. train on cuhk02 -> caviar (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


#################################################################################################
#    TRAIN ON CUHK02, MARKET, FULL NETWORK
#################################################################################################

def e_027():
    a = ProjectVariable()
    a.experiment_name = '027. train on cuhk02, market -> viper (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['cuhk02']
    a.log_experiment = False
    scn.super_main(a)

    a.ranking_number = 2
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.datasets = ['market']
    scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['viper']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_028():
    a = ProjectVariable()
    a.experiment_name = '028. train on cuhk02, market -> grid (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_029():
    a = ProjectVariable()
    a.experiment_name = '029. train on cuhk02, market -> prid450 (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_030():
    a = ProjectVariable()
    a.experiment_name = '030. train on cuhk02, market -> caviar (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


#################################################################################################
#    TRAIN ON CUHK02, MARKET, ONLY CLASSIFIER
#################################################################################################

def e_031():
    a = ProjectVariable()
    a.experiment_name = '031. train on cuhk02, market -> viper (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['viper']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_032():
    a = ProjectVariable()
    a.experiment_name = '032. train on cuhk02, market -> grid (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['grid']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_033():
    a = ProjectVariable()
    a.experiment_name = '033. train on cuhk02, market -> prid450 (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 100
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['prid450']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_034():
    a = ProjectVariable()
    a.experiment_name = '034. train on cuhk02, market -> caviar (only classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 5
    a.epochs = 100
    a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [100]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['cuhk02']
    # a.log_experiment = False
    # scn.super_main(a)

    # a.ranking_number = 2
    # a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    # a.datasets = ['market']
    # scn.super_main(a)

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_035():
    a = ProjectVariable()
    a.experiment_name = '035. train on market and save, then prime (full network)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 100
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [1]
    a.name_indication = 'dataset_name'
    a.datasets = ['market']
    scn.super_main(a)

    a.save_inbetween = False
    a.priming = True
    a.load_model_name = 'market_model_%s.h5' % a.use_gpu
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.datasets = ['market']
    a.prime_epochs = 5
    a.batch_size = 32
    a.iterations = 1
    prime.super_main(a)


def e_036():
    a = ProjectVariable()
    a.experiment_name = '036. train on market and save, then prime (only_classifier)'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 100
    # a.epochs = 100
    # a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [1]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['market']
    # scn.super_main(a)

    a.save_inbetween = False
    a.priming = True
    a.load_model_name = 'market_model_%s.h5' % a.use_gpu
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.datasets = ['market']
    a.prime_epochs = 5
    a.batch_size = 32
    a.iterations = 1
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    prime.super_main(a)


def e_037():
    a = ProjectVariable()
    a.experiment_name = '037. train on market and save, then prime (full network), prime_epochs=10'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    # a.ranking_number = 100
    # a.epochs = 100
    # a.iterations = 1
    # a.save_inbetween = True
    # a.save_points = [1]
    # a.name_indication = 'dataset_name'
    # a.datasets = ['market']
    # scn.super_main(a)

    a.save_inbetween = False
    a.priming = True
    a.load_model_name = 'market_model_%s.h5' % a.use_gpu
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.datasets = ['market']
    a.prime_epochs = 10
    a.batch_size = 32
    a.iterations = 1
    prime.super_main(a)


def e_038():
    a = ProjectVariable()
    a.experiment_name = '038. train on cuhk02, market, save'
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'

    a.ranking_number = 2
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    a.datasets = ['market']
    scn.super_main(a)


def e_039():
    a = ProjectVariable()
    a.experiment_name = '039. train on cuhk02, market, save -> viper (full network), rank=316'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 316
    a.iterations = 5
    a.datasets = ['viper']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_040():
    a = ProjectVariable()
    a.experiment_name = '040. train on cuhk02, market, save -> prid450 (full network), rank=225'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 225
    a.iterations = 5
    a.datasets = ['prid450']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_041():
    a = ProjectVariable()
    a.experiment_name = '041. train on cuhk02, market, save -> grid (full network), rank=125'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 125
    a.iterations = 5
    a.datasets = ['grid']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_042():
    a = ProjectVariable()
    a.experiment_name = '042. train on cuhk02, market, save -> caviar (full network), rank=36'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.ranking_number = 36
    a.iterations = 5
    a.datasets = ['caviar']
    a.load_weights_name = 'cuhk02_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_043():
    a = ProjectVariable()
    a.experiment_name = '043. sanity test. train on viper only: rank=316'
    a.ranking_number = 316
    a.iterations = 5
    a.datasets = ['viper']
    scn.super_main(a)


def e_044():
    a = ProjectVariable()
    a.experiment_name = '044. sanity test. train on prid450 only: rank=225'
    a.ranking_number = 225
    a.iterations = 5
    a.datasets = ['prid450']
    scn.super_main(a)


def e_045():
    a = ProjectVariable()
    a.experiment_name = '045. sanity test. train on grid only: rank=125'
    a.ranking_number = 125
    a.iterations = 5
    a.datasets = ['grid']
    scn.super_main(a)


def e_046():
    a = ProjectVariable()
    a.experiment_name = '046. sanity test. train on caviar only: rank=36'
    a.ranking_number = 36
    a.iterations = 5
    a.datasets = ['caviar']
    scn.super_main(a)


def e_047():
    a = ProjectVariable()
    a.experiment_name = '047. train on viper only: rank=316, euclidean, no CLR'
    a.ranking_number = 316
    a.iterations = 5
    # a.epochs = 100
    a.datasets = ['viper']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False

    scn.super_main(a)


def e_048():
    a = ProjectVariable()
    a.experiment_name = '048. train on prid450 only: rank=225, euclidean, no CLR'
    a.ranking_number = 225
    a.iterations = 5
    # a.epochs = 100
    a.datasets = ['prid450']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False

    scn.super_main(a)


def e_049():
    a = ProjectVariable()
    a.experiment_name = '048. train on grid only: rank=125, euclidean, no CLR'
    a.ranking_number = 125
    a.iterations = 5
    # a.epochs = 100
    a.datasets = ['grid']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False

    scn.super_main(a)


def e_050():
    a = ProjectVariable()
    a.experiment_name = '050. train on caviar only: rank=36, euclidean, no CLR'
    a.ranking_number = 36
    a.iterations = 5
    # a.epochs = 100
    a.datasets = ['caviar']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False

    scn.super_main(a)


def e_051():
    a = ProjectVariable()
    a.experiment_name = '051. viper. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['viper']
    scn.super_main(a)


def e_052():
    a = ProjectVariable()
    a.experiment_name = '052. market. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['market']
    scn.super_main(a)


def e_053():
    a = ProjectVariable()
    a.experiment_name = '053. cuhk02. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['cuhk02']
    scn.super_main(a)


def e_054():
    a = ProjectVariable()
    a.experiment_name = '054. grid. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['grid']
    scn.super_main(a)


def e_055():
    a = ProjectVariable()
    a.experiment_name = '055. prid450. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['prid450']
    scn.super_main(a)


def e_056():
    a = ProjectVariable()
    a.experiment_name = '056. caviar. exp with 1 dataset only, rank=20, epoch=40'
    a.ranking_number = 20
    a.iterations = 5
    a.epochs = 40
    a.datasets = ['caviar']
    scn.super_main(a)


def e_1():
    a = ProjectVariable()
    a.experiment_name = '1. train and save viper model'
    a.ranking_number = 100
    a.iterations = 1
    a.epochs = 100
    a.datasets = ['viper']
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def e_2():
    a = ProjectVariable()
    a.experiment_name = '2. train and save cuhk02 model'
    a.ranking_number = 100
    a.iterations = 1
    a.epochs = 100
    a.datasets = ['cuhk02']
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def e_3():
    a = ProjectVariable()
    a.experiment_name = '3. train and save market model'
    a.ranking_number = 100
    a.iterations = 1
    a.epochs = 100
    a.datasets = ['market']
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def experiment_lstm():
    a = ProjectVariable()
    a.experiment_name = 'trying out LSTM'
    a.ranking_number = 10
    a.epochs = 10
    a.iterations = 1

    a.datasets = ['ilids-vid']
    srcn.super_main(a)

    # a.datasets = ['ilids-vid']
    # srcn.super_main(a)


def experiment_lstm_more():
    a = ProjectVariable()
    a.experiment_name = 'trying out LSTM more'
    a.ranking_number = 30
    a.epochs = 1
    a.iterations = 3

    a.datasets = ['ilids-vid']
    srcn.super_main(a)


def experiment_lstm_moore():
    a = ProjectVariable()
    a.experiment_name = 'trying out LSTM moore'
    a.ranking_number = 30
    a.epochs = 50
    a.iterations = 1

    a.datasets = ['ilids-vid']
    srcn.super_main(a)


def experiment_lstm_mooore():
    a = ProjectVariable()
    a.experiment_name = 'trying out LSTM mooore'
    a.ranking_number = 30
    a.epochs = 100
    a.iterations = 1

    a.datasets = ['ilids-vid']
    srcn.super_main(a)


def normal_scnn():
    a = ProjectVariable()
    a.experiment_name = 'normal scnn to check for stuff'
    a.ranking_number = 10
    a.epochs = 1
    a.iterations = 1

    a.datasets = ['viper']
    scn.super_main(a)


def testing_cosine():
    a = ProjectVariable()
    a.experiment_name = 'testing cosine'
    a.ranking_number = 20
    a.iterations = 3
    a.epochs = 50
    a.datasets = ['prid450']
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def recreate_sota_1():
    a = ProjectVariable()
    a.experiment_name = 'try to recreate viper sota: set to keras + using merge instead of Lambda + pooling size'
    a.epochs = 100
    a.ranking_number = 316
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    a.pooling_size = [[2, 2], [2, 2]]
    scn.super_main(a)


def recreate_sota_2():
    a = ProjectVariable()
    a.experiment_name = 'try to recreate viper sota: set to tf + using Lambda + pooling size'
    a.epochs = 100
    a.ranking_number = 316
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    a.pooling_size = [[2, 2], [2, 2]]
    scn.super_main(a)


def recreate_sota_3():
    a = ProjectVariable()
    a.experiment_name = 'try to recreate viper sota: set to tf + using Lambda + pooling size + numfil=2'
    a.epochs = 100
    a.ranking_number = 316
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 2
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    a.pooling_size = [[2, 2], [2, 2]]
    scn.super_main(a)


def recreate_sota_4():
    a = ProjectVariable()
    a.experiment_name = 'try to recreate viper sota: set to tf + using Lambda + numfil=2'
    a.epochs = 100
    a.ranking_number = 316
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 2
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    scn.super_main(a)


def recreate_sota_5():
    a = ProjectVariable()
    a.experiment_name = 'try to recreate viper sota: set to keras only 30 iters'
    a.epochs = 100
    a.ranking_number = 316
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    scn.super_main(a)


def test_siamese_video_1():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: 3D convolutions on ilids-vid'
    a.epochs = 2
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.datasets = ['ilids-vid']
    a.video_head_type = '3d_convolution'
    a.sequence_length = 22
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    # a.dropout_rate = 0.5
    # a.lstm_units = 64
    srcn.super_main(a)


def test_siamese_video_2():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on ilids-vid'
    a.epochs = 2
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    # a.kernel = (3, 3, 3)
    # a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.lstm_units = 64
    srcn.super_main(a)


def test_siamese_video_3():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: 3D convolutions on prid2011'
    a.epochs = 2
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.datasets = ['prid2011']
    a.video_head_type = '3d_convolution'
    a.sequence_length = 20
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    # a.dropout_rate = 0.5
    # a.lstm_units = 64
    srcn.super_main(a)


def test_siamese_video_4():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on prid2011'
    a.epochs = 2
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['prid2011']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 20
    # a.kernel = (3, 3, 3)
    # a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.lstm_units = 32
    srcn.super_main(a)


def scnn_fix_confusion_matrix():
    a = ProjectVariable()
    a.experiment_name = 'towards fixing confusion matrix'
    a.epochs = 10
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [10]
    a.ranking_number = 50
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    scn.super_main(a)


def shit_priming():
    a = ProjectVariable()
    a.experiment_name = 'priming shit'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.priming = True
    a.ranking_number = 50
    # a.load_model_name = 'viper_model_%s.h5' % a.use_gpu
    a.load_weights_name = 'viper_weigths_%s.h5' % a.use_gpu
    a.datasets = ['viper']
    a.prime_epochs = 10
    a.batch_size = 32
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.000001
    a.iterations = 1
    prime.super_main(a)


def test_siamese_video_5():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on ilids-vid, no CLR'
    a.epochs = 10
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.optimizer = 'nadam'
    a.lstm_units = 512
    srcn.super_main(a)


def test_siamese_video_6():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on ilids-vid, no CLR + L2 regularizer in dense layers'
    a.epochs = 30
    a.iterations = 5
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.optimizer = 'nadam'
    a.lstm_units = 512
    srcn.super_main(a)


def test_siamese_video_7():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on ilids-vid, clr 0.00001-0.00005'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.05
    # a.cl_min = 0.000001
    # a.cl_max = 0.00005
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.optimizer = 'nadam'
    a.lstm_units = 512
    srcn.super_main(a)


def test_siamese_video_8():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on prid2011, clr 0.00001-0.00005'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['prid2011']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 20
    a.ranking_number = 30
    a.dropout_rate = 0.05
    # a.cl_min = 0.000001
    # a.cl_max = 0.00005
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.optimizer = 'nadam'
    a.lstm_units = 512
    srcn.super_main(a)


def test_siamese_video_9():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: cnn_lstm on ilids-vid, clr 0.00001-0.00005'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.05
    # a.cl_min = 0.000001
    # a.cl_max = 0.00005
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.optimizer = 'nadam'
    a.lstm_units = 1024
    srcn.super_main(a)


def saving_scnn_image_weights():
    a = ProjectVariable()
    a.experiment_name = 'saving viper weights'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.activation_function = 'elu'
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    a.log_file = 'log_0.txt'
    scn.super_main(a)


def test_new_cosine():
    a = ProjectVariable()
    a.experiment_name = 'testing normalized cosine distance'
    a.epochs = 100
    a.ranking_number = 125
    a.iterations = 1
    a.cost_module_type = 'cosine'
    a.activation_function = 'relu'
    a.neural_distance = 'absolute'
    a.datasets = ['grid']
    a.log_file = 'log_0.txt'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def test_cnn():
    a = ProjectVariable()
    a.experiment_name = 'test cnn'
    a.epochs = 10
    a.iterations = 2
    a.batch_size = 32
    a.datasets = ['inria']
    cnn.super_main(a)


def test_cnn_with_save():
    a = ProjectVariable()
    a.experiment_name = 'test cnn and save'
    a.epochs = 100
    a.batch_size = 32
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.datasets = ['inria']
    cnn.super_main(a)


def test_pipeline_1():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 1: train + test single dataset, no saving, 2 iterations'
    a.iterations = 2
    a.epochs = 3

    a.dataset_test = 'viper'
    a.ranking_number_test = 100

    scn.super_main(a)


def test_pipeline_2():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 2: train + test multiple datasets, mix==False, no saving, 2 iterations'
    a.iterations = 2
    a.epochs = 3

    a.datasets_train = ['prid450', 'grid']
    a.dataset_test = 'viper'
    a.ranking_number_train = [50, 50]
    a.ranking_number_test = 50

    scn.super_main(a)


def test_pipeline_3():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 3: train + test multiple datasets, mix==True, mix_with_test=False, no saving, 2 iterations'
    a.iterations = 2
    a.epochs = 3

    a.datasets_train = ['prid450', 'grid']
    a.dataset_test = 'viper'
    a.ranking_number_train = [50, 50]
    a.ranking_number_test = 50
    a.mix = True

    scn.super_main(a)


def test_pipeline_4():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 4: train + test multiple datasets, mix==True, mix_with_test=True, no saving, 2 iterations'
    a.iterations = 2
    a.epochs = 3

    a.datasets_train = ['prid450', 'grid']
    a.dataset_test = 'viper'
    a.ranking_number_train = [50, 50]
    a.ranking_number_test = 50
    a.mix = True
    a.mix_with_test = True

    scn.super_main(a)


def test_pipeline_5():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 5: only test, make new ranking file, no saving, 2 iterations'
    a.iterations = 2
    a.epochs = 1

    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.only_test = True

    scn.super_main(a)


def test_pipeline_6():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 6: only train on multiple datasets, no mixing, no saving, 1 iteration'
    a.iterations = 1
    a.epochs = 10

    a.datasets_train = ['viper', 'grid']
    a.log_experiment = False

    scn.super_main(a)


def test_pipeline_7():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 7: only train on multiple datasets, with mixing, no saving, 1 iteration'
    a.iterations = 1
    a.epochs = 10

    a.datasets_train = ['viper', 'grid']
    a.mix = True
    a.log_experiment = False

    scn.super_main(a)


def test_pipeline_8():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 8: only train on single dataset, no mixing, no saving, 1 iteration'
    a.iterations = 1
    a.epochs = 10

    a.datasets_train = ['viper']
    # a.mix = True
    # a.log_experiment = False

    scn.super_main(a)


def test_pipeline_9():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 9: only train on single dataset, with mixing, no saving, 1 iteration'
    a.iterations = 1
    a.epochs = 10

    a.datasets_train = ['viper']
    a.mix = True
    # a.log_experiment = False

    scn.super_main(a)


def test_pipeline_10():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 10: train only on viper then save'
    a.iterations = 1
    a.save_inbetween = True
    a.epochs = 2
    a.save_points = [2]
    # TODO: fix a.name_indication = 'dataset_name'
    a.name_indication = 'epoch'

    a.datasets_train = ['viper']
    scn.super_main(a)


def test_pipeline_11():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 11: load weights viper and train only on grid'
    a.iterations = 1
    a.load_weights_name = 'scnn_26072017_1735_epoch_2_weigths.h5'
    a.epochs = 2

    a.datasets_train = ['cuhk02']
    scn.super_main(a)


def test_pipeline_12():
    a = ProjectVariable()
    a.experiment_name = 'test pipeline 11: load weights viper and test only on grid'
    a.iterations = 1
    a.load_weights_name = 'scnn_26072017_1735_epoch_2_weigths.h5'
    a.ranking_number_test = 10

    a.dataset_test = 'grid'
    a.only_test = True
    scn.super_main(a)


def mixing_no_batchnorm_1():
    a = ProjectVariable()
    a.experiment_name = 'mixing no BN 1: train on viper, grid, market'
    a.iterations = 1
    a.epochs = 100
    a.save_inbetween = True
    a.save_points = [100]
    # TODO: fix a.name_indication = 'dataset_name'
    a.name_indication = 'epoch'

    a.datasets_train = ['viper', 'grid', 'market']
    a.mix = True

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def mixing_no_batchnorm_2():
    a = ProjectVariable()
    a.experiment_name = 'mixing no BN 2: train on viper, grid, market and prid450 + test + mix_with_test=True'
    a.iterations = 5
    a.epochs = 100
    # a.save_inbetween = True
    # a.save_points = [100]
    # # TODO: fix a.name_indication = 'dataset_name'
    # a.name_indication = 'epoch'

    a.datasets_train = ['viper', 'grid', 'market']
    a.ranking_number_train = [10, 10, 10]
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.mix = True
    a.mix_with_test = True

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def mixing_no_batchnorm_3():
    a = ProjectVariable()
    a.experiment_name = 'mixing no BN 3: train on viper, grid, market and prid450 + test + mix_with_test=False, batch_size=128'
    a.iterations = 10
    a.epochs = 100
    a.batch_size = 128
    # a.save_inbetween = True
    # a.save_points = [100]
    # # TODO: fix a.name_indication = 'dataset_name'
    # a.name_indication = 'epoch'

    a.datasets_train = ['viper', 'grid', 'market']
    a.ranking_number_train = [10, 10, 10]
    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.mix = True

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def mixing_no_batchnorm_4():
    a = ProjectVariable()
    a.experiment_name = 'mixing no BN 4: retrain network from exp. `mixing no BN 1` on prid450'
    a.iterations = 10
    a.epochs = 100
    a.batch_size = 32

    a.load_weights_name = 'scnn_26072017_1834_epoch_100_weigths.h5'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def order_many_datasets_1():
    a = ProjectVariable()
    a.experiment_name = 'order many datasets 1: train on viper, grid, market in order + no batchnorm'
    a.iterations = 1
    a.epochs = 100
    a.save_inbetween = True
    a.save_points = [100]
    # TODO: fix a.name_indication = 'dataset_name'
    a.name_indication = 'epoch'

    a.datasets_train = ['viper', 'grid', 'market']

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def order_many_datasets_2():
    a = ProjectVariable()
    a.experiment_name = 'order on many datasets 2: retrain network from exp. `order many datasets 1` on prid450'
    a.iterations = 10
    a.epochs = 100
    a.batch_size = 32

    a.load_weights_name = 'scnn_27072017_1135_epoch_100_weigths.h5'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


def test_video():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video'
    a.epochs = 2
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'

    # a.dataset_test = 'ilids-vid'
    # a.ranking_number_test = 10

    a.datasets_train = ['prid2011']
    a.ranking_number_train = [10]

    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 20
    # a.kernel = (3, 3, 3)
    # a.pooling_size = [[1, 4, 2], [1, 2, 2]]

    a.dropout_rate = 0.05
    a.head_type = 'simple'
    # a.lstm_units = 64
    srcn.super_main(a)


def test_cuhk02():
    a = ProjectVariable()
    a.experiment_name = 'test market train + test'
    a.iterations = 1
    a.epochs = 1
    a.batch_size = 128

    a.dataset_test = 'market'
    a.ranking_number_test = 'half'

    # a.head_type = 'simple'
    # a.activation_function = 'selu'
    # a.dropout_rate = 0.05

    scn.super_main(a)


def test_market_1():
    a = ProjectVariable()
    a.experiment_name = 'test market 1: train only on all datasets, mix=True, no batchnorm, save'
    a.iterations = 1
    a.epochs = 200
    a.batch_size = 128

    a.datasets_train = ['grid', 'viper', 'cuhk02', 'prid450']
    a.mix = True

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    a.save_inbetween = True
    a.save_points = [50, 100, 150, 200]

    scn.super_main(a)


def test_market_2():
    a = ProjectVariable()
    a.experiment_name = 'test market 2: load model weights from exp. `test market 1`, train + test on market'
    a.load_weights_name = 'scnn_27072017_2030_epoch_200_weigths.h5'
    a.epochs = 100
    a.iterations = 10

    a.dataset_test = 'market'
    a.ranking_number_test = 'half'

    a.head_type = 'simple'
    a.activation_function = 'selu'
    a.dropout_rate = 0.05

    scn.super_main(a)


# --------------------------------------------------------------------------------------------

# 0	CLR vs. LR with decay
#
# no CLR
# 0_0	no CLR: lr=0.001, decay=0.95
def ex_000():
    a = ProjectVariable()
    a.experiment_name = 'experiment 000: viper, no CLR: lr=0.001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.001

    scn.super_main(a)

# 0_1	no CLR: lr=0.0001, decay=0.95
def ex_001():
    a = ProjectVariable()
    a.experiment_name = 'experiment 001: viper, no CLR: lr=0.0001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.0001

    scn.super_main(a)

# 0_2	no CLR: lr=0.00001, decay=0.95
def ex_002():
    a = ProjectVariable()
    a.experiment_name = 'experiment 002: viper, no CLR: lr=0.00001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001

    scn.super_main(a)

# 0_3	no CLR: lr=0.000001, decay=0.95
def ex_003():
    a = ProjectVariable()
    a.experiment_name = 'experiment 003: viper, no CLR: lr=0.000001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.000001

    scn.super_main(a)
#
# with CLR
# 0_4	with CLR: min=0.000001, max=0.00001
def ex_004():
    a = ProjectVariable()
    a.experiment_name = 'experiment 004: viper, with CLR: min=0.000001, max=0.00001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.cl_min = 0.000001
    a.cl_max = 0.00001

    scn.super_main(a)

# 0_5	with CLR: min=0.00001, max=0.0001
def ex_005():
    a = ProjectVariable()
    a.experiment_name = 'experiment 005: viper, with CLR: min=0.00001, max=0.0001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.cl_min = 0.00001
    a.cl_max = 0.0001

    scn.super_main(a)

# 0_6	with CLR: min=0.0001, max=0.001
def ex_006():
    a = ProjectVariable()
    a.experiment_name = 'experiment 006: viper, with CLR: min=0.0001, max=0.001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.cl_min = 0.0001
    a.cl_max = 0.001

    scn.super_main(a)

# 0_7	with CLR: min=0.00005, max=0.001 [BASELINE]
def ex_007():
    a = ProjectVariable()
    a.experiment_name = 'experiment 007: with CLR: min=0.00005, max=0.001 [BASELINE]'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'viper'
    a.ranking_number_test = 316

    a.cl_min = 0.00005
    a.cl_max = 0.001

    scn.super_main(a)


# --------------------------------------------------------------------------------------------

# 0	CLR vs. LR with decay
#
# no CLR
# 0_0	no CLR: lr=0.001, decay=0.95
def ex_008():
    a = ProjectVariable()
    a.experiment_name = 'experiment 008: grid, no CLR: lr=0.001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.001

    scn.super_main(a)

# 0_1	no CLR: lr=0.0001, decay=0.95
def ex_009():
    a = ProjectVariable()
    a.experiment_name = 'experiment 009: grid, no CLR: lr=0.0001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.0001

    scn.super_main(a)

# 0_2	no CLR: lr=0.00001, decay=0.95
def ex_010():
    a = ProjectVariable()
    a.experiment_name = 'experiment 010: grid, no CLR: lr=0.00001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001

    scn.super_main(a)

# 0_3	no CLR: lr=0.000001, decay=0.95
def ex_011():
    a = ProjectVariable()
    a.experiment_name = 'experiment 011: grid, no CLR: lr=0.000001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.000001

    scn.super_main(a)
#
# with CLR
# 0_4	with CLR: min=0.000001, max=0.00001
def ex_012():
    a = ProjectVariable()
    a.experiment_name = 'experiment 012: grid, with CLR: min=0.000001, max=0.00001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.cl_min = 0.000001
    a.cl_max = 0.00001

    scn.super_main(a)

# 0_5	with CLR: min=0.00001, max=0.0001
def ex_013():
    a = ProjectVariable()
    a.experiment_name = 'experiment 013: grid, with CLR: min=0.00001, max=0.0001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.cl_min = 0.00001
    a.cl_max = 0.0001

    scn.super_main(a)

# 0_6	with CLR: min=0.0001, max=0.001
def ex_014():
    a = ProjectVariable()
    a.experiment_name = 'experiment 014: grid, with CLR: min=0.0001, max=0.001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.cl_min = 0.0001
    a.cl_max = 0.001

    scn.super_main(a)

# 0_7	with CLR: min=0.00005, max=0.001 [BASELINE]
def ex_015():
    a = ProjectVariable()
    a.experiment_name = 'experiment 015: with CLR: min=0.00005, max=0.001 [BASELINE]'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'grid'
    a.ranking_number_test = 125

    a.cl_min = 0.00005
    a.cl_max = 0.001

    scn.super_main(a)


# --------------------------------------------------------------------------------------------

# 0	CLR vs. LR with decay
#
# no CLR
# 0_0	no CLR: lr=0.001, decay=0.95
def ex_016():
    a = ProjectVariable()
    a.experiment_name = 'experiment 016: prid450, no CLR: lr=0.001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.001

    scn.super_main(a)

# 0_1	no CLR: lr=0.0001, decay=0.95
def ex_017():
    a = ProjectVariable()
    a.experiment_name = 'experiment 017: prid450, no CLR: lr=0.0001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.0001

    scn.super_main(a)

# 0_2	no CLR: lr=0.00001, decay=0.95
def ex_018():
    a = ProjectVariable()
    a.experiment_name = 'experiment 018: prid450, no CLR: lr=0.00001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001

    scn.super_main(a)

# 0_3	no CLR: lr=0.000001, decay=0.95
def ex_019():
    a = ProjectVariable()
    a.experiment_name = 'experiment 019: prid450, no CLR: lr=0.000001, decay=0.95'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.000001

    scn.super_main(a)
#
# with CLR
# 0_4	with CLR: min=0.000001, max=0.00001
def ex_020():
    a = ProjectVariable()
    a.experiment_name = 'experiment 020: prid450, with CLR: min=0.000001, max=0.00001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.cl_min = 0.000001
    a.cl_max = 0.00001

    scn.super_main(a)

# 0_5	with CLR: min=0.00001, max=0.0001
def ex_021():
    a = ProjectVariable()
    a.experiment_name = 'experiment 021: prid450, with CLR: min=0.00001, max=0.0001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.cl_min = 0.00001
    a.cl_max = 0.0001

    scn.super_main(a)

# 0_6	with CLR: min=0.0001, max=0.001
def ex_022():
    a = ProjectVariable()
    a.experiment_name = 'experiment 022: prid450, with CLR: min=0.0001, max=0.001'
    a.epochs = 100
    a.iterations = 30

    a.dataset_test = 'prid450'
    a.ranking_number_test = 225

    a.cl_min = 0.0001
    a.cl_max = 0.001

    scn.super_main(a)

# 0_7	with CLR: min=0.00005, max=0.001 [BASELINE]
def ex_023():
    a = ProjectVariable()
    a.experiment_name = 'experiment 023: with CLR: min=0.00005, max=0.001 [BASELINE]'
    a.epochs = 100
    a.iterations = 1

    a.dataset_test = 'prid450'
    a.ranking_number_test = 10

    # a.cl_min = 0.00005
    # a.cl_max = 0.001

    a.log_experiment = False

    scn.super_main(a)


def test_sideshuf():
    a = ProjectVariable()
    a.experiment_name = 'side shuffle activated'
    a.iterations = 10

    a.dataset_test = 'prid450'
    a.ranking_number_test = 10

    scn.super_main(a)


def test_sideshuf_idxstart():
    a = ProjectVariable()
    a.experiment_name = 'side shuffle activated + index_start'
    a.iterations = 10

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    scn.super_main(a)


# 10_1	video_head_type=3d_convolution, with batchnorm
def ex_10_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_1_0: video_head_type=3d_convolution on ilids, with batchnorm'
    a.epochs = 10
    a.iterations = 1
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)


# 6_0	no batch_norm
def ex_6_0_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 2

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 10
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
        a.epochs = 10
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


# 6_1	with batch_norm
def ex_6_1_0():
    all_confusion = []
    all_cmc = []
    total_time = 0
    name = 'viper'
    iterations = 2

    for iter in range(iterations):
        # first train on the datasets and save
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 2
        a.iterations = 1

        a.datasets_train = ['grid', 'prid450']
        a.mix = True

        a.save_inbetween = True
        a.save_points = [2]
        a.name_of_saved_file = 'g_p_mix'

        a.log_experiment = False

        scn.super_main(a)

        # then load + retrain
        # -----------------------------------------------------------------------------------------------------------
        a = ProjectVariable()
        a.epochs = 2
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


def save_4_priming():
    # first train model and save
    a = ProjectVariable()
    a.log_experiment = False
    # a.save_inbetween = True
    # a.save_points = [50]
    # a.name_of_saved_file = 'priming_on_prid450'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 10
    a.epochs = 10
    a.iterations = 1

    scn.super_main(a)

def priming():
    a = ProjectVariable()
    a.experiment_name = 'see if priming is working'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.priming = True
    a.load_weights_name = 'priming_on_prid450'
    a.dataset_test = 'prid450'
    a.prime_epochs = 10
    a.use_cyclical_learning_rate = False
    a.learning_rate = 0.00001
    a.iterations = 1
    a.only_test = True
    prime.super_main(a)


def test_augmenting():
    a = ProjectVariable()
    a.experiment_name = 'testing augmenting'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 3
    scn.super_main(a)


def test_no_augmenting():
    a = ProjectVariable()
    a.experiment_name = 'testing no augmenting'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 3
    scn.super_main(a)


def load_model_test():
    a = ProjectVariable()
    a.experiment_name = 'loading a trained model'
    a.load_model_name = 'priming_on_viper'
    a.dataset_test = 'viper'
    a.only_test = True
    a.ranking_number_test = 100
    a.iterations = 1
    scn.super_main(a)


def augmentation_1():
    a = ProjectVariable()
    a.experiment_name = 'augmenting positive data 1. on viper + concatenation'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 5
    a.neural_distance = 'concatenate'
    scn.super_main(a)


def augmentation_2():
    a = ProjectVariable()
    a.experiment_name = 'augmenting positive data 2. on grid + concatenation'
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.iterations = 5
    a.neural_distance = 'concatenate'
    scn.super_main(a)


def augmentation_3():
    a = ProjectVariable()
    a.experiment_name = 'augmenting positive data 3. on prid450 + concatenation'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 5
    a.neural_distance = 'concatenate'
    scn.super_main(a)

'/home/gabi/Documents/datasets/VIPeR/padded/000_a.bmp'


def load_and_retrain():
    a = ProjectVariable()
    a.experiment_name = 'Load weights and retrain + test on prid450'

    a.load_weights_name = 'viper_epoch_50'

    a.iterations = 10
    a.log_file = 'experiment_log.txt'

    a.dataset_test = 'prid450'
    a.ranking_number_test = 100

    scn.super_main(a)


def ex_10_2_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 10_2_0: video_head_type=cnn_lstm on ilids, 512'
    a.epochs = 50
    a.iterations = 1
    a.dataset_test = 'ilids-vid'
    a.ranking_number_test = 30
    a.sequence_length = 22
    a.video_head_type = 'cnn_lstm'
    a.dropout_rate = 0.05
    a.activation_function = 'selu'
    a.lstm_units = 512
    a.use_cyclical_learning_rate = False
    srcn.super_main(a)


def numfil_0():
    a = ProjectVariable()
    a.experiment_name = 'numfil 0: make numfil=2 + concatenation'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 5
    a.neural_distance = 'concatenate'
    a.numfil = 2
    scn.super_main(a)


def numfil_1():
    a = ProjectVariable()
    a.experiment_name = 'numfil 1: make numfil=2 + absolute'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 5
    a.neural_distance = 'absolute'
    a.numfil = 2
    scn.super_main(a)


def saving_models():
    a = ProjectVariable()
    a.experiment_name = 'saving model: market'
    a.dataset_test = 'market'
    a.ranking_number_test = 100

    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_of_saved_file = 'priming_on_%s' % a.dataset_test

    a.neural_distance = 'concatenate'
    a.numfil = 2
    scn.super_main(a)


def augment_2_0():
    a = ProjectVariable()
    a.experiment_name = 'aug 2_0: with positive data augmentation, train+test on viper, neural distance=add'
    a.use_gpu = '0'
    a.log_file = 'log_0.txt'
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


def augment_2_1():
    a = ProjectVariable()
    a.experiment_name = 'aug 2_1: with positive data augmentation, train+test on grid, neural distance=add'
    a.use_gpu = '0'
    a.log_file = 'log_0.txt'
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


def augment_2_2():
    a = ProjectVariable()
    a.experiment_name = 'aug 2_2: with positive data augmentation, train+test on prid450, neural distance=add'
    a.use_gpu = '0'
    a.log_file = 'log_0.txt'
    a.epochs = 100
    a.iterations = 20
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


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
    a.use_gpu = '0'
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
    a.use_gpu = '0'
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
    a.use_gpu = '0'
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
        a.use_gpu = '0'
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
        a.use_gpu = '0'
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
    a.use_gpu = '0'
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
        a.use_gpu = '0'
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
        a.use_gpu = '0'
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
    a.use_gpu = '0'
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
        a.use_gpu = '0'
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
        a.use_gpu = '0'
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
    a.use_gpu = '0'
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


# try to run experiment where we mix data for video
def ex_10_4():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.experiment_name = 'see if we can mix video data'
    a.epochs = 1
    a.iterations = 1
    a.neural_distance = 'concatenate'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]

    a.dataset_test = 'ilids-vid-20'
    a.ranking_number_test = 30
    a.sequence_length = 20

    a.datasets_train = ['prid2011']
    a.ranking_number_train = [5]

    a.mix = True
    a.mix_with_test = True

    srcn.super_main(a)


def whuwhu():
    a = ProjectVariable()
    a.experiment_name = 'no dropout'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 10
    a.neural_distance = 'concatenate'
    a.numfil = 1
    a.dropout_rate = 0
    scn.super_main(a)


# 2_0 decay=0.004 instead of 0.95. seeing if decay will change anything euclidean
def ex_2_0_0():
    a = ProjectVariable()
    a.experiment_name = '2_0_0: euclidean with nadam. no CLR. decay=0.004'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def ex_2_0_1():
    a = ProjectVariable()
    a.experiment_name = '2_0_1: euclidean with nadam. no CLR. decay=0.004'
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)
    

def ex_2_0_2():
    a = ProjectVariable()
    a.experiment_name = '2_0_2: euclidean with nadam. no CLR. decay=0.004'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


# 2_1 decay=0 instead of 0.95 + rmsprop instead of nadam euclidean
def ex_2_1_0():
    a = ProjectVariable()
    a.experiment_name = '2_1_0: euclidean with rms. no CLR. decay=0'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


def ex_2_1_1():
    a = ProjectVariable()
    a.experiment_name = '2_1_1: euclidean with rms. no CLR. decay=0'
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


def ex_2_1_2():
    a = ProjectVariable()
    a.experiment_name = '2_1_2: euclidean with rms. no CLR. decay=0'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


# 2_2 decay=0.004 instead of 0.95. seeing if decay will change anything cosine
def ex_2_2_0():
    a = ProjectVariable()
    a.experiment_name = '2_2_0: cosine with nadam. no CLR. decay=0.004'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def ex_2_2_1():
    a = ProjectVariable()
    a.experiment_name = '2_2_1: cosine with nadam. no CLR. decay=0.004'
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def ex_2_2_2():
    a = ProjectVariable()
    a.experiment_name = '2_2_2: cosine with nadam. no CLR. decay=0.004'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


# 2_3 decay=0 instead of 0.95 + rmsprop instead of nadam. cosine
def ex_2_3_0():
    a = ProjectVariable()
    a.experiment_name = '2_3_0: cosine with rms. no CLR. decay=0'
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


def ex_2_3_1():
    a = ProjectVariable()
    a.experiment_name = '2_3_1: cosine with rms. no CLR. decay=0'
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


def ex_2_3_2():
    a = ProjectVariable()
    a.experiment_name = '2_3_2: cosine with rms. no CLR. decay=0'
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.iterations = 20
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.optimizer = 'rms'
    a.decay = 0
    scn.super_main(a)


def test_prid2011_450():
    a = ProjectVariable()
    a.experiment_name = 'testing siamese video: 3D convolutions on prid2011, ranking number=308'
    a.epochs = 1
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.dataset_test = 'prid2011'
    a.video_head_type = '3d_convolution'
    a.sequence_length = 20
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number_test = '308'
    srcn.super_main(a)


# 22 priming with augmented data, ratio = 50%

def ex_22_0():
    all_confusion_base = []
    all_cmc_base = []
    total_time = 0
    all_confusion_primed_5_clr_same = []
    all_cmc_primed_5_clr_same = []

    all_confusion_primed_10_clr_same = []
    all_cmc_primed_10_clr_same = []

    all_confusion_primed_5_clr_diff = []
    all_cmc_primed_5_clr_diff = []

    all_confusion_primed_10_clr_diff = []
    all_cmc_primed_10_clr_diff = []

    all_confusion_primed_5_lr_00001 = []
    all_cmc_primed_5_lr_00001 = []

    all_confusion_primed_10_lr_00001 = []
    all_cmc_primed_10_lr_00001 = []

    all_confusion_primed_5_lr_000001 = []
    all_cmc_primed_5_lr_000001 = []

    all_confusion_primed_10_lr_000001 = []
    all_cmc_primed_10_lr_000001 = []
    name = 'viper'
    iterations = 20
    gpu = '0'

    for itera in range(iterations):
        # ----------------------------
        # train and save model+weigths
        # ----------------------------
        a = ProjectVariable()
        a.use_gpu = gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.dataset_test = name
        a.ranking_number_test = 100
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = name
        a.log_experiment = False
        rank, matrix, tim = scn.super_main(a, get_data=True)
        all_cmc_base.append(rank)
        all_confusion_base.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_same.append(rank)
        all_confusion_primed_5_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_same.append(rank)
        all_confusion_primed_10_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_diff.append(rank)
        all_confusion_primed_5_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_diff.append(rank)
        all_confusion_primed_10_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_00001.append(rank)
        all_confusion_primed_5_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_00001.append(rank)
        all_confusion_primed_10_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_000001.append(rank)
        all_confusion_primed_5_lr_000001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_000001.append(rank)
        all_confusion_primed_10_lr_000001.append(matrix)
        total_time += tim

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = gpu
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 22_0: priming on %s with augmented data. ratio 1:1' % name
    # get the means for base
    matrix_means_base = np.mean(all_confusion_base, axis=0)
    matrix_std_base = np.std(all_confusion_base, axis=0)
    ranking_means_base = np.mean(all_cmc_base, axis=0)
    ranking_std_base = np.std(all_cmc_base, axis=0)
    # get the means and std for primed 5 epochs CLR same
    matrix_means_primed_5_clr_same = np.mean(all_confusion_primed_5_clr_same, axis=0)
    matrix_std_primed_5_clr_same = np.std(all_confusion_primed_5_clr_same, axis=0)
    ranking_means_primed_5_clr_same = np.mean(all_cmc_primed_5_clr_same, axis=0)
    ranking_std_primed_5_clr_same = np.std(all_cmc_primed_5_clr_same, axis=0)
    # get the means and std for primed 10 epochs CLR same
    matrix_means_primed_10_clr_same = np.mean(all_confusion_primed_10_clr_same, axis=0)
    matrix_std_primed_10_clr_same = np.std(all_confusion_primed_10_clr_same, axis=0)
    ranking_means_primed_10_clr_same = np.mean(all_cmc_primed_10_clr_same, axis=0)
    ranking_std_primed_10_clr_same = np.std(all_cmc_primed_10_clr_same, axis=0)
    # get the means and std for primed 5 epochs CLR diff
    matrix_means_primed_5_clr_diff = np.mean(all_confusion_primed_5_clr_diff, axis=0)
    matrix_std_primed_5_clr_diff = np.std(all_confusion_primed_5_clr_diff, axis=0)
    ranking_means_primed_5_clr_diff = np.mean(all_cmc_primed_5_clr_diff, axis=0)
    ranking_std_primed_5_clr_diff = np.std(all_cmc_primed_5_clr_diff, axis=0)
    # get the means and std for primed 10 epochs CLR diff
    matrix_means_primed_10_clr_diff = np.mean(all_confusion_primed_10_clr_diff, axis=0)
    matrix_std_primed_10_clr_diff = np.std(all_confusion_primed_10_clr_diff, axis=0)
    ranking_means_primed_10_clr_diff = np.mean(all_cmc_primed_10_clr_diff, axis=0)
    ranking_std_primed_10_clr_diff = np.std(all_cmc_primed_10_clr_diff, axis=0)

    # get the means and std for primed 5 epochs Lr 0.00001
    matrix_means_primed_5_lr_00001 = np.mean(all_confusion_primed_5_lr_00001, axis=0)
    matrix_std_primed_5_lr_00001 = np.std(all_confusion_primed_5_lr_00001, axis=0)
    ranking_means_primed_5_lr_00001 = np.mean(all_cmc_primed_5_lr_00001, axis=0)
    ranking_std_primed_5_lr_00001 = np.std(all_cmc_primed_5_lr_00001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.00001
    matrix_means_primed_10_lr_00001 = np.mean(all_confusion_primed_10_lr_00001, axis=0)
    matrix_std_primed_10_lr_00001 = np.std(all_confusion_primed_10_lr_00001, axis=0)
    ranking_means_primed_10_lr_00001 = np.mean(all_cmc_primed_10_lr_00001, axis=0)
    ranking_std_primed_10_lr_00001 = np.std(all_cmc_primed_10_lr_00001, axis=0)
    # get the means and std for primed 5 epochs Lr 0.000001
    matrix_means_primed_5_lr_000001 = np.mean(all_confusion_primed_5_lr_000001, axis=0)
    matrix_std_primed_5_lr_000001 = np.std(all_confusion_primed_5_lr_000001, axis=0)
    ranking_means_primed_5_lr_000001 = np.mean(all_cmc_primed_5_lr_000001, axis=0)
    ranking_std_primed_5_lr_000001 = np.std(all_cmc_primed_5_lr_000001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.000001
    matrix_means_primed_10_lr_000001 = np.mean(all_confusion_primed_10_lr_000001, axis=0)
    matrix_std_primed_10_lr_000001 = np.std(all_confusion_primed_10_lr_000001, axis=0)
    ranking_means_primed_10_lr_000001 = np.mean(all_cmc_primed_10_lr_000001, axis=0)
    ranking_std_primed_10_lr_000001 = np.std(all_cmc_primed_10_lr_000001, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log_priming_augment(a, a.experiment_name, file_name, name,
                                        matrix_means_base, matrix_std_base, ranking_means_base, ranking_std_base,
                                        matrix_means_primed_5_clr_same, matrix_std_primed_5_clr_same,
                                        ranking_means_primed_5_clr_same, ranking_std_primed_5_clr_same,
                                        matrix_means_primed_10_clr_same, matrix_std_primed_10_clr_same,
                                        ranking_means_primed_10_clr_same, ranking_std_primed_10_clr_same,
                                        matrix_means_primed_5_clr_diff, matrix_std_primed_5_clr_diff,
                                        ranking_means_primed_5_clr_diff, ranking_std_primed_5_clr_diff,
                                        matrix_means_primed_10_clr_diff, matrix_std_primed_10_clr_diff,
                                        ranking_means_primed_10_clr_diff, ranking_std_primed_10_clr_diff,
                                        matrix_means_primed_5_lr_00001, matrix_std_primed_5_lr_00001,
                                        ranking_means_primed_5_lr_00001, ranking_std_primed_5_lr_00001,
                                        matrix_means_primed_10_lr_00001, matrix_std_primed_10_lr_00001,
                                        ranking_means_primed_10_lr_00001, ranking_std_primed_10_lr_00001,
                                        matrix_means_primed_5_lr_000001, matrix_std_primed_5_lr_000001,
                                        ranking_means_primed_5_lr_000001, ranking_std_primed_5_lr_000001,
                                        matrix_means_primed_10_lr_000001, matrix_std_primed_10_lr_000001,
                                        ranking_means_primed_10_lr_000001, ranking_std_primed_10_lr_000001,
                                        total_time)


def ex_22_1():
    all_confusion_base = []
    all_cmc_base = []
    total_time = 0
    all_confusion_primed_5_clr_same = []
    all_cmc_primed_5_clr_same = []

    all_confusion_primed_10_clr_same = []
    all_cmc_primed_10_clr_same = []

    all_confusion_primed_5_clr_diff = []
    all_cmc_primed_5_clr_diff = []

    all_confusion_primed_10_clr_diff = []
    all_cmc_primed_10_clr_diff = []

    all_confusion_primed_5_lr_00001 = []
    all_cmc_primed_5_lr_00001 = []

    all_confusion_primed_10_lr_00001 = []
    all_cmc_primed_10_lr_00001 = []

    all_confusion_primed_5_lr_000001 = []
    all_cmc_primed_5_lr_000001 = []

    all_confusion_primed_10_lr_000001 = []
    all_cmc_primed_10_lr_000001 = []
    name = 'grid'
    iterations = 20
    gpu = '0'

    for itera in range(iterations):
        # ----------------------------
        # train and save model+weigths
        # ----------------------------
        a = ProjectVariable()
        a.use_gpu = gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.dataset_test = name
        a.ranking_number_test = 100
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = name
        a.log_experiment = False
        rank, matrix, tim = scn.super_main(a, get_data=True)
        all_cmc_base.append(rank)
        all_confusion_base.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_same.append(rank)
        all_confusion_primed_5_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_same.append(rank)
        all_confusion_primed_10_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_diff.append(rank)
        all_confusion_primed_5_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_diff.append(rank)
        all_confusion_primed_10_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_00001.append(rank)
        all_confusion_primed_5_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_00001.append(rank)
        all_confusion_primed_10_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_000001.append(rank)
        all_confusion_primed_5_lr_000001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_000001.append(rank)
        all_confusion_primed_10_lr_000001.append(matrix)
        total_time += tim

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = gpu
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 22_1: priming on %s with augmented data. ratio 1:1' % name
    # get the means for base
    matrix_means_base = np.mean(all_confusion_base, axis=0)
    matrix_std_base = np.std(all_confusion_base, axis=0)
    ranking_means_base = np.mean(all_cmc_base, axis=0)
    ranking_std_base = np.std(all_cmc_base, axis=0)
    # get the means and std for primed 5 epochs CLR same
    matrix_means_primed_5_clr_same = np.mean(all_confusion_primed_5_clr_same, axis=0)
    matrix_std_primed_5_clr_same = np.std(all_confusion_primed_5_clr_same, axis=0)
    ranking_means_primed_5_clr_same = np.mean(all_cmc_primed_5_clr_same, axis=0)
    ranking_std_primed_5_clr_same = np.std(all_cmc_primed_5_clr_same, axis=0)
    # get the means and std for primed 10 epochs CLR same
    matrix_means_primed_10_clr_same = np.mean(all_confusion_primed_10_clr_same, axis=0)
    matrix_std_primed_10_clr_same = np.std(all_confusion_primed_10_clr_same, axis=0)
    ranking_means_primed_10_clr_same = np.mean(all_cmc_primed_10_clr_same, axis=0)
    ranking_std_primed_10_clr_same = np.std(all_cmc_primed_10_clr_same, axis=0)
    # get the means and std for primed 5 epochs CLR diff
    matrix_means_primed_5_clr_diff = np.mean(all_confusion_primed_5_clr_diff, axis=0)
    matrix_std_primed_5_clr_diff = np.std(all_confusion_primed_5_clr_diff, axis=0)
    ranking_means_primed_5_clr_diff = np.mean(all_cmc_primed_5_clr_diff, axis=0)
    ranking_std_primed_5_clr_diff = np.std(all_cmc_primed_5_clr_diff, axis=0)
    # get the means and std for primed 10 epochs CLR diff
    matrix_means_primed_10_clr_diff = np.mean(all_confusion_primed_10_clr_diff, axis=0)
    matrix_std_primed_10_clr_diff = np.std(all_confusion_primed_10_clr_diff, axis=0)
    ranking_means_primed_10_clr_diff = np.mean(all_cmc_primed_10_clr_diff, axis=0)
    ranking_std_primed_10_clr_diff = np.std(all_cmc_primed_10_clr_diff, axis=0)

    # get the means and std for primed 5 epochs Lr 0.00001
    matrix_means_primed_5_lr_00001 = np.mean(all_confusion_primed_5_lr_00001, axis=0)
    matrix_std_primed_5_lr_00001 = np.std(all_confusion_primed_5_lr_00001, axis=0)
    ranking_means_primed_5_lr_00001 = np.mean(all_cmc_primed_5_lr_00001, axis=0)
    ranking_std_primed_5_lr_00001 = np.std(all_cmc_primed_5_lr_00001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.00001
    matrix_means_primed_10_lr_00001 = np.mean(all_confusion_primed_10_lr_00001, axis=0)
    matrix_std_primed_10_lr_00001 = np.std(all_confusion_primed_10_lr_00001, axis=0)
    ranking_means_primed_10_lr_00001 = np.mean(all_cmc_primed_10_lr_00001, axis=0)
    ranking_std_primed_10_lr_00001 = np.std(all_cmc_primed_10_lr_00001, axis=0)
    # get the means and std for primed 5 epochs Lr 0.000001
    matrix_means_primed_5_lr_000001 = np.mean(all_confusion_primed_5_lr_000001, axis=0)
    matrix_std_primed_5_lr_000001 = np.std(all_confusion_primed_5_lr_000001, axis=0)
    ranking_means_primed_5_lr_000001 = np.mean(all_cmc_primed_5_lr_000001, axis=0)
    ranking_std_primed_5_lr_000001 = np.std(all_cmc_primed_5_lr_000001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.000001
    matrix_means_primed_10_lr_000001 = np.mean(all_confusion_primed_10_lr_000001, axis=0)
    matrix_std_primed_10_lr_000001 = np.std(all_confusion_primed_10_lr_000001, axis=0)
    ranking_means_primed_10_lr_000001 = np.mean(all_cmc_primed_10_lr_000001, axis=0)
    ranking_std_primed_10_lr_000001 = np.std(all_cmc_primed_10_lr_000001, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log_priming_augment(a, a.experiment_name, file_name, name,
                                        matrix_means_base, matrix_std_base, ranking_means_base, ranking_std_base,
                                        matrix_means_primed_5_clr_same, matrix_std_primed_5_clr_same,
                                        ranking_means_primed_5_clr_same, ranking_std_primed_5_clr_same,
                                        matrix_means_primed_10_clr_same, matrix_std_primed_10_clr_same,
                                        ranking_means_primed_10_clr_same, ranking_std_primed_10_clr_same,
                                        matrix_means_primed_5_clr_diff, matrix_std_primed_5_clr_diff,
                                        ranking_means_primed_5_clr_diff, ranking_std_primed_5_clr_diff,
                                        matrix_means_primed_10_clr_diff, matrix_std_primed_10_clr_diff,
                                        ranking_means_primed_10_clr_diff, ranking_std_primed_10_clr_diff,
                                        matrix_means_primed_5_lr_00001, matrix_std_primed_5_lr_00001,
                                        ranking_means_primed_5_lr_00001, ranking_std_primed_5_lr_00001,
                                        matrix_means_primed_10_lr_00001, matrix_std_primed_10_lr_00001,
                                        ranking_means_primed_10_lr_00001, ranking_std_primed_10_lr_00001,
                                        matrix_means_primed_5_lr_000001, matrix_std_primed_5_lr_000001,
                                        ranking_means_primed_5_lr_000001, ranking_std_primed_5_lr_000001,
                                        matrix_means_primed_10_lr_000001, matrix_std_primed_10_lr_000001,
                                        ranking_means_primed_10_lr_000001, ranking_std_primed_10_lr_000001,
                                        total_time)


def ex_22_2():
    all_confusion_base = []
    all_cmc_base = []
    total_time = 0
    all_confusion_primed_5_clr_same = []
    all_cmc_primed_5_clr_same = []

    all_confusion_primed_10_clr_same = []
    all_cmc_primed_10_clr_same = []

    all_confusion_primed_5_clr_diff = []
    all_cmc_primed_5_clr_diff = []

    all_confusion_primed_10_clr_diff = []
    all_cmc_primed_10_clr_diff = []

    all_confusion_primed_5_lr_00001 = []
    all_cmc_primed_5_lr_00001 = []

    all_confusion_primed_10_lr_00001 = []
    all_cmc_primed_10_lr_00001 = []

    all_confusion_primed_5_lr_000001 = []
    all_cmc_primed_5_lr_000001 = []

    all_confusion_primed_10_lr_000001 = []
    all_cmc_primed_10_lr_000001 = []
    name = 'prid450'
    iterations = 20
    gpu = '0'

    for itera in range(iterations):
        # ----------------------------
        # train and save model+weigths
        # ----------------------------
        a = ProjectVariable()
        a.use_gpu = gpu
        a.epochs = 100
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.dataset_test = name
        a.ranking_number_test = 100
        a.save_inbetween = True
        a.save_points = [100]
        a.name_of_saved_file = name
        a.log_experiment = False
        rank, matrix, tim = scn.super_main(a, get_data=True)
        all_cmc_base.append(rank)
        all_confusion_base.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_same.append(rank)
        all_confusion_primed_5_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR same
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_same.append(rank)
        all_confusion_primed_10_clr_same.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_clr_diff.append(rank)
        all_confusion_primed_5_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, CLR diff
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.cl_min = 0.000001
        a.cl_max = 0.0001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_clr_diff.append(rank)
        all_confusion_primed_10_clr_diff.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_00001.append(rank)
        all_confusion_primed_5_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.00001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_00001.append(rank)
        all_confusion_primed_10_lr_00001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 5 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 5
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_5_lr_000001.append(rank)
        all_confusion_primed_5_lr_000001.append(matrix)
        total_time += tim
        # -----------------------------
        # load weights, do priming, 10 epochs, LR=0.000001
        # -----------------------------
        a = ProjectVariable()
        a.negative_priming_ratio = 8
        a.prime_epochs = 10
        a.use_cyclical_learning_rate = False
        a.learning_rate = 0.000001
        a.use_gpu = gpu
        a.iterations = 1
        a.neural_distance = 'concatenate'
        a.priming = True
        a.dataset_test = name
        a.ranking_number_test = 100
        a.load_model_name = '%s_epoch_100' % name
        a.load_weights_name = '%s_epoch_100' % name
        a.log_experiment = False
        rank, matrix, tim = prime.super_main(a, get_data=True)
        all_cmc_primed_10_lr_000001.append(rank)
        all_confusion_primed_10_lr_000001.append(matrix)
        total_time += tim

    # calculate the mean information and the std
    # ---------------------------------------------------------------------------------------------------------------
    a = ProjectVariable()
    a.use_gpu = gpu
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 22_2: priming on %s with augmented data. ratio 1:1' % name
    # get the means for base
    matrix_means_base = np.mean(all_confusion_base, axis=0)
    matrix_std_base = np.std(all_confusion_base, axis=0)
    ranking_means_base = np.mean(all_cmc_base, axis=0)
    ranking_std_base = np.std(all_cmc_base, axis=0)
    # get the means and std for primed 5 epochs CLR same
    matrix_means_primed_5_clr_same = np.mean(all_confusion_primed_5_clr_same, axis=0)
    matrix_std_primed_5_clr_same = np.std(all_confusion_primed_5_clr_same, axis=0)
    ranking_means_primed_5_clr_same = np.mean(all_cmc_primed_5_clr_same, axis=0)
    ranking_std_primed_5_clr_same = np.std(all_cmc_primed_5_clr_same, axis=0)
    # get the means and std for primed 10 epochs CLR same
    matrix_means_primed_10_clr_same = np.mean(all_confusion_primed_10_clr_same, axis=0)
    matrix_std_primed_10_clr_same = np.std(all_confusion_primed_10_clr_same, axis=0)
    ranking_means_primed_10_clr_same = np.mean(all_cmc_primed_10_clr_same, axis=0)
    ranking_std_primed_10_clr_same = np.std(all_cmc_primed_10_clr_same, axis=0)
    # get the means and std for primed 5 epochs CLR diff
    matrix_means_primed_5_clr_diff = np.mean(all_confusion_primed_5_clr_diff, axis=0)
    matrix_std_primed_5_clr_diff = np.std(all_confusion_primed_5_clr_diff, axis=0)
    ranking_means_primed_5_clr_diff = np.mean(all_cmc_primed_5_clr_diff, axis=0)
    ranking_std_primed_5_clr_diff = np.std(all_cmc_primed_5_clr_diff, axis=0)
    # get the means and std for primed 10 epochs CLR diff
    matrix_means_primed_10_clr_diff = np.mean(all_confusion_primed_10_clr_diff, axis=0)
    matrix_std_primed_10_clr_diff = np.std(all_confusion_primed_10_clr_diff, axis=0)
    ranking_means_primed_10_clr_diff = np.mean(all_cmc_primed_10_clr_diff, axis=0)
    ranking_std_primed_10_clr_diff = np.std(all_cmc_primed_10_clr_diff, axis=0)

    # get the means and std for primed 5 epochs Lr 0.00001
    matrix_means_primed_5_lr_00001 = np.mean(all_confusion_primed_5_lr_00001, axis=0)
    matrix_std_primed_5_lr_00001 = np.std(all_confusion_primed_5_lr_00001, axis=0)
    ranking_means_primed_5_lr_00001 = np.mean(all_cmc_primed_5_lr_00001, axis=0)
    ranking_std_primed_5_lr_00001 = np.std(all_cmc_primed_5_lr_00001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.00001
    matrix_means_primed_10_lr_00001 = np.mean(all_confusion_primed_10_lr_00001, axis=0)
    matrix_std_primed_10_lr_00001 = np.std(all_confusion_primed_10_lr_00001, axis=0)
    ranking_means_primed_10_lr_00001 = np.mean(all_cmc_primed_10_lr_00001, axis=0)
    ranking_std_primed_10_lr_00001 = np.std(all_cmc_primed_10_lr_00001, axis=0)
    # get the means and std for primed 5 epochs Lr 0.000001
    matrix_means_primed_5_lr_000001 = np.mean(all_confusion_primed_5_lr_000001, axis=0)
    matrix_std_primed_5_lr_000001 = np.std(all_confusion_primed_5_lr_000001, axis=0)
    ranking_means_primed_5_lr_000001 = np.mean(all_cmc_primed_5_lr_000001, axis=0)
    ranking_std_primed_5_lr_000001 = np.std(all_cmc_primed_5_lr_000001, axis=0)
    # get the means and std for primed 10 epochs Lr 0.000001
    matrix_means_primed_10_lr_000001 = np.mean(all_confusion_primed_10_lr_000001, axis=0)
    matrix_std_primed_10_lr_000001 = np.std(all_confusion_primed_10_lr_000001, axis=0)
    ranking_means_primed_10_lr_000001 = np.mean(all_cmc_primed_10_lr_000001, axis=0)
    ranking_std_primed_10_lr_000001 = np.std(all_cmc_primed_10_lr_000001, axis=0)

    if a.log_experiment:
        file_name = os.path.basename(__file__)
        pu.enter_in_log_priming_augment(a, a.experiment_name, file_name, name,
                                        matrix_means_base, matrix_std_base, ranking_means_base, ranking_std_base,
                                        matrix_means_primed_5_clr_same, matrix_std_primed_5_clr_same,
                                        ranking_means_primed_5_clr_same, ranking_std_primed_5_clr_same,
                                        matrix_means_primed_10_clr_same, matrix_std_primed_10_clr_same,
                                        ranking_means_primed_10_clr_same, ranking_std_primed_10_clr_same,
                                        matrix_means_primed_5_clr_diff, matrix_std_primed_5_clr_diff,
                                        ranking_means_primed_5_clr_diff, ranking_std_primed_5_clr_diff,
                                        matrix_means_primed_10_clr_diff, matrix_std_primed_10_clr_diff,
                                        ranking_means_primed_10_clr_diff, ranking_std_primed_10_clr_diff,
                                        matrix_means_primed_5_lr_00001, matrix_std_primed_5_lr_00001,
                                        ranking_means_primed_5_lr_00001, ranking_std_primed_5_lr_00001,
                                        matrix_means_primed_10_lr_00001, matrix_std_primed_10_lr_00001,
                                        ranking_means_primed_10_lr_00001, ranking_std_primed_10_lr_00001,
                                        matrix_means_primed_5_lr_000001, matrix_std_primed_5_lr_000001,
                                        ranking_means_primed_5_lr_000001, ranking_std_primed_5_lr_000001,
                                        matrix_means_primed_10_lr_000001, matrix_std_primed_10_lr_000001,
                                        ranking_means_primed_10_lr_000001, ranking_std_primed_10_lr_000001,
                                        total_time)

def ex_20_0():
    a = ProjectVariable()
    a.use_gpu = '0'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.experiment_name = 'experiment 20_0: video_head_type=3d_convolution on prid2011_450, batchnorm, with concatenation, dataset has 700 training instances'
    a.neural_distance = 'concatenate'
    a.epochs = 100
    a.iterations = 5
    a.dataset_test = 'prid2011_450'
    a.ranking_number_test = 100
    a.sequence_length = 20
    a.video_head_type = '3d_convolution'
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)

# euclidean
def ex_23_0_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_0_0: market, cost_module_type=euclidean, lr=0.00001'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def ex_23_0_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_0_1: cuhk02, cost_module_type=euclidean, lr=0.00001'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)
    
# cosine
def ex_23_1_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_1_0: market, cost_module_type=cosine, lr=0.00001'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)


def ex_23_1_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_1_1: cuhk02, cost_module_type=cosine, lr=0.00001'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)

# concatenate
def ex_23_2_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_0: market, neural_distance=concatenate'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'concatenate'
    scn.super_main(a)


def ex_23_2_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_1: cuhk02, neural_distance=concatenate'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'concatenate'
    scn.super_main(a)


# absolute
def ex_23_2_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_2: market, neural_distance=absolute'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'absolute'
    scn.super_main(a)


def ex_23_2_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_3: cuhk02, neural_distance=absolute'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'absolute'
    scn.super_main(a)


# add
def ex_23_2_4():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_4: market, neural_distance=add'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


def ex_23_2_5():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_5: cuhk02, neural_distance=add'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'add'
    scn.super_main(a)


# subtract
def ex_23_2_6():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_6: market, neural_distance=subtract'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'subtract'
    scn.super_main(a)


def ex_23_2_7():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_7: cuhk02, neural_distance=subtract'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'subtract'
    scn.super_main(a)


# multiply
def ex_23_2_8():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_8: market, neural_distance=multiply'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'multiply'
    scn.super_main(a)


def ex_23_2_9():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_9: cuhk02, neural_distance=multiply'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'multiply'
    scn.super_main(a)


# divide
def ex_23_2_10():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_10: market, neural_distance=divide'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.neural_distance = 'divide'
    scn.super_main(a)


def ex_23_2_11():
    a = ProjectVariable()
    a.experiment_name = 'experiment 23_2_11: cuhk02, neural_distance=divide'
    a.use_gpu = '0'
    a.log_file = 'thesis_results_%s.txt' % a.use_gpu
    a.epochs = 100
    a.iterations = 10
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.neural_distance = 'divide'
    scn.super_main(a)



def gab_ex_0():
    a = ProjectVariable()
    a.experiment_name = 'train + test, on 1 dataset, then save model and weights'
    a.iterations = 1
    a.dataset_test = 'viper'
    a.ranking_number_test = 50
    a.save_inbetween = [100]
    a.name_of_saved_file = 'viper'
    # files saved:
    # viper_epoch_100_model.h5
    # viper_epoch_100_weights.h5
    scn.super_main(a)


def gab_ex_1():
    a = ProjectVariable()
    a.experiment_name = 'train + test, on 3 dataset, then save model and weights'
    a.iterations = 1
    a.dataset_test = 'prid450'
    a.ranking_number_test = 50

    a.datasets_train = ['grid', 'viper']
    a.ranking_number_train = [5, 5]

    # save model + weights
    a.save_inbetween = [100]
    a.name_of_saved_file = 'viper'
    scn.super_main(a)


def gab_ex_3():
    a = ProjectVariable()
    a.experiment_name = 'load model + test on some datase'
    a.iterations = 1
    a.dataset_test = 'viper'
    a.ranking_number_test = 50

    a.only_test = True
    # assuming the naming convention
    a.load_model_name = 'viper_epoch_100'
    scn.super_main(a)


def gab_ex_4():
    a = ProjectVariable()
    a.experiment_name = 'only train + save model and weights'
    a.iterations = 1

    a.datasets_train = ['viper']
    a.save_inbetween = [100]
    a.name_of_saved_file = 'viper'

    scn.super_main(a)


def gab_ex_5():
    a = ProjectVariable()
    a.experiment_name = 'load weights + test on some datase'
    a.iterations = 1
    a.dataset_test = 'viper'
    a.ranking_number_test = 50

    a.only_test = True
    # assuming the naming convention
    a.load_weights_name = 'viper_epoch_100'
    scn.super_main(a)


def gab_ex_6():
    a = ProjectVariable()
    a.experiment_name = 'train model with video data'
    a.iterations = 1
    a.datasets_test = 'prid2011'
    a.ranking_number_test = 100
    a.sequence_length = 20
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    srcn.super_main(a)


def debug():
    a = ProjectVariable()
    a.experiment_name = 'debug'
    a.iterations = 1
    a.epochs = 1
    a.dataset_test = 'viper_augmented'
    a.ranking_number_test = 10
    a.upper_bound_pos_pairs_per_id = 6
    scn.super_main(a)


debug()


def main():
    num = sys.argv[1]
    print(sys.argv)

    # if num == '23_0_0': ex_23_0_0() # rerun
    # if num == '23_0_1': ex_23_0_1() # rerun
    # if num == '23_1_0': ex_23_1_0() # rerun
    # if num == '23_1_1': ex_23_1_1() # rerun
    # if num == '23_2_0': ex_23_2_0() # done
    # if num == '23_2_1': ex_23_2_1() # done
    # if num == '23_2_2': ex_23_2_2() # done
    # if num == '23_2_3': ex_23_2_3() # run
    # if num == '23_2_4': ex_23_2_4() # run
    # if num == '23_2_5': ex_23_2_5() # run
    # if num == '23_2_6': ex_23_2_6() # run
    # if num == '23_2_7': ex_23_2_7() # run
    # if num == '23_2_8': ex_23_2_8() # run
    # if num == '23_2_9': ex_23_2_9() # run
    # if num == '23_2_10': ex_23_2_10() # run
    # if num == '23_2_11': ex_23_2_11() # run

# main()
