import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn


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


def main():
    # num = sys.argv[1]
    # print(sys.argv)
    #
    # if num == '1':
    test_pipeline_9()


main()

'''
From scratch:
--WORKING-- a.experiment_name = 'test mixing: train + test multiple datasets + mix==True + mix_with_test==True' --WORKING--
--WORKING-- a.experiment_name = 'test mixing: train + test multiple datasets + mix==True + mix_with_test==False' --WORKING--
--WORKING-- a.experiment_name = 'test mixing: train + test multiple datasets + mix==False (+ mix_with_test==False)' --WORKING--

--WORKING-- a.experiment_name = 'test mixing: train + test single dataset' --WORKING--

--WORKING-- a.experiment_name = 'test mixing: only test' --WORKING--
--WORKING-- a.experiment_name = 'test mixing: only train on multiple dataset, no mixing' --WORKING--
--WORKING-- a.experiment_name = 'test mixing: only train on multiple dataset, with mixing' --WORKING--

Load data:
a.experiment_name = 'test mixing: train + test multiple datasets + mix==True + mix_with_test==True'
a.experiment_name = 'test mixing: train + test multiple datasets + mix==True + mix_with_test==False'
a.experiment_name = 'test mixing: train + test multiple datasets + mix==False (+ mix_with_test==True)'

a.experiment_name = 'test mixing: train + test single datasets'

a.experiment_name = 'test mixing: only test'
a.experiment_name = 'test mixing: only train on multiple dataset'
a.experiment_name = 'test mixing: only train on single dataset'
'''