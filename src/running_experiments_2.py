import siamese_cnn_clean as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os


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
    a.epochs = 100 # rem
    a.iterations = 10 # rem
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
    a.epochs = 100 # rem
    a.iterations = 10 # rem
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
    a.epochs = 100 # rem
    a.iterations = 10 # rem
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
    a.use_gpu = '1'
    a.log_file = 'log_%s.txt' % a.use_gpu
    a.datasets = ['market']
    scn.super_main(a)
    

def e_003():
    a = ProjectVariable()
    a.experiment_name = '003. baseline cuhk02'
    a.use_gpu = '1'
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
    a.use_gpu = '2'
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
    a.use_gpu = '2'
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
    a.use_gpu = '3'
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
    a.use_gpu = '3'
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
    a.use_gpu = '3'
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
    a.use_gpu = '2'
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
    a.use_gpu = '2'
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
    a.use_gpu = '3'
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
    a.use_gpu = '3'
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
    a.use_gpu = '3'
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
    a.use_gpu = '1'
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
    scn.super_main(a)


def e_019():
    a = ProjectVariable()
    a.experiment_name = '019. train on cuhk02 -> grid (full network)'
    a.use_gpu = '1'
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
    scn.super_main(a)


def e_020():
    a = ProjectVariable()
    a.experiment_name = '020. train on cuhk02 -> prid450 (full network)'
    a.use_gpu = '1'
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
    scn.super_main(a)


def e_021():
    a = ProjectVariable()
    a.experiment_name = '021. train on cuhk02 -> caviar (full network)'
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
    scn.super_main(a)

#################################################################################################
#    TRAIN ON CUHK02, ONLY CLASSIFIER
#################################################################################################

def e_022():
    a = ProjectVariable()
    a.experiment_name = '022. train on cuhk02 -> market (only classifier)'
    a.use_gpu = '1'
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
    a.use_gpu = '1'
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
    a.use_gpu = '1'
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
    a.experiment_name = '021. train on cuhk02 -> caviar (only classifier)'
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
    a.use_gpu = '2'
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
    a.use_gpu = '2'
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
    a.datasets = ['grid']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_029():
    a = ProjectVariable()
    a.experiment_name = '029. train on cuhk02, market -> prid450 (full network)'
    a.use_gpu = '3'
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
    a.datasets = ['prid450']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    scn.super_main(a)


def e_030():
    a = ProjectVariable()
    a.experiment_name = '030. train on cuhk02, market -> caviar (full network)'
    a.use_gpu = '3'
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
    a.use_gpu = '2'
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
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_032():
    a = ProjectVariable()
    a.experiment_name = '032. train on cuhk02, market -> grid (only classifier)'
    a.use_gpu = '2'
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
    a.datasets = ['grid']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_033():
    a = ProjectVariable()
    a.experiment_name = '033. train on cuhk02, market -> prid450 (only classifier)'
    a.use_gpu = '3'
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
    a.datasets = ['prid450']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)


def e_034():
    a = ProjectVariable()
    a.experiment_name = '034. train on cuhk02, market -> caviar (only classifier)'
    a.use_gpu = '3'
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

    a.ranking_number = 36
    a.save_inbetween = False
    a.log_experiment = True
    a.datasets = ['caviar']
    a.load_weights_name = 'market_weigths_%s.h5' % a.use_gpu
    a.trainable_12 = False
    a.trainable_34 = False
    a.trainable_56 = False
    scn.super_main(a)




# TODO: adapt the learning rate. set it lower

def main():
    # num = sys.argv[1]
    # print(sys.argv)
    #
    # if num == '012':
    #     experiment_012()
    # if num == '013':
    #     experiment_013()
    # if num == '014':
    #     experiment_014()
    experiment_test()

main()
