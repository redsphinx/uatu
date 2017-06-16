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
    a.epochs = 1 # rem
    a.iterations = 1 # rem
    a.ranking_number = 'half'
    # a.save_inbetween = True
    # a.save_points = [1]
    # a.name_indication = 'dataset_name'
    scn.super_main(a)



def experiment_009():
    a = ProjectVariable()
    a.experiment_name = '009. train on viper, then cuhk02'
    a.ranking_number = 2
    # a.epochs = 100
    a.epochs = 1
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    scn.super_main(a)

    a.ranking_number = 'half'
    a.datasets = ['cuhk02']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def experiment_010():
    a = ProjectVariable()
    a.experiment_name = '010. train on market only baseline'
    a.datasets = ['market']
    a.epochs = 1 # rem
    a.iterations = 1 # rem
    scn.super_main(a)


def experiment_011():
    a = ProjectVariable()
    a.experiment_name = '011. train on viper then market'
    a.ranking_number = 2
    # a.epochs = 100
    a.epochs = 1
    a.iterations = 1
    a.save_inbetween = True
    a.save_points = [100]
    a.name_indication = 'dataset_name'
    a.datasets = ['viper']
    scn.super_main(a)

    a.ranking_number = 'half'
    a.datasets = ['market']
    a.load_weights_name = 'viper_weigths.h5'
    scn.super_main(a)


def main():
    # num = sys.argv[1]
    # print(sys.argv)

    # if num == '008':
    #     experiment_008()
    # if num == '009':
    #     experiment_009()
    # if num == '010':
    #     experiment_010()
    # if num == '011':
    #     experiment_011()
    experiment_008()

main()
