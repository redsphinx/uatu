import siamese_cnn_clean as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_rcnn as srcn


'''
Experiments with Euclidean distance
'''
def e_001():
    a = ProjectVariable()
    a.experiment_name = '001. viper, euclidean, no CLR, rank=316'
    a.ranking_number = 316
    a.iterations = 10
    a.datasets = ['viper']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_002():
    a = ProjectVariable()
    a.experiment_name = '002. grid, euclidean, no CLR, rank=125'
    a.ranking_number = 125
    a.iterations = 10
    a.datasets = ['grid']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)
    

def e_003():
    a = ProjectVariable()
    a.experiment_name = '003. prid450, euclidean, no CLR, rank=225'
    a.ranking_number = 225
    a.iterations = 10
    a.datasets = ['prid450']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)
    

def e_004():
    a = ProjectVariable()
    a.experiment_name = '004. caviar, euclidean, no CLR, rank=36'
    a.ranking_number = 36
    a.iterations = 10
    a.datasets = ['caviar']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


'''
Experiments with Cosine distance
'''


def e_005():
    a = ProjectVariable()
    a.experiment_name = '005. viper, cosine, no CLR, rank=316'
    a.ranking_number = 316
    a.iterations = 10
    a.datasets = ['viper']
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_006():
    a = ProjectVariable()
    a.experiment_name = '006. grid, cosine, no CLR, rank=125'
    a.ranking_number = 125
    a.iterations = 10
    a.datasets = ['grid']
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_007():
    a = ProjectVariable()
    a.experiment_name = '007. prid450, cosine, no CLR, rank=225'
    a.ranking_number = 225
    a.iterations = 10
    a.datasets = ['prid450']
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_008():
    a = ProjectVariable()
    a.experiment_name = '008. caviar, cosine, no CLR, rank=36'
    a.ranking_number = 36
    a.iterations = 10
    a.datasets = ['caviar']
    a.cost_module_type = 'cosine'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


'''
Experiments with FC Layers
'''


def e_009():
    a = ProjectVariable()
    a.experiment_name = '009. viper, normal settings, rank=316'
    a.ranking_number = 316
    a.iterations = 10
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_010():
    a = ProjectVariable()
    a.experiment_name = '010. grid, normal settings, rank=125'
    a.ranking_number = 125
    a.iterations = 10
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_011():
    a = ProjectVariable()
    a.experiment_name = '011. prid450, normal settings, rank=225'
    a.ranking_number = 225
    a.iterations = 10
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_012():
    a = ProjectVariable()
    a.experiment_name = '012. caviar, normal settings, rank=36'
    a.ranking_number = 36
    a.iterations = 10
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)

'''
Experiments with selu, alpha dropout and removing batch norm
'''


def e_013():
    a = ProjectVariable()
    a.experiment_name = '013. viper: selu + alphadropout'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_014():
    a = ProjectVariable()
    a.experiment_name = '014. grid: selu + alphadropout'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_015():
    a = ProjectVariable()
    a.experiment_name = '015. prid450: selu + alphadropout'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_016():
    a = ProjectVariable()
    a.experiment_name = '016. caviar: selu + alphadropout'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)

'''
removing batchnorm
'''

def e_017():
    a = ProjectVariable()
    a.experiment_name = '017. viper: selu + alphadropout + no batchnorm'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_018():
    a = ProjectVariable()
    a.experiment_name = '018. grid: selu + alphadropout + no batchnorm'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_019():
    a = ProjectVariable()
    a.experiment_name = '019. prid450: selu + alphadropout + no batchnorm'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_020():
    a = ProjectVariable()
    a.experiment_name = '020. caviar: selu + alphadropout + no batchnorm'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)

'''
alphadropout=0.05
'''

def e_021():
    a = ProjectVariable()
    a.experiment_name = '021. viper: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_022():
    a = ProjectVariable()
    a.experiment_name = '022. grid: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_023():
    a = ProjectVariable()
    a.experiment_name = '023. prid450: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def e_024():
    a = ProjectVariable()
    a.experiment_name = '024. caviar: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    scn.super_main(a)


def main():
    num = sys.argv[1]
    print(sys.argv)

    if num == '21':
        e_021()
    if num == '22':
        e_022()
    if num == '23':
        e_023()
    if num == '24':
        e_024()

main()
