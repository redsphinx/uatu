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






def main():
    num = sys.argv[1]
    print(sys.argv)

    if num == '1':
        e_001()
    if num == '2':
        e_002()
    if num == '3':
        e_003()
    if num == '4':
        e_004()
    if num == '5':
        e_005()
    if num == '6':
        e_006()
    if num == '7':
        e_007()
    if num == '8':
        e_008()
    if num == '9':
        e_009()
    if num == '10':
        e_010()
    if num == '11':
        e_011()
    if num == '12':
        e_012()


main()