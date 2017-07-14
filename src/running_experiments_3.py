import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn


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
Experiments with selu, alpha dropout 
'''


def e_013():
    a = ProjectVariable()
    a.experiment_name = '013. viper: selu + alphadropout=0.1'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_014():
    a = ProjectVariable()
    a.experiment_name = '014. grid: selu + alphadropout=0.1'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_015():
    a = ProjectVariable()
    a.experiment_name = '015. prid450: selu + alphadropout=0.1'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_016():
    a = ProjectVariable()
    a.experiment_name = '016. caviar: selu + alphadropout=0.1'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)

'''
removing batchnorm
'''

def e_017():
    a = ProjectVariable()
    a.experiment_name = '017. viper: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_018():
    a = ProjectVariable()
    a.experiment_name = '018. grid: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_019():
    a = ProjectVariable()
    a.experiment_name = '019. prid450: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_020():
    a = ProjectVariable()
    a.experiment_name = '020. caviar: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
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
    a.dropout_rate = 0.05
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
    a.dropout_rate = 0.05
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
    a.dropout_rate = 0.05
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
    a.dropout_rate = 0.05
    scn.super_main(a)


'''
Experiments with selu, alpha dropout=0.05 + batchnorm
'''


def e_025():
    a = ProjectVariable()
    a.experiment_name = '025. viper: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 316
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_026():
    a = ProjectVariable()
    a.experiment_name = '026. grid: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 125
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_027():
    a = ProjectVariable()
    a.experiment_name = '027. prid450: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 225
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_028():
    a = ProjectVariable()
    a.experiment_name = '028. caviar: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 36
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['caviar']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)

'''
video data experiments
'''

def e_029():
    a = ProjectVariable()
    a.experiment_name = '029. 3D convolutions on ilids-vid'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.datasets = ['ilids-vid']
    a.video_head_type = '3d_convolution'
    a.sequence_length = 22
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    srcn.super_main(a)


def e_030():
    a = ProjectVariable()
    a.experiment_name = '030. cnn_lstm on ilids-vid, AD=0.05'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.lstm_units = 64
    srcn.super_main(a)


def e_031():
    a = ProjectVariable()
    a.experiment_name = '031. cnn_lstm on ilids-vid, AD=0.1'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['ilids-vid']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 22
    a.ranking_number = 30
    a.dropout_rate = 0.1
    a.lstm_units = 64
    srcn.super_main(a)


def e_032():
    a = ProjectVariable()
    a.experiment_name = '032. 3D convolutions on prid2011'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.datasets = ['prid2011']
    a.video_head_type = '3d_convolution'
    a.sequence_length = 20
    a.kernel = (3, 3, 3)
    a.pooling_size = [[1, 4, 2], [1, 2, 2]]
    a.ranking_number = 30
    srcn.super_main(a)


def e_033():
    a = ProjectVariable()
    a.experiment_name = '033. cnn_lstm on prid2011, AD=0.05'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['prid2011']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 20
    a.ranking_number = 30
    a.dropout_rate = 0.05
    a.lstm_units = 64
    srcn.super_main(a)


def e_034():
    a = ProjectVariable()
    a.experiment_name = '034. cnn_lstm on prid2011, AD=0.1'
    a.epochs = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'selu'
    a.datasets = ['prid2011']
    a.video_head_type = 'cnn_lstm'
    a.sequence_length = 20
    a.ranking_number = 30
    a.dropout_rate = 0.1
    a.lstm_units = 64
    srcn.super_main(a)

'''
saving trained models for priming later on
'''


def e_035():
    a = ProjectVariable()
    a.experiment_name = '035. save viper for priming'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_036():
    a = ProjectVariable()
    a.experiment_name = '036. save grid for priming'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_037():
    a = ProjectVariable()
    a.experiment_name = '037. save prid450 for priming'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_038():
    a = ProjectVariable()
    a.experiment_name = '038. save market for priming'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['market']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_039():
    a = ProjectVariable()
    a.experiment_name = '039. save cuhk02 for priming'
    a.epochs = 100
    a.save_inbetween = True
    a.name_indication = 'dataset_name'
    a.save_points = [100]
    a.ranking_number = 100
    a.iterations = 1
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    a.datasets = ['cuhk02']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


# running again with rank 100 to see how rank affects the results


def e_040():
    a = ProjectVariable()
    a.experiment_name = '040. viper, euclidean, no CLR, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['viper']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_041():
    a = ProjectVariable()
    a.experiment_name = '041. grid, euclidean, no CLR, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['grid']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_042():
    a = ProjectVariable()
    a.experiment_name = '042. prid450, euclidean, no CLR, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['prid450']
    a.cost_module_type = 'euclidean'
    a.use_cyclical_learning_rate = False
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


'''
Experiments with FC Layers
'''


def e_049():
    a = ProjectVariable()
    a.experiment_name = '049. viper, normal settings, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_050():
    a = ProjectVariable()
    a.experiment_name = '050. grid, normal settings, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


def e_051():
    a = ProjectVariable()
    a.experiment_name = '051. prid450, normal settings, rank=100'
    a.ranking_number = 100
    a.iterations = 10
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    scn.super_main(a)


'''
Experiments with selu, alpha dropout 
'''


def e_053():
    a = ProjectVariable()
    a.experiment_name = '053. viper: selu + alphadropout=0.1'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_054():
    a = ProjectVariable()
    a.experiment_name = '054. grid: selu + alphadropout=0.1'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_055():
    a = ProjectVariable()
    a.experiment_name = '055. prid450: selu + alphadropout=0.1'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.1
    scn.super_main(a)


'''
removing batchnorm
'''

def e_057():
    a = ProjectVariable()
    a.experiment_name = '057. viper: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_058():
    a = ProjectVariable()
    a.experiment_name = '058. grid: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


def e_059():
    a = ProjectVariable()
    a.experiment_name = '059. prid450: selu + alphadropout=0.1 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.1
    scn.super_main(a)


'''
alphadropout=0.05
'''

def e_061():
    a = ProjectVariable()
    a.experiment_name = '061. viper: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_062():
    a = ProjectVariable()
    a.experiment_name = '062. grid: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_063():
    a = ProjectVariable()
    a.experiment_name = '063. prid450: selu + alphadropout=0.05 + no batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.head_type = 'simple'
    a.dropout_rate = 0.05
    scn.super_main(a)


'''
Experiments with selu, alpha dropout=0.05 + batchnorm
'''


def e_065():
    a = ProjectVariable()
    a.experiment_name = '065. viper: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['viper']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_066():
    a = ProjectVariable()
    a.experiment_name = '066. grid: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['grid']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def e_067():
    a = ProjectVariable()
    a.experiment_name = '067. prid450: selu + alphadropout=0.05 + batchnorm'
    a.ranking_number = 100
    a.iterations = 10
    a.activation_function = 'selu'
    a.datasets = ['prid450']
    a.log_file = 'thesis_experiment_log.txt'
    a.dropout_rate = 0.05
    scn.super_main(a)


def main():
    num = sys.argv[1]
    print(sys.argv)

    if num == '40':
        e_040()
    if num == '41':
        e_041()
    if num == '42':
        e_042()

    if num == '49':
        e_049()
    if num == '50':
        e_050()
    if num == '51':
        e_051()

    if num == '53':
        e_053()
    if num == '54':
        e_054()
    if num == '55':
        e_055()

    if num == '57':
        e_057()
    if num == '58':
        e_058()
    if num == '59':
        e_059()

    if num == '61':
        e_061()
    if num == '62':
        e_062()
    if num == '63':
        e_063()

    if num == '65':
        e_065()
    if num == '66':
        e_066()
    if num == '67':
        e_067()


main()
