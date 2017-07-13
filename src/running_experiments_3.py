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


def main():
    num = sys.argv[1]
    print(sys.argv)

    if num == '1':
        e_001()
    if num == '2':
        e_002()
    if num == '3':
        e_003()

    if num == '5':
        e_005()
    if num == '6':
        e_006()
    if num == '7':
        e_007()

    if num == '9':
        e_009()
    if num == '10':
        e_010()
    if num == '11':
        e_011()

    if num == '13':
        e_013()
    if num == '14':
        e_014()
    if num == '15':
        e_015()

    if num == '17':
        e_017()
    if num == '18':
        e_018()
    if num == '19':
        e_019()

    if num == '21':
        e_021()
    if num == '22':
        e_022()
    if num == '23':
        e_023()

    if num == '25':
        e_025()
    if num == '26':
        e_026()
    if num == '27':
        e_027()

    if num == '35':
        e_035()
    if num == '36':
        e_036()
    if num == '37':
        e_037()
    if num == '38':
        e_038()
    if num == '39':
        e_039()


main()
