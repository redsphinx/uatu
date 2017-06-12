import siamese_cnn_clean as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os

def experiment_000():
    a = ProjectVariable()
    a.experiment_name = 'improved baseline SCNN setup 105: rank=100'
    a.iterations = 10
    a.epochs = 100
    a.batch_size = 32
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_001():
    a = ProjectVariable()
    a.experiment_name = 'train on viper only'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1214_market_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['caviar']
    a.name_indication = 'dataset_name'
    scn.super_main(a)