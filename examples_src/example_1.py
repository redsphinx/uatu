import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn
import numpy as np
import project_utils as pu


def my_experiment():
    a = ProjectVariable()
    a.experiment_name = 'Experiment to test something'
    a.dataset_test = 'some_dataset'
    scn.super_main(a)


def main():
    my_experiment()


main()