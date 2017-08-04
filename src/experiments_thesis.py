import siamese_cnn_image as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os
import siamese_cnn_video as srcn
import cnn_human_detection as cnn
import numpy as np
import project_utils as pu


def ex_0_7_0():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_0: viper, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'viper'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)

def ex_0_7_1():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_1: grid, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'grid'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)

def ex_0_7_2():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_2: prid450, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'prid450'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)

def ex_0_7_3():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_3: cuhk01, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk01'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)

def ex_0_7_4():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: cuhk02, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'cuhk02'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)

def ex_0_7_5():
    a = ProjectVariable()
    a.experiment_name = 'experiment 0_7_4: market, BASELINE'
    a.epochs = 100
    a.iterations = 30
    a.dataset_test = 'market'
    a.ranking_number_test = 100
    a.log_file = 'experiment_results_for_thesis.txt'
    scn.super_main(a)


def main():
    num = sys.argv[1]
    print(sys.argv)

    if num == '0_7_0':
        ex_0_7_0()
    if num == '0_7_1':
        ex_0_7_1()
    if num == '0_7_2':
        ex_0_7_2()
    if num == '0_7_3':
        ex_0_7_3()
    if num == '0_7_4':
        ex_0_7_4()
    if num == '0_7_5':
        ex_0_7_5()


main()

