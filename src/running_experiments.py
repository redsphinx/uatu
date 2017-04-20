import tensorflow as tf
import keras
from keras import backend as K

import project_constants as pc
import project_utils as pu
import cnn_clean as cnn
import siamese_cnn_clean as scn
import preprocessing as pre

import os
import numpy as np
import time


def experiment_0(data):
    # testing stuff
    experiment_name = 'running the CNN on raw images'
    print('experiment: %s' %experiment_name)
    cnn.main(experiment_name, data)


def experiment_1(data):
    experiment_name = 'subtracting the mean image'
    print('experiment: %s' %experiment_name)
    cnn.main(experiment_name, data)


def experiment_2(data):
    experiment_name = 'subtracting mean image + normalizing'
    print('experiment: %s' %experiment_name)
    cnn.main(experiment_name, data)
    

def experiment_3(data):
    experiment_name = 'subtracting mean image + PCA whitening'
    print('experiment: %s' %experiment_name)
    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.initialize_cnn_data()
    # train_data = pre.PCA_whiten(pre.center(train_data))
    # test_data = pre.PCA_whiten(pre.center(test_data))
    # validation_data = pre.PCA_whiten(pre.center(validation_data))
    # data = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
    cnn.main(experiment_name, data)


def experiment_4(data):
    experiment_name = 'normalizing the image'
    print('experiment: %s' %experiment_name)
    cnn.main(experiment_name, data)


def experiment_5(data):
    experiment_name = 'batch normalization '
    print('experiment: %s' % experiment_name)
    iterations = 3
    cnn.super_main(experiment_name, data, iterations, do_dropout=False)


def experiment_6(data):
    experiment_name = 'saving model: batch normalization after relu with bias, cyclical learning rate, mode=exp_range'
    print('experiment: %s' % experiment_name)
    iterations = 1
    cnn.super_main(experiment_name, data, iterations, do_dropout=False)

#triangular, triangular2, exp_range

def experiment_7(data):
    experiment_name = 'training SCN network on the new CNN weights, clr mode=triangular'
    print('experiment: %s' % experiment_name)
    clr_mode = 'triangular'
    iterations = 10
    scn.super_main(experiment_name, data, iterations, clr_mode)


def experiment_8(data):
    experiment_name = 'training SCN network on the new CNN weights, clr mode=triangular2'
    print('experiment: %s' % experiment_name)
    clr_mode = 'triangular2'
    iterations = 10
    scn.super_main(experiment_name, data, iterations, clr_mode)


def experiment_9(data):
    experiment_name = 'training SCN network on the new CNN weights, clr mode=exp_range'
    print('experiment: %s' % experiment_name)
    clr_mode = 'exp_range'
    iterations = 10
    scn.super_main(experiment_name, data, iterations, clr_mode)


def experiment_10(data):
    experiment_name = 'training SCN, no clr, batchnorm on but not trainable'
    print('experiment: %s' % experiment_name)
    iterations = 2
    scn.super_main(experiment_name, data, iterations)


def experiment_11(data):
    experiment_name = 'save simple CNN with 1D filters, no BN, no CLR. start with 16 filters.'
    print('experiment: %s' % experiment_name)
    iterations = 1
    numfil = 1
    weights_name = 'cnn_1D_filters_16.h5'
    cnn.super_main(experiment_name, data, iterations, numfil, weights_name)


def experiment_12(data):
    experiment_name = 'save simple CNN with 1D filters, no BN, no CLR. start with 32 filters.'
    print('experiment: %s' % experiment_name)
    iterations = 1
    numfil = 2
    weights_name = 'cnn_1D_filters_32.h5'
    cnn.super_main(experiment_name, data, iterations, numfil, weights_name)


def experiment_13(data):
    experiment_name = 'training SCN with 1D filters 16'
    print('experiment: %s' % experiment_name)
    iterations = 10
    numfil = 1
    weights_name = 'cnn_1D_filters_16.h5'
    scn.super_main(experiment_name, data, iterations, numfil, weights_name)


def experiment_14(data):
    experiment_name = 'training SCN with 1D filters 32'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    weights_name = 'cnn_1D_filters_32.h5'
    scn.super_main(experiment_name, data, iterations, numfil, weights_name)


def experiment_15():
    experiment_name = 'simple CNN with 1D filters, start 16 filters, DDL'
    print('experiment: %s' % experiment_name)
    iterations = 5
    weights_name = 'cnn_1D_filters_ddl.h5'
    numfil=1
    cnn.super_main(experiment_name, iterations, weights_name, numfil)


def experiment_16():
    experiment_name = 'simple CNN with 1D filters, start 32 filters, DDL'
    print('experiment: %s' % experiment_name)
    iterations = 5
    weights_name = 'cnn_1D_filters_ddl.h5'
    numfil=2
    cnn.super_main(experiment_name, iterations, weights_name, numfil)


def main():
    # data loading, so it happens only once
    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.initialize_cnn_data()
    # data = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
    # experiment_6(data)

    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.initialize_cnn_data()
    # data = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
    # experiment_11(data)
    # experiment_12(data)
    #
    # # wait a bit to make sure the weights are saved to file
    # time.sleep(1200)
    # del train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.initialize_scn_data()
    # data = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
    # experiment_13(data)
    # experiment_14(data)
    #
    experiment_15()
    experiment_16()

    # centered_train_data = pre.center(train_data)
    # centered_test_data = pre.center(test_data)
    # centered_validation_data = pre.center(validation_data)
    #
    # normalized_centered_train_data = pre.normalize(centered_train_data)
    # normalized_centered_test_data = pre.normalize(centered_test_data)
    # normalized_centered_validation_data = pre.normalize(centered_validation_data)
    #
    # normalized_train_data = pre.normalize(train_data)
    # normalized_test_data = pre.normalize(test_data)
    # normalized_validation_data = pre.normalize(validation_data)
    #
    # centered_data = [centered_train_data, train_labels, centered_validation_data, validation_labels, centered_test_data, test_labels]
    # normalized_centered_data = [normalized_centered_train_data, train_labels, normalized_centered_validation_data, validation_labels, normalized_centered_test_data, test_labels]
    # normalized_data = [normalized_train_data, train_labels, normalized_validation_data, validation_labels, normalized_test_data, test_labels]

    # running the experiments
    # experiment_0(pu.initialize_cnn_data())
    # experiment_1(centered_data)
    # experiment_2(normalized_centered_data)

    # experiment_5(data)
    # experiment_6(data)

main()