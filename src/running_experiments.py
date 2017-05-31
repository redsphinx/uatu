# import cnn_clean as cnn
import siamese_cnn_clean as scn
import sys
from project_variables import ProjectVariable
import priming as prime
import os

#
# def experiment_0(data):
#     # testing stuff
#     experiment_name = 'running the CNN on raw images'
#     print('experiment: %s' %experiment_name)
#     cnn.main(experiment_name, data)
#
#
# def experiment_1(data):
#     experiment_name = 'subtracting the mean image'
#     print('experiment: %s' %experiment_name)
#     cnn.main(experiment_name, data)
#
#
# def experiment_2(data):
#     experiment_name = 'subtracting mean image + normalizing'
#     print('experiment: %s' %experiment_name)
#     cnn.main(experiment_name, data)
#
#
# def experiment_3(data):
#     experiment_name = 'subtracting mean image + PCA whitening'
#     print('experiment: %s' %experiment_name)
#     # [train_data, train_labels, validation_data, validation_labels, test_data, test_labels] = pu.initialize_cnn_data()
#     # train_data = pre.PCA_whiten(pre.center(train_data))
#     # test_data = pre.PCA_whiten(pre.center(test_data))
#     # validation_data = pre.PCA_whiten(pre.center(validation_data))
#     # data = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
#     cnn.main(experiment_name, data)
#
#
# def experiment_4(data):
#     experiment_name = 'normalizing the image'
#     print('experiment: %s' %experiment_name)
#     cnn.main(experiment_name, data)
#
#
# def experiment_5(data):
#     experiment_name = 'batch normalization '
#     print('experiment: %s' % experiment_name)
#     iterations = 3
#     cnn.super_main(experiment_name, data, iterations, do_dropout=False)
#
#
# def experiment_6(data):
#     experiment_name = 'saving model: batch normalization after relu with bias, cyclical learning rate, mode=exp_range'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     cnn.super_main(experiment_name, data, iterations, do_dropout=False)

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


# def experiment_11(data):
#     experiment_name = 'save simple CNN with 1D filters, no BN, no CLR. start with 16 filters.'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     numfil = 1
#     weights_name = 'cnn_1D_filters_16.h5'
#     cnn.super_main(experiment_name, data, iterations, numfil, weights_name)
#
#
# def experiment_12(data):
#     experiment_name = 'save simple CNN with 1D filters, no BN, no CLR. start with 32 filters.'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     numfil = 2
#     weights_name = 'cnn_1D_filters_32.h5'
#     cnn.super_main(experiment_name, data, iterations, numfil, weights_name)


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


# def experiment_15():
#     experiment_name = 'simple CNN with 1D filters, start 16 filters, DDL using HDF5, 5 validation steps per epoch'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     weights_name = 'cnn_1D_filters_ddl.h5'
#     numfil=1
#     cnn.super_main(experiment_name, iterations, weights_name, numfil)
#
#
# def experiment_16():
#     experiment_name = 'saving weights simple CNN with 2D filters, start 32 filters, DDL with HDF5'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
#     numfil = 2
#     cnn.super_main(experiment_name, iterations, weights_name, numfil)


def experiment_17():
    experiment_name = 'training SCN with 2D filters 32 with DDL images'
    print('experiment: %s' % experiment_name)
    iterations = 3
    numfil = 2
    epochs = 5
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# def experiment_18():
#     experiment_name = '18:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.01'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     save_weights = True
#     weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-01.h5'
#     numfil = 2
#     epochs = 10
#     batch_size = 128
#     lr = 0.01
#     cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)
#
# def experiment_19():
#     experiment_name = '19:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.001'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     save_weights = True
#     weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-001.h5'
#     numfil = 2
#     epochs = 10
#     batch_size = 128
#     lr = 0.001
#     cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)
#
# def experiment_20():
#     experiment_name = '20:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.0001'
#     print('experiment: %s' % experiment_name)
#     iterations = 1
#     save_weights = True
#     weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
#     numfil = 2
#     epochs = 10
#     batch_size = 128
#     lr = 0.0001
#     cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)

# ---
# ---

def experiment_21():
    experiment_name = '21:training SCN with CNN weights: 2D filters 32, BN_lr_0-01. lr=0.01'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-01.h5'
    lr = 0.01
    cl = False
    cl_max = None
    cl_min = None
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_max, cl_min, bn)

def experiment_22():
    experiment_name = '22:training SCN with CNN weights: 2D filters 32, BN_lr_0-001. lr=0.001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-001.h5'
    lr = 0.001
    cl = False
    cl_max = None
    cl_min = None
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

def experiment_23():
    experiment_name = '23:training SCN with CNN weights: 2D filters 32, BN_lr_0-0001. lr=0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.0001
    cl = False
    cl_max = None
    cl_min = None
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_23_1():
    experiment_name = '23_1:training SCN with CNN weights: 2D filters 32, BN_lr_0-00001. lr=0.00001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = False
    cl_max = None
    cl_min = None
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ---
# ---

def experiment_24():
    experiment_name = '24:training SCN with CNN weights: 2D filters 32, BN_lr_0-01. use CL 0.001-0.01'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-01.h5'
    lr = 0.001
    cl = True
    cl_min = 0.001
    cl_max = 0.01
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

def experiment_25():
    experiment_name = '25:training SCN with CNN weights: 2D filters 32, BN_lr_0-01. use CL 0.0001-0.01'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-01.h5'
    lr = 0.0001
    cl = True
    cl_min = 0.0001
    cl_max = 0.01
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ---
# ---

def experiment_26():
    experiment_name = '26:training SCN with CNN weights: 2D filters 32, BN_lr_0-001. use CL 0.0001-0.001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-001.h5'
    lr = 0.0001
    cl = True
    cl_min = 0.0001
    cl_max = 0.001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

def experiment_27():
    experiment_name = '27:training SCN with CNN weights: 2D filters 32, BN_lr_0-001. use CL 0.00001-0.001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ---
# ---

def experiment_28():
    experiment_name = '28:training SCN with CNN weights: 2D filters 32, BN_lr_0-0001. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ---
# ---

def experiment_29():
    experiment_name = '29:training SCN with CNN weights: 2D filters 32, no BN. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = False
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

def experiment_30():
    experiment_name = '30:training SCN with CNN weights: 2D filters 32, no BN. use CL 0.00001-0.00005'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.00005
    bn = False
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_30_1():
    experiment_name = '30_1:training SCN with CNN weights: 2D filters 32, no BN. no CL. lr=0.00001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
    lr = 0.00001
    cl = False
    cl_min = 0.00001
    cl_max = 0.00005
    bn = False
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ----
# ----

def experiment_31():
    experiment_name = '31.SCNN: random weight initialization'
    print('experiment: %s' % experiment_name)
    iterations = 1
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)

# ---
# ---

def experiment_32():
    experiment_name = '32.SCNN: RIW, no BN, no CL'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.00001
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = False
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_33():
    experiment_name = '33.SCNN: RIW, BN, no CL, lr=0.00001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.00001
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_34():
    experiment_name = '34.SCNN: RIW, BN, no CL, lr=0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.0001
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_35():
    experiment_name = '35.SCNN: RIW, BN, no CL, lr=0.001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.001
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_36():
    experiment_name = '36.SCNN: RIW, BN, no CL, lr=0.01'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.01
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_37():
    experiment_name = '37.SCNN: RIW, BN, no CL, lr=0.1'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.1
    cl = False
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_38():
    experiment_name = '38.SCNN: RIW, BN, CL, lr=0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_39():
    experiment_name = '39.SCNN: RIW, BN, CL, lr=0.0001-0.001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.0001
    cl = True
    cl_min = 0.0001
    cl_max = 0.001
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_40():
    experiment_name = '40.SCNN: RIW, BN, CL, lr=0.001-0.01'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.001
    cl = True
    cl_min = 0.001
    cl_max = 0.01
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


def experiment_41():
    experiment_name = '41.SCNN: RIW, BN, CL, lr=0.01-0.1'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = ''
    lr = 0.01
    cl = True
    cl_min = 0.01
    cl_max = 0.1
    bn = True
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn)


# -- -- --
# -- -- --

def experiment_42():
    experiment_name = '42.ranking training SCN with CNN weights: 2D filters 32, BN_lr_0-0001. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 1
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)



def experiment_43():
    experiment_name = '43.ranking training SCN with CNN weights: 2D filters 32, BN_lr_0-0001. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 3
    numfil = 2
    epochs = 15
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_44():
    experiment_name = '44.ranking training SCN: 2D filters 32, BN_lr_0-0001. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 3
    numfil = 2
    epochs = 20
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_45():
    experiment_name = '45.ranking training SCN: 2D filters 32, BN_lr_0-0001. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 3
    numfil = 2
    epochs = 30
    batch_size = 64
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_46():
    experiment_name = '46.ranking training SCN: 2D filters 32, use BN. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 15
    batch_size = 64
    weights_name = None
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_47():
    experiment_name = '47.ranking training SCN: 2D filters 32, use BN. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 10
    numfil = 2
    epochs = 20
    batch_size = 64
    weights_name = None
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_48():
    experiment_name = '48.ranking training SCN: 2D filters 32, use BN. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 3
    numfil = 2
    epochs = 25
    batch_size = 64
    weights_name = None
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_49():
    experiment_name = '48.ranking training SCN: 2D filters 32, use BN. use CL 0.00001-0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 5
    numfil = 2
    epochs = 40
    batch_size = 64
    weights_name = None
    lr = 0.00001
    cl = True
    cl_min = 0.00001
    cl_max = 0.0001
    bn = True
    # save_weights_name = 'scnn_settings_exp_28_42_2.h5'
    save_weights_name = None
    scn.super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_min, cl_max, bn,
                   save_weights_name)


def experiment_50():
    a = ProjectVariable()
    a.iterations = 5
    a.experiment_name = '50. experimenting with batch_size=64 vs 256'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 60
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 64
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_51():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '51. debugging'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 1
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 256
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_52():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '52. testing with 40 epochs, batchsize 256'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 40
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 256
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_53():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '53. testing with 40 epochs,batchsize 128'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 40
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 128
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_54():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '54. testing with 40 epochs,batchsize 64'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 40
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 64
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_55():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '55. testing with 40 epochs,batchsize 32'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 40
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 32
    a.scnn_save_weights_name = None
    scn.super_main(a)


def experiment_56():
    a = ProjectVariable()
    a.iterations = 1
    a.experiment_name = '56. testing with 40 epochs,batchsize 16'
    a.numfil = 2
    a.head_type = 'batch_normalized'
    a.cost_module_type = 'neural_network'  # 'euclidean' not implemented yet
    a.neural_distance = 'concatenate'
    a.trainable = True
    a.transfer_weights = False
    a.cnn_weights_name = None
    a.learning_rate = 0.00001
    a.epochs = 40
    a.use_cyclical_learning_rate = True
    a.cl_min = 0.00001
    a.cl_max = 0.0001
    a.batch_size = 16
    a.scnn_save_weights_name = None
    scn.super_main(a)

# base experiment: all other experiments will be compared to this one
def experiment_57():
    a = ProjectVariable()
    a.experiment_name = '57. baseline'
    scn.super_main(a)

# neural distance
def experiment_58():
    a = ProjectVariable()
    a.experiment_name = '58. neural distance: add'
    a.neural_distance = 'add'
    scn.super_main(a)


def experiment_59():
    a = ProjectVariable()
    a.experiment_name = '59. neural distance: multiply'
    a.neural_distance = 'multiply'
    scn.super_main(a)

# neural distance layers
def experiment_60():
    a = ProjectVariable()
    a.experiment_name = '60. neural distance layer: (8192, 1024)'
    a.neural_distance_layers = (8192, 1024)
    scn.super_main(a)


def experiment_61():
    a = ProjectVariable()
    a.experiment_name = '61. neural distance layer: (4096, 1024)'
    a.neural_distance_layers = (4096, 1024)
    scn.super_main(a)


def experiment_62():
    a = ProjectVariable()
    a.experiment_name = '62. neural distance layer: (1024, 512)'
    a.neural_distance_layers = (1024, 512)
    scn.super_main(a)


def experiment_62_2():
    a = ProjectVariable()
    a.experiment_name = '62_2. neural distance layer: (512, 256)'
    a.neural_distance_layers = (512, 256)
    scn.super_main(a)


def experiment_62_3():
    a = ProjectVariable()
    a.experiment_name = '62_3. neural distance layer: (128, 256)'
    a.neural_distance_layers = (128, 256)
    scn.super_main(a)


def experiment_63():
    a = ProjectVariable()
    a.experiment_name = '63. neural distance layer: (8192, 128)'
    a.neural_distance_layers = (8192, 128)
    scn.super_main(a)


def experiment_64():
    a = ProjectVariable()
    a.experiment_name = '64. neural distance layer: (8192, 4096)'
    a.neural_distance_layers = (8192, 4096)
    scn.super_main(a)

# max pool
def experiment_65():
    a = ProjectVariable()
    a.experiment_name = '65. max pooling size: [[4,2], [2,2]]'
    a.max_pooling_size = [[4,2], [2,2]]
    scn.super_main(a)

# activation type
def experiment_66():
    a = ProjectVariable()
    a.experiment_name = '66. activation type: elu'
    a.activation_function = 'elu'
    scn.super_main(a)

# loss function
def experiment_67():
    a = ProjectVariable()
    a.experiment_name = '67. loss function: kullback_leibler_divergence'
    a.loss_function = 'kullback_leibler_divergence'
    scn.super_main(a)


def experiment_68():
    a = ProjectVariable()
    a.experiment_name = '68. loss function: mean_squared_error'
    a.loss_function = 'mean_squared_error'
    scn.super_main(a)


def experiment_69():
    a = ProjectVariable()
    a.experiment_name = '69. loss function: mean_absolute_error'
    a.loss_function = 'mean_absolute_error'
    scn.super_main(a)


def experiment_70():
    a = ProjectVariable()
    a.experiment_name = '70. pooling type: avg_pooling'
    a.pooling_type = 'avg_pooling'
    scn.super_main(a)


def experiment_71():
    a = ProjectVariable()
    a.experiment_name = '71. pooling type: avg_pooling + pooling size: [[4,2], [2,2]]'
    a.pooling_type = 'avg_pooling'
    a.max_pooling_size = [[4, 2], [2, 2]]
    scn.super_main(a)


def experiment_72():
    a = ProjectVariable()
    a.experiment_name = '72. combo: avg_pooling, elu, dif pooling size, dif neural distance layers'
    a.pooling_type = 'avg_pooling'
    a.activation_function = 'elu'
    a.pooling_size = [[4, 2], [2, 2]]
    a.neural_distance_layers = (256, 128)
    scn.super_main(a)


def experiment_73():
    a = ProjectVariable()
    a.experiment_name = '73. combo: avg_pooling, elu, dif pooling size'
    a.pooling_type = 'avg_pooling'
    a.activation_function = 'elu'
    a.pooling_size = [[4, 2], [2, 2]]
    scn.super_main(a)


def experiment_74():
    a = ProjectVariable()
    a.experiment_name = '74. no cyclical learning rate'
    a.activation_function = 'elu'
    a.use_cyclical_learning_rate = False
    scn.super_main(a)
    

def experiment_75():
    a = ProjectVariable()
    a.experiment_name = '75. combo: elu + dif pooling size'
    a.activation_function = 'elu'
    a.pooling_size = [[4, 2], [2, 2]]
    scn.super_main(a)


def experiment_76():
    a = ProjectVariable()
    a.experiment_name = '76. combo: elu + numfil=1'
    a.activation_function = 'elu'
    a.numfil = 1
    scn.super_main(a)


def experiment_77():
    a = ProjectVariable()
    a.experiment_name = '77. combo: elu + no BN'
    a.activation_function = 'elu'
    a.head_type = 'simple'
    scn.super_main(a)


def experiment_78():
    a = ProjectVariable()
    a.experiment_name = '78. combo: elu + CLR min=0.00005, max=0.0005'
    a.activation_function = 'elu'
    scn.super_main(a)


def experiment_78_2():
    a = ProjectVariable()
    a.experiment_name = '78_2. elu + CLR min=0.00005, max=0.0005'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    scn.super_main(a)


def experiment_79():
    a = ProjectVariable()
    a.experiment_name = '79. elu + avg pooling'
    a.activation_function = 'elu'
    a.pooling_type = 'avg_pooling'
    scn.super_main(a)

def experiment_80():
    a = ProjectVariable()
    a.experiment_name = '80. elu + avg pooling + CLR min=0.00005, max=0.0005'
    a.activation_function = 'elu'
    a.pooling_type = 'avg_pooling'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    scn.super_main(a)


def experiment_81():
    a = ProjectVariable()
    a.experiment_name = '81. elu + avg pooling + pooling size [[4,2],[2,2]]'
    a.activation_function = 'elu'
    a.pooling_type = 'avg_pooling'
    a.pooling_size = [[4,2],[2,2]]
    scn.super_main(a)


def experiment_82():
    a = ProjectVariable()
    a.experiment_name = '82. elu + neural distance layer: (128, 256)'
    a.activation_function = 'elu'
    a.neural_distance_layers = (128, 256)
    scn.super_main(a)


def experiment_83():
    a = ProjectVariable()
    a.experiment_name = '83. elu + numfil=1 + pooling size [[4,2],[2,2]]'
    a.activation_function = 'elu'
    a.pooling_size = [[4,2],[2,2]]
    a.numfil = 1
    scn.super_main(a)


def experiment_84():
    a = ProjectVariable()
    a.experiment_name = '84. elu + numfil=1 + neural distance layer: (128, 256)'
    a.activation_function = 'elu'
    a.neural_distance_layers = (128, 256)
    a.numfil = 1
    scn.super_main(a)


def experiment_85():
    a = ProjectVariable()
    a.experiment_name = '85. elu + CLR min=0.00005 max=0.0005 + test=100'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    scn.super_main(a)



def experiment_85_2():
    a = ProjectVariable()
    a.experiment_name = '85_2. elu + CLR min=0.00005 max=0.0005 + test=100 + epoch=60'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 60
    scn.super_main(a)

def experiment_85_3():
    a = ProjectVariable()
    a.experiment_name = '85_3. elu + CLR min=0.00005 max=0.0005 + test=100 + epoch=100'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 100
    scn.super_main(a)


def experiment_85_4():
    a = ProjectVariable()
    a.experiment_name = '85_4. elu + CLR min=0.00005 max=0.0005 + test=100 + epoch=100'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.0005
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 100
    a.batch_size = 32
    scn.super_main(a)


def experiment_85_5():
    a = ProjectVariable()
    a.experiment_name = '85_5. elu + CLR min=0.00005 max=0.001 + test=100 + epoch=100'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 100
    a.batch_size = 32
    scn.super_main(a)

def experiment_85_6():
    a = ProjectVariable()
    a.experiment_name = '85_6. elu + CLR min=0.0001 max=0.001 + test=100 + epoch=100'
    a.activation_function = 'elu'
    a.cl_min = 0.0001
    a.cl_max = 0.001
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 100
    a.batch_size = 32
    scn.super_main(a)


def experiment_86():
    a = ProjectVariable()
    a.experiment_name = '86. saving the model for priming, 40 epochs rankingnumber=20'
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450']
    a.epochs = 40
    a.batch_size = 32
    a.iterations = 1
    a.scnn_save_model_name = 'scn_86_model_20_40ep.h5'
    a.scnn_save_weights_name = 'scn_86_weights_20_40ep.h5'
    scn.super_main(a)


def experishit(test_number):
    a = ProjectVariable()
    a.use_gpu = str(test_number)
    scn.super_main(a)


def thing():
    a = ProjectVariable()
    a.datasets = ['viper', 'grid', 'prid450', 'caviar']
    scn.super_main(a)


def experiment_priming():
    a = ProjectVariable()
    a.experiment_name = 'debugging priming - clmin=0.00005 clmax=0.001 5 epoch '
    a.priming = True
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.load_weights_name = 'scn_86_weights_20_40ep.h5'
    a.prime_epochs = 5
    a.batch_size = 1
    prime.main(a)


def experiment_87():
    a = ProjectVariable()
    a.experiment_name = '87. baseline (testing only)'
    a.priming = True
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.only_test = True
    a.iterations = 1
    prime.super_main(a)


def experiment_88():
    a = ProjectVariable()
    a.experiment_name = '88. epoch=1'
    a.priming = True
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.load_weights_name = 'scn_86_weights_20_40ep.h5'
    a.prime_epochs = 1
    a.batch_size = 1
    a.iterations = 5
    prime.super_main(a)


def experiment_89():
    a = ProjectVariable()
    a.experiment_name = '89. epoch=3'
    a.priming = True
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.load_weights_name = 'scn_86_weights_20_40ep.h5'
    a.prime_epochs = 3
    a.batch_size = 1
    a.iterations = 5
    prime.super_main(a)


def experiment_90():
    a = ProjectVariable()
    a.experiment_name = '90. epoch=5'
    a.priming = True
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.load_weights_name = 'scn_86_weights_20_40ep.h5'
    a.prime_epochs = 5
    a.batch_size = 1
    a.iterations = 5
    prime.super_main(a)


def experiment_91():
    a = ProjectVariable()
    a.experiment_name = '91. epoch=7'
    a.priming = True
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.load_weights_name = 'scn_86_weights_20_40ep.h5'
    a.prime_epochs = 7
    a.batch_size = 1
    a.iterations = 5
    prime.super_main(a)


def experiment_priming_only_test():
    a = ProjectVariable()
    a.experiment_name = 'debug priming rewrite: only testing'
    a.priming = True
    a.load_model_name = 'scn_86_model_20_40ep.h5'
    a.only_test = True
    prime.main(a)

def main():
    # TODO run these experiments
    num = sys.argv[1]
    print(sys.argv)

    if num == '87':
        experiment_87()

    if num == '88':
        experiment_88()

    if num == '89':
        experiment_89()

    if num == '90':
        experiment_90()

    if num == '91':
        experiment_91()


main()
