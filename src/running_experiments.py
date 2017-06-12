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


def experiment_ds():
    a = ProjectVariable()
    a.experiment_name = ''
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    # a.datasets = ['caviar', 'grid']
    a.epochs = 10
    a.batch_size = 64
    a.iterations = 1
    scn.super_main(a)


def experiment_92():
    a = ProjectVariable()
    a.experiment_name = '92. baseline: debugged network'
    a.epochs = 40
    scn.super_main(a)


def experiment_92_2():
    a = ProjectVariable()
    a.experiment_name = '92_2. improved baseline: elu + CL 0.00005-0.001'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    scn.super_main(a)


def experiment_93():
    a = ProjectVariable()
    a.experiment_name = '93. euclidean baseline: debugged network'
    a.cost_module_type = 'euclidean'
    a.epochs = 40
    scn.super_main(a)


def experiment_93_2():
    a = ProjectVariable()
    a.experiment_name = '93_2. euclidean improved baseline: elu + CL 0.00005-0.001'
    a.cost_module_type = 'euclidean'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    scn.super_main(a)


def experiment_94():
    a = ProjectVariable()
    a.experiment_name = '94. numfil=1'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    scn.super_main(a)


def experiment_95():
    a = ProjectVariable()
    a.experiment_name = '95. pooling_size=[[4,2],[2,2]]'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.pooling_size= [[4,2],[2,2]]
    scn.super_main(a)


def experiment_96():
    a = ProjectVariable()
    a.experiment_name = '96. neural_distance_layers=(128, 256)'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance_layers = (128, 256)
    scn.super_main(a)


def experiment_97():
    a = ProjectVariable()
    a.experiment_name = '97. kernel=(5,5)'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.kernel = (5, 5)
    a.iterations = 1
    scn.super_main(a)


def experiment_98():
    a = ProjectVariable()
    a.experiment_name = '98. numfil=1 + pooling_size=[[4,2],[2,2]] + neural_distance=(128,256)'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.pooling_size= [[4,2],[2,2]]
    a.neural_distance_layers = (128, 256)
    scn.super_main(a)


def experiment_99():
    a = ProjectVariable()
    a.experiment_name = '99. numfil=1 + pooling_size=[[4,2],[2,2]]'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.pooling_size= [[4,2],[2,2]]
    scn.super_main(a)


def experiment_100():
    a = ProjectVariable()
    a.experiment_name = '100. pooling_size=[[4,2],[2,2]] + neural_distance=(128,256)'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.pooling_size= [[4,2],[2,2]]
    a.neural_distance_layers = (128, 256)
    scn.super_main(a)


def experiment_101():
    a = ProjectVariable()
    a.experiment_name = '101. pooling_type=avg_pooling'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.pooling_type = 'avg_pooling'
    scn.super_main(a)


def experiment_102():
    a = ProjectVariable()
    a.experiment_name = '102. neural_distance=absolute'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_103():
    a = ProjectVariable()
    a.experiment_name = '103. neural_distance=subtract'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'subtract'
    scn.super_main(a)


def experiment_104():
    a = ProjectVariable()
    a.experiment_name = '104. neural_distance=divide'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.neural_distance = 'divide'
    scn.super_main(a)


def experiment_105():
    a = ProjectVariable()
    a.experiment_name = '105. adjustable.numfil = 1 + neural_distance=absolute'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_106():
    a = ProjectVariable()
    a.experiment_name = '106. adjustable.numfil = 1 + pooling_size=[[4,2], [2,2]] +neural_distance=absolute'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.pooling_size = [[4,2],[2,2]]
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_107():
    a = ProjectVariable()
    a.experiment_name = '107. adjustable.numfil = 1 + neural_distance_layers=(128,256) + neural_distance=absolute'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance_layers = (128, 256)
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_108():
    a = ProjectVariable()
    a.experiment_name = '108. adjustable.numfil = 1 + pooling_size=[[4,2],[2,2]]  +neural_distance_layers=(128,256) + neural_distance=absolute'
    a.epochs = 40
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance_layers = (128, 256)
    a.pooling_size = [[4,2],[2,2]]
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_109():
    a = ProjectVariable()
    a.experiment_name = '109. saving network with config. 105 each 10 epochs. preparation for priming experiments'
    a.epochs = 100
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.iterations = 1
    a.neural_distance = 'absolute'
    a.save_inbetween = True
    scn.super_main(a)


def experiment_110():
    a = ProjectVariable()
    a.experiment_name = '110. test only epoch 100'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.only_test = True
    prime.super_main(a)
    

def experiment_111():
    a = ProjectVariable()
    a.experiment_name = '111. priming: test only 105 at epoch 10'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_111_2():
    a = ProjectVariable()
    a.experiment_name = '111_2. priming 105 at epoch 10'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_112():
    a = ProjectVariable()
    a.experiment_name = '112. priming: test only 105 at epoch 20'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_112_2():
    a = ProjectVariable()
    a.experiment_name = '112_2. priming 105 at epoch 20'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_113():
    a = ProjectVariable()
    a.experiment_name = '113. priming: test only 105 at epoch 30'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_113_2():
    a = ProjectVariable()
    a.experiment_name = '113_2. priming 105 at epoch 30'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_114():
    a = ProjectVariable()
    a.experiment_name = '114. priming: test only 105 at epoch 40'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_114_2():
    a = ProjectVariable()
    a.experiment_name = '114_2. priming 105 at epoch 40'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_115():
    a = ProjectVariable()
    a.experiment_name = '115. priming: test only 105 at epoch 50'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_115_2():
    a = ProjectVariable()
    a.experiment_name = '115_2. priming 105 at epoch 50'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_116():
    a = ProjectVariable()
    a.experiment_name = '116. priming: test only 105 at epoch 60'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_116_2():
    a = ProjectVariable()
    a.experiment_name = '116_2. priming 105 at epoch 60'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_117():
    a = ProjectVariable()
    a.experiment_name = '117. priming: test only 105 at epoch 70'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_117_2():
    a = ProjectVariable()
    a.experiment_name = '117_2. priming 105 at epoch 70'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_118():
    a = ProjectVariable()
    a.experiment_name = '118. priming: test only 105 at epoch 80'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_118_2():
    a = ProjectVariable()
    a.experiment_name = '118_2. priming 105 at epoch 80'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_119():
    a = ProjectVariable()
    a.experiment_name = '119. priming: test only 105 at epoch 90'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_119_2():
    a = ProjectVariable()
    a.experiment_name = '119_2. priming 105 at epoch 90'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_120():
    a = ProjectVariable()
    a.experiment_name = '120. priming: test only 105 at epoch 100'
    a.iterations = 1
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.only_test = True
    prime.super_main(a)



def experiment_120_2():
    a = ProjectVariable()
    a.experiment_name = '120_2. priming 105 at epoch 100'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.prime_epochs = 5
    prime.super_main(a)


def experiment_121():
    a = ProjectVariable()
    a.experiment_name = '121. priming 105 at epoch 10, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_122():
    a = ProjectVariable()
    a.experiment_name = '122. priming 105 at epoch 20, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_123():
    a = ProjectVariable()
    a.experiment_name = '123. priming 105 at epoch 30, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_124():
    a = ProjectVariable()
    a.experiment_name = '124. priming 105 at epoch 40, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_125():
    a = ProjectVariable()
    a.experiment_name = '125. priming 105 at epoch 50, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_126():
    a = ProjectVariable()
    a.experiment_name = '126. priming 105 at epoch 60, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_127():
    a = ProjectVariable()
    a.experiment_name = '127. priming 105 at epoch 70, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_128():
    a = ProjectVariable()
    a.experiment_name = '128. priming 105 at epoch 80, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_129():
    a = ProjectVariable()
    a.experiment_name = '129. priming 105 at epoch 90, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_130():
    a = ProjectVariable()
    a.experiment_name = '130. priming 105 at epoch 100, train 3 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.prime_epochs = 3
    prime.super_main(a)


def experiment_131():
    a = ProjectVariable()
    a.experiment_name = '131. priming 105 at epoch 10, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_132():
    a = ProjectVariable()
    a.experiment_name = '132. priming 105 at epoch 20, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_133():
    a = ProjectVariable()
    a.experiment_name = '133. priming 105 at epoch 30, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_134():
    a = ProjectVariable()
    a.experiment_name = '134. priming 105 at epoch 40, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_135():
    a = ProjectVariable()
    a.experiment_name = '135. priming 105 at epoch 50, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_136():
    a = ProjectVariable()
    a.experiment_name = '136. priming 105 at epoch 60, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_137():
    a = ProjectVariable()
    a.experiment_name = '137. priming 105 at epoch 70, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_138():
    a = ProjectVariable()
    a.experiment_name = '138. priming 105 at epoch 80, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_139():
    a = ProjectVariable()
    a.experiment_name = '139. priming 105 at epoch 90, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_140():
    a = ProjectVariable()
    a.experiment_name = '140. priming 105 at epoch 100, train 1 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.prime_epochs = 1
    prime.super_main(a)


def experiment_141():
    a = ProjectVariable()
    a.experiment_name = '141. priming 105 at epoch 10, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_142():
    a = ProjectVariable()
    a.experiment_name = '142. priming 105 at epoch 20, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_143():
    a = ProjectVariable()
    a.experiment_name = '143. priming 105 at epoch 30, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_144():
    a = ProjectVariable()
    a.experiment_name = '144. priming 105 at epoch 40, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_145():
    a = ProjectVariable()
    a.experiment_name = '145. priming 105 at epoch 50, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_146():
    a = ProjectVariable()
    a.experiment_name = '146. priming 105 at epoch 60, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_147():
    a = ProjectVariable()
    a.experiment_name = '147. priming 105 at epoch 70, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_148():
    a = ProjectVariable()
    a.experiment_name = '148. priming 105 at epoch 80, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_149():
    a = ProjectVariable()
    a.experiment_name = '149. priming 105 at epoch 90, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_150():
    a = ProjectVariable()
    a.experiment_name = '150. priming 105 at epoch 100, train 10 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.prime_epochs = 10
    prime.super_main(a)


def experiment_151():
    a = ProjectVariable()
    a.experiment_name = '151. priming 105 at epoch 10, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1522_epoch_10_weights.h5'
    a.load_model_name = 'scnn_05062017_1522_epoch_10_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_152():
    a = ProjectVariable()
    a.experiment_name = '152. priming 105 at epoch 20, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1529_epoch_20_weights.h5'
    a.load_model_name = 'scnn_05062017_1529_epoch_20_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_153():
    a = ProjectVariable()
    a.experiment_name = '153. priming 105 at epoch 30, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1535_epoch_30_weights.h5'
    a.load_model_name = 'scnn_05062017_1535_epoch_30_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_154():
    a = ProjectVariable()
    a.experiment_name = '154. priming 105 at epoch 40, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1542_epoch_40_weights.h5'
    a.load_model_name = 'scnn_05062017_1542_epoch_40_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_155():
    a = ProjectVariable()
    a.experiment_name = '155. priming 105 at epoch 50, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1549_epoch_50_weights.h5'
    a.load_model_name = 'scnn_05062017_1549_epoch_50_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_156():
    a = ProjectVariable()
    a.experiment_name = '156. priming 105 at epoch 60, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1555_epoch_60_weights.h5'
    a.load_model_name = 'scnn_05062017_1555_epoch_60_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_157():
    a = ProjectVariable()
    a.experiment_name = '157. priming 105 at epoch 70, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1602_epoch_70_weights.h5'
    a.load_model_name = 'scnn_05062017_1602_epoch_70_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_158():
    a = ProjectVariable()
    a.experiment_name = '158. priming 105 at epoch 80, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1608_epoch_80_weights.h5'
    a.load_model_name = 'scnn_05062017_1608_epoch_80_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_159():
    a = ProjectVariable()
    a.experiment_name = '159. priming 105 at epoch 90, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1615_epoch_90_weights.h5'
    a.load_model_name = 'scnn_05062017_1615_epoch_90_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_160():
    a = ProjectVariable()
    a.experiment_name = '160. priming 105 at epoch 100, train 20 epochs'
    a.iterations = 5
    a.load_weights_name = 'scnn_05062017_1622_epoch_100_weights.h5'
    a.load_model_name = 'scnn_05062017_1622_epoch_100_model.h5'
    a.prime_epochs = 20
    prime.super_main(a)


def experiment_161():
    a = ProjectVariable()
    a.experiment_name = '161. network 105 train only on viper for 40 epochs'
    a.epochs = 40
    a.datasets = ['viper']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)
    
    
def experiment_162():
    a = ProjectVariable()
    a.experiment_name = '162. network 105 train only on cuhk02 for 40 epochs'
    a.epochs = 40
    a.datasets = ['cuhk02']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)
    
    
def experiment_163():
    a = ProjectVariable()
    a.experiment_name = '163. network 105 train only on market for 40 epochs'
    a.epochs = 40
    a.datasets = ['market']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)
    
    
def experiment_164():
    a = ProjectVariable()
    a.experiment_name = '164. network 105 train only on grid for 40 epochs'
    a.epochs = 40
    a.datasets = ['grid']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_165():
    a = ProjectVariable()
    a.experiment_name = '165. network 105 train only on prid450 for 40 epochs'
    a.epochs = 40
    a.datasets = ['prid450']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_166():
    a = ProjectVariable()
    a.experiment_name = '166. network 105 train only on caviar for 40 epochs. ran with ranking=20 because only 72 IDs'
    a.epochs = 40
    a.datasets = ['caviar']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_171():
    a = ProjectVariable()
    a.experiment_name = '171. network 105 train only on viper for 40 epochs ranking=20'
    a.epochs = 40
    a.datasets = ['viper']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_172():
    a = ProjectVariable()
    a.experiment_name = '172. network 105 train only on cuhk02 for 40 epochs ranking=20'
    a.epochs = 40
    a.datasets = ['cuhk02']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_173():
    a = ProjectVariable()
    a.experiment_name = '173. network 105 train only on market for 40 epochs ranking=20'
    a.epochs = 40
    a.datasets = ['market']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_174():
    a = ProjectVariable()
    a.experiment_name = '174. network 105 train only on grid for 40 epochs ranking=20'
    a.epochs = 40
    a.datasets = ['grid']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_175():
    a = ProjectVariable()
    a.experiment_name = '175. network 105 train only on prid450 for 40 epochs ranking=20'
    a.epochs = 40
    a.datasets = ['prid450']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_176():
    a = ProjectVariable()
    a.experiment_name = '176. network 105 train only on caviar for 40 epochs. ran with ranking=20. again for fun'
    a.epochs = 40
    a.datasets = ['caviar']
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    scn.super_main(a)


def experiment_cos():
    a = ProjectVariable()
    a.experiment_name = 'cosine'
    a.cost_module_type = 'cosine'
    a.epochs = 100
    a.iterations = 1
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.datasets = ['grid']
    scn.super_main(a)


def experiment_order_cuhk02():
    a = ProjectVariable()
    a.experiment_name = 'train on cuhk02'
    a.epochs = 50
    a.iterations = 1
    a.activation_function = 'elu'
    a.cl_min = 0.00005
    a.cl_max = 0.001
    a.numfil = 1
    a.neural_distance = 'absolute'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['cuhk02']
    a.name_indication = 'dataset_name'
    a.batch_size = 32
    scn.super_main(a)


def experiment_order_market():
    a = ProjectVariable()
    a.experiment_name = 'load cuhk02, train on market'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1146_cuhk02_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['market']
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def experiment_order_caviar():
    a = ProjectVariable()
    a.experiment_name = 'load market, train on caviar'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1214_market_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['caviar']
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def experiment_order_viper():
    a = ProjectVariable()
    a.experiment_name = 'load caviar, train on viper'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1232_caviar_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['viper']
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def experiment_order_prid450():
    a = ProjectVariable()
    a.experiment_name = 'load viper, train on prid450'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1241_viper_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['prid450']
    a.name_indication = 'dataset_name'
    scn.super_main(a)

def experiment_order_grid():
    a = ProjectVariable()
    a.experiment_name = 'load prid450, train on grid'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1248_prid450_model.h5' #
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['grid']
    a.name_indication = 'dataset_name'
    scn.super_main(a)

def experiment_test_all_on_order():
    a = ProjectVariable()
    a.experiment_name = 'after training on all data in order test on all ranking test sets'
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1254_grid_model.h5'
    a.only_test = True
    prime.super_main(a)


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
    a.load_model_name = 'scnn_08062017_1339_viper_model.h5' #
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['cuhk02']
    a.name_indication = 'dataset_name'
    scn.super_main(a)


def experiment_ordered_priming():
    a = ProjectVariable()
    a.experiment_name = 'ordered priming on cuhk02'
    a.priming = True
    a.load_model_name = 'scnn_08062017_1400_cuhk02_model.h5'
    a.load_weights_name = 'scnn_08062017_1400_cuhk02_weights.h5'
    a.prime_epochs = 20
    a.iterations = 10
    a.batch_size = 32
    prime.super_main(a)

def experiment_order_market_2():
    a = ProjectVariable()
    a.experiment_name = 'load cuhk02, train on market'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1400_cuhk02_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['market']
    a.name_indication = 'dataset_name'
    scn.super_main(a)

def experiment_ordered_priming_2():
    a = ProjectVariable()
    a.experiment_name = 'ordered priming on market'
    a.priming = True
    a.load_model_name = 'scnn_08062017_1438_market_model.h5'
    a.load_weights_name = 'scnn_08062017_1438_market_weights.h5'
    a.prime_epochs = 2
    a.iterations = 10
    a.batch_size = 32
    prime.super_main(a)

def experiment_order_market_3():
    a = ProjectVariable()
    a.experiment_name = 'load viper, train on market'
    a.epochs = 50
    a.iterations = 1
    a.load_model_name = 'scnn_08062017_1339_viper_model.h5'
    a.save_inbetween = True
    a.save_points = [50]
    a.datasets = ['market']
    a.name_indication = 'dataset_name'
    scn.super_main(a)

def experiment_ordered_priming_3():
    a = ProjectVariable()
    a.experiment_name = 'ordered priming on market'
    a.priming = True
    a.load_model_name = 'scnn_08062017_1535_market_model.h5'
    a.load_weights_name = 'scnn_08062017_1535_market_weights.h5'
    a.prime_epochs = 10
    a.iterations = 10
    a.batch_size = 32
    prime.super_main(a)





def main():
    # TODO run these experiments
    # num = sys.argv[1]
    # print(sys.argv)


    # if num == '173':
    #     experiment_173()
    # if num == '174':
    #     experiment_174()
    # if num == '175':
    #     experiment_175()
    # if num == '176':
    #     experiment_176()
    # experiment_order_market_3()
    experiment_ordered_priming_3()


main()
