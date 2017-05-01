import cnn_clean as cnn
import siamese_cnn_clean as scn
# import time
from pympler.tracker import SummaryTracker
from numba import cuda
from numba.cuda.cudadrv.driver import Device
from numba.cuda.cudadrv.devices import reset
import sys


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
    experiment_name = 'simple CNN with 1D filters, start 16 filters, DDL using HDF5, 5 validation steps per epoch'
    print('experiment: %s' % experiment_name)
    iterations = 1
    weights_name = 'cnn_1D_filters_ddl.h5'
    numfil=1
    cnn.super_main(experiment_name, iterations, weights_name, numfil)


def experiment_16():
    experiment_name = 'saving weights simple CNN with 2D filters, start 32 filters, DDL with HDF5'
    print('experiment: %s' % experiment_name)
    iterations = 1
    weights_name = 'cnn_2D_32_filter_ddl_hdf5.h5'
    numfil = 2
    cnn.super_main(experiment_name, iterations, weights_name, numfil)


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

def experiment_18():
    experiment_name = '18:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.01'
    print('experiment: %s' % experiment_name)
    iterations = 1
    save_weights = True
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-01.h5'
    numfil = 2
    epochs = 10
    batch_size = 128
    lr = 0.01
    cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)

def experiment_19():
    experiment_name = '19:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.001'
    print('experiment: %s' % experiment_name)
    iterations = 1
    save_weights = True
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-001.h5'
    numfil = 2
    epochs = 10
    batch_size = 128
    lr = 0.001
    cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)

def experiment_20():
    experiment_name = '20:saving weights simple CNN with 2D filters, start 32 filters, BatchNorm, lr=0.0001'
    print('experiment: %s' % experiment_name)
    iterations = 1
    save_weights = True
    weights_name = 'cnn_2D_32_filter_ddl_hdf5_BN_lr_0-0001.h5'
    numfil = 2
    epochs = 10
    batch_size = 128
    lr = 0.0001
    cnn.super_main(experiment_name, iterations, weights_name, numfil, epochs, batch_size, lr, save_weights=save_weights)

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


def main():
    num = sys.argv[1]
    print(sys.argv)
    
    if num == "23_1":
        experiment_23_1()

    if num == "30_1":
        experiment_30_1()

main()
