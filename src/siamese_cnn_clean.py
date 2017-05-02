import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, Input, Lambda, BatchNormalization
from keras import optimizers
import dynamic_data_loading as ddl
from keras import backend as K
import project_constants as pc
import project_utils as pu
# import os
# import numpy as np
# import time
import h5py
from clr_callback import *
import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# #
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def gpu_memory():
    out = os.popen("nvidia-smi").read()
    ret = '0MiB'
    for item in out.split("\n"):
        if str(os.getpid()) in item and 'python' in item:
            ret = item.strip().split(' ')[-2]
    return float(ret[:-3])

# gpu_mem = []
# gpu_mem.append(gpu_memory())


def alt_create_fc(inputs):
    # dense_layer = Dense(512)(inputs)
    # norm = BatchNormalization()(dense_layer)
    # dense_layer = Dense(1024)(norm)
    # norm = BatchNormalization()(dense_layer)
    # output_layer = Dense(pc.NUM_CLASSES)(norm)
    # softmax = Activation('softmax')(output_layer)
    # return softmax
    dense_layer = Dense(512) (inputs)
    activation = Activation('relu') (dense_layer)
    dropout_layer = Dropout(pc.DROPOUT)(activation)
    dense_layer = Dense(1024)(dropout_layer)
    activation = Activation('relu')(dense_layer)
    dropout_layer = Dropout(pc.DROPOUT)(activation)
    output_layer = Dense(pc.NUM_CLASSES) (dropout_layer)
    softmax = Activation('softmax')(output_layer)
    return  softmax


def add_activation_and_relu(model):
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    return model


def create_base_network_simple(numfil, weights_name):
    model = Sequential()
    model.add(Conv2D(16 * numfil, kernel_size=(3, 3), padding='same',
                     input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH,
                                  pc.NUM_CHANNELS), name='conv_1', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(32*numfil, kernel_size=(3, 3), padding='same', name='conv_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(64*numfil, kernel_size=(3, 3), padding='same', name='conv_3', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(128*numfil, kernel_size=(3, 3), padding='same', name='conv_4', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(256*numfil, kernel_size=(3, 3), padding='same', name='conv_5', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(512*numfil, kernel_size=(3, 3), padding='same', name='conv_6', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(Conv2D(1024*numfil, kernel_size=(3, 3), padding='same', name='conv_7', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name), by_name=True)

    return model


def create_base_network_simple_BN(numfil, weights_name):
    if pc.TRANSFER_LEARNING:
        train = False
    else:
        train = True
    model = Sequential()
    model.add(Conv2D(16 * numfil, kernel_size=(3, 3), padding='same',
                     input_shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH,
                                  pc.NUM_CHANNELS), name='conv_1', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_1', trainable=train))
    model.add(Conv2D(32*numfil, kernel_size=(3, 3), padding='same', name='conv_2', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_2', trainable=train))
    model.add(Conv2D(64*numfil, kernel_size=(3, 3), padding='same', name='conv_3', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_3', trainable=train))
    model.add(Conv2D(128*numfil, kernel_size=(3, 3), padding='same', name='conv_4', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_4', trainable=train))
    model.add(Conv2D(256*numfil, kernel_size=(3, 3), padding='same', name='conv_5', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_5', trainable=train))
    model.add(Conv2D(512*numfil, kernel_size=(3, 3), padding='same', name='conv_6', trainable=train))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_6', trainable=train))
    model.add(Conv2D(1024*numfil, kernel_size=(3, 3), padding='same', name='conv_7', trainable=train))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_7', trainable=train))
    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name), by_name=True)

    # gpu_mem.append(gpu_memory())
    return model



def create_base_network_1d_filter(train_data_, numfil, weights_name):
    model = Sequential()
    model.add(Conv2D(16*numfil, kernel_size=(1, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1_1',
                     trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(16*numfil, kernel_size=(3, 1), padding='same', name='conv_1_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(32*numfil, kernel_size=(1, 3), padding='same', name='conv_2_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(32*numfil, kernel_size=(3, 1), padding='same', name='conv_2_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(64*numfil, kernel_size=(1, 3), padding='same', name='conv_3_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(64*numfil, kernel_size=(3, 1), padding='same', name='conv_3_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(128*numfil, kernel_size=(1, 3), padding='same', name='conv_4_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(128*numfil, kernel_size=(3, 1), padding='same', name='conv_4_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(256*numfil, kernel_size=(1, 3), padding='same', name='conv_5_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(256*numfil, kernel_size=(3, 1), padding='same', name='conv_5_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(512*numfil, kernel_size=(1, 3), padding='same', name='conv_6_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(512*numfil, kernel_size=(3, 1), padding='same', name='conv_6_2', trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)

    model.add(Conv2D(1024*numfil, kernel_size=(1, 3), padding='same', name='conv_7_1', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(Conv2D(1024*numfil, kernel_size=(3, 1), padding='same', name='conv_7_2', trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))

    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, weights_name),
                           by_name=True)

    return model


def create_base_network_with_BN(train_data_):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, 3), padding='same', input_shape=train_data_.shape[1:], name='conv_1_1', 
                     use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_1', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(16, kernel_size=(3, 1), padding='same', name='conv_1_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_2', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(32, kernel_size=(1, 3), padding='same', name='conv_2_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_3', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(32, kernel_size=(3, 1), padding='same', name='conv_2_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_4', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(64, kernel_size=(1, 3), padding='same', name='conv_3_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_5', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='same', name='conv_3_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_6', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(128, kernel_size=(1, 3), padding='same', name='conv_4_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_7', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(128, kernel_size=(3, 1), padding='same', name='conv_4_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_8', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(256, kernel_size=(1, 3), padding='same', name='conv_5_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_9', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(256, kernel_size=(3, 1), padding='same', name='conv_5_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_10', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(512, kernel_size=(1, 3), padding='same', name='conv_6_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_11', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(512, kernel_size=(3, 1), padding='same', name='conv_6_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model = add_activation_and_relu(model)
    model.add(BatchNormalization(name='bn_12', trainable=pc.TRAIN_CNN))

    model.add(Conv2D(1024, kernel_size=(1, 3), padding='same', name='conv_7_1', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_13', trainable=pc.TRAIN_CNN))
    model.add(Conv2D(1024, kernel_size=(3, 1), padding='same', name='conv_7_2', use_bias=pc.USE_BIAS, trainable=pc.TRAIN_CNN))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='bn_14', trainable=pc.TRAIN_CNN))

    model.add(Flatten(name='cnn_flat'))

    if pc.TRANSFER_LEARNING:
        model.load_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, 'cnn_model_weights_bn_clr_1_bias.h5'), by_name=True)

    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_siamese(numfil, weights_name, bn, similarity_metric='fc_layers'):
    input_a = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    input_b = Input(shape=(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    if bn:
        base_network = create_base_network_simple_BN(numfil, weights_name)
    else:
        base_network = create_base_network_simple(numfil, weights_name)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    if similarity_metric == 'fc_layers':
        # use a fc to come up with a metric
        merged_processed = keras.layers.concatenate([processed_a, processed_b])
        distance = alt_create_fc(merged_processed)
        model = Model([input_a, input_b], distance)
    elif similarity_metric == 'euclid':
        # useful to show drawback of euclidean distance as a metric
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)

    return model


def confusion_matrix(name, predictions, labels, verbose=False):
    matrix = pu.make_confusion_matrix(predictions, labels)
    if verbose:
        pu.print_confusion_matrix(name, matrix)
    return matrix


def main(experiment_name, weights_name, numfil, epochs, batch_size, lr, cl, cl_max, cl_min, bn, save_weights_name,
         similarity_metric='fc_layers', ranking=True):

    total_data_list_pos = '/home/gabi/PycharmProjects/uatu/data/reid_all_positives.txt'
    file_data_list_neg = '/home/gabi/PycharmProjects/uatu/data/reid_all_negatives_uncompressed.h5'

    total_data_list_pos = np.genfromtxt(total_data_list_pos, dtype=None)
    with h5py.File(file_data_list_neg, 'r') as hf:
        total_data_list_neg = hf['data'][()]

    val_list, test_list_viper, test_list_cuhk, total_data_list_pos, total_data_list_neg = ddl.make_validation_test_list(total_data_list_pos,
                                                                                                  total_data_list_neg,
                                                                                                  val_pos_percent=0.1,
                                                                                                  test_pos_percent=0.1,
                                                                                                  data_type='images',
                                                                                                  ranking=True)

    model = create_siamese(numfil, weights_name, bn)
    start = time.time()
    validation_data, validation_labels = ddl.load_in_array(data_list=val_list,
                                                           data_type='images',
                                                           heads=2)
    print('Time loading validation data: %0.3f seconds' % (time.time() - start))
    start = time.time()
    test_data_viper, test_labels_viper = ddl.load_in_array(data_list=test_list_viper,
                                                           data_type='images',
                                                           heads=2)

    test_data_cuhk, test_labels_cuhk = ddl.load_in_array(data_list=test_list_cuhk,
                                                           data_type='images',
                                                           heads=2)
    
    print('Time loading test data: %0.3f seconds' % (time.time() - start))

    slice_size = 5000
    train_data_size = 2 * min(len(total_data_list_pos), len(total_data_list_neg))
    num_steps_per_epoch = np.ceil(train_data_size * 1.0 / slice_size).astype(int)

    num_validations = 1
    if num_validations > num_steps_per_epoch: num_validations = num_steps_per_epoch
    validation_interval = np.floor(num_steps_per_epoch / num_validations).astype(int)
    print('validation happens every %d step(s)' % validation_interval)



    if similarity_metric == 'fc_layers':
        nadam = optimizers.Nadam(lr=lr, schedule_decay=pc.DECAY_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        # print('---!!!--- %s' % (str(cuda.detect())))
        # model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
        #           batch_size=pc.BATCH_SIZE,
        #           epochs=pc.NUM_EPOCHS,
        #           validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
        #           verbose=2)

        for epoch in xrange(epochs):
            print('------EPOCH: %d' % epoch)
            slice_size_queue = ddl.make_slice_queue(train_data_size, slice_size)

            total_train_data_list = ddl.make_train_batches(total_data_list_pos, total_data_list_neg, data_type='images')
            for step in xrange(num_steps_per_epoch):
                # print('..epoch %d step: %d out of %d' % (epoch, step, num_steps_per_epoch))
                train_data_list = total_train_data_list[step * slice_size : step * slice_size + slice_size_queue[step]]

                start = time.time()
                train_data, train_labels = ddl.load_in_array(data_list=train_data_list,
                                                             heads=2,
                                                             data_type='images')
                # print('Time loading training data: %0.3f seconds' % (time.time() - start))
                # let validation happen every x steps

                if cl:
                    clr = CyclicLR(step_size=(np.shape(train_data)[0]/batch_size)*8, base_lr=cl_min,
                                              max_lr=cl_max)
                    if step % validation_interval == 0:
                        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                                  batch_size=batch_size,
                                  epochs=1,
                                  validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                                  verbose=2,
                                  callbacks=[clr])
                    else:
                        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                                  batch_size=batch_size,
                                  epochs=1,
                                  verbose=0,
                                  callbacks=[clr])
                else:
                    if step % validation_interval == 0:
                        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                                  batch_size=batch_size,
                                  epochs=1,
                                  validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels),
                                  verbose=2)
                    else:
                        model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                                  batch_size=batch_size,
                                  epochs=1,
                                  verbose=0)


    elif similarity_metric == 'euclid':
        # rms = RMSprop()
        # model.compile(loss=contrastive_loss, optimizer=rms)
        # model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
        #           batch_size=pc.BATCH_SIZE,
        #           epochs=pc.NUM_EPOCHS,
        #           validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_labels))
        pass


    
    te_pred_viper = model.predict([test_data_viper[:, 0], test_data_viper[:, 1]])
    te_pred_cuhk = model.predict([test_data_cuhk[:, 0], test_data_cuhk[:, 1]])
    
    te_matrix_viper = confusion_matrix('Testing', te_pred_viper, test_labels_viper)
    te_matrix_cuhk = confusion_matrix('Testing', te_pred_cuhk, test_labels_cuhk)

    # delete objects else we run out of memory
    accuracy_viper = (te_matrix_viper[0] + te_matrix_viper[2]) * 1.0 / (sum(te_matrix_viper) * 1.0)
    precision_viper = (te_matrix_viper[0] * 1.0 / (te_matrix_viper[0]+te_matrix_viper[1] * 1.0))

    accuracy_cuhk = (te_matrix_cuhk[0] + te_matrix_cuhk[2]) * 1.0 / (sum(te_matrix_cuhk) * 1.0)
    precision_cuhk = (te_matrix_cuhk[0] * 1.0 / (te_matrix_cuhk[0] + te_matrix_cuhk[1] * 1.0))
    
    
    print('VIPeR: accuracy = %0.2f, precision = %0.2f, confusion matrix = %s' %(accuracy_viper, precision_viper, 
                                                                                str(te_matrix_viper)))
    print('CUHK: accuracy = %0.2f, precision = %0.2f, confusion matrix = %s' % (accuracy_cuhk, precision_cuhk,
                                                                                 str(te_matrix_cuhk)))
    te_matrix = [te_matrix_viper, te_matrix_cuhk]

    if ranking:
        datasets = ['viper', 'cuhk']
        rankings = []
        for item in datasets:
            ranking_matrix_abs = np.zeros((pc.RANKING_NUMBER, pc.RANKING_NUMBER))

            tmp = te_pred_viper[:, 1] if item == 'viper' else te_pred_cuhk[:, 1]
            
            ranking_matrix_probs = np.reshape(tmp, (pc.RANKING_NUMBER, pc.RANKING_NUMBER))
            rank_range = np.zeros(pc.RANKING_NUMBER)
            for row in range(len(ranking_matrix_probs)):
                ranking_matrix_abs[row] = [i[0] for i in sorted(enumerate(ranking_matrix_probs[row]), key=lambda x:x[1],
                                                                reverse=True)]
                list_form = ranking_matrix_abs[row].tolist()
                num = list_form.index(row)
                rank_range[num] += 1
            
            final_ranking = []
            for tallies in range(len(rank_range)):
                # print(tallies)
                percentage = sum(rank_range[0:tallies+1])*1.0 / sum(rank_range)*1.0
                final_ranking.append(percentage)
            print('FINAL RANKING %s: ' % item)
            print(final_ranking)
            rankings.append(final_ranking)
    else:
        rankings = None


    if not save_weights_name == None:
        model.save_weights(os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, save_weights_name))

    del model
    return te_matrix, rankings


def super_main(experiment_name, iterations, numfil, weights_name, epochs, batch_size, lr, cl, cl_max, cl_min, bn,
               save_weights_name):
    
    viper_matrix = np.zeros((iterations, 4))
    viper_ranking = np.zeros((iterations, pc.RANKING_NUMBER))
    cuhk_matrix = np.zeros((iterations, 4))
    cuhk_ranking = np.zeros((iterations, pc.RANKING_NUMBER))
    
    start = time.time()
    for iter in range(0, iterations):
        print('-----ITERATION %d' % iter)
        matrix, ranking = main(experiment_name, weights_name, numfil, epochs, batch_size, lr, cl, cl_max, cl_min, bn,
                          save_weights_name)
        
        viper_matrix[iter] = matrix[0]
        viper_ranking[iter] = ranking[0]
        cuhk_matrix[iter] = matrix[1]
        cuhk_ranking[iter] = ranking[1]
        
    stop = time.time()

    total_time = stop - start

    print('viper\nTP    FP    TN    FN')
    print(viper_matrix)
    
    print('viper\nRANKING')
    print(viper_ranking)
    
    print('cuhk\nTP    FP    TN    FN')
    print(cuhk_matrix)

    print('cuhk\nRANKING')
    print(cuhk_ranking)

    viper_matrix_mean = np.mean(viper_matrix, axis=0)
    print('viper matrix mean values:' + str(viper_matrix_mean))
    
    viper_ranking_mean = np.mean(viper_ranking, axis=0)
    print('viper ranking mean values:' + str(viper_ranking_mean))

    cuhk_matrix_mean = np.mean(cuhk_matrix, axis=0)
    print('cuhk matrix mean values:' + str(cuhk_matrix_mean))

    cuhk_ranking_mean = np.mean(cuhk_ranking, axis=0)
    print('cuhk ranking mean values:' + str(cuhk_ranking_mean))
    

    mean = [str(viper_matrix_mean), str(viper_ranking_mean), str(cuhk_matrix_mean), str(cuhk_ranking_mean)]


    # note: TURN ON if you want to log results!!
    if pc.LOGGING:
        file_name = os.path.basename(__file__)
        dataset_name = 'VIPeR, CUHK1'
        pu.enter_in_log(experiment_name, file_name, iterations, mean, dataset_name, total_time)


